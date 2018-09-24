
# coding: utf-8

# In[62]:


import os
import sys
import regex as re
import logging
import datetime

level = 'INFO' #'ERROR'
logging.basicConfig(level=level)

import requests
from bs4 import BeautifulSoup

import numpy as np
import nltk as nl
import pandas as pd
import sklearn as sk

from sqlalchemy import create_engine, Column, Integer, BigInteger, String, DateTime
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import FetchedValue
Base = declarative_base()

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# In[1]:


# Derived from https://github.com/CRutkowski/Kijiji-Scraper
def parse_ad_summary(html):
    '''
    Parses several fields of a listing summary (except the description, which is truncated),
    then returns a dict of several these fields
    '''
    ad_info = {}
    
    try:
        ad_info['title'] = html.find('a', {'class': 'title'}).text.strip()
    except:
        logging.error('Unable to parse Title data.')

    try:
        ad_info['img'] = str(html.find('img'))
    except:
        logging.error('Unable to parse Image data')

    try:
        ad_info['url'] = 'http://www.kijiji.ca' + html.get('data-vip-url')
    except:
        logging.error('Unable to parse URL data.')

    try:
        ad_info['details'] = html.find('div', {'class': 'details'}).text.strip()
    except:
        logging.error('Unable to parse Details data.')

    try:
        ad_info['date'] = html.find('span', {'class': 'date-posted'}).text.strip()
    except:
        logging.error('Unable to parse Date data.')

    # The location field is affixed with the date for some reason
    try:
        location = html.find('div', {'class': 'location'}).text.strip()
        location = location.replace(ad_info['date'], '')
        ad_info['location'] = location
    except:
        logging.error('Unable to parse Location data.')

    # In addition to normal prices, there can be 'Free' ($0), 'Please Contact' ($-1),
    # some other value ($-2), or empty ($-3)
    try:
        price = html.find('div', {'class': 'price'}).text.strip()
        if price[0] == '$':
            price = price[1:]
        elif price == 'Please Contact':
            price = '-1.00'
        elif price == 'Free':
            price = '0'
        else:
            price = '-2.00'
        ad_info['price'] = price
    except:
        logging.error('Unable to parse Price data.')
        ad_info['price'] = '-3.00'

    return ad_info

def parse_ad_page(url):
    '''Parses the description from an ad page'''
    try:
        page = requests.get(url)
    except:
        print('[Error] Unable to load ' + url)
        return ''
    
    soup = BeautifulSoup(page.content, 'html.parser')
    
    try:
        return soup.find('div', {'itemprop': 'description'}).get_text()
    except:
        logging.error('Unable to parse ad description.')

def scrape(subject, existing_ad_ids = None, limit = None, url = None):
    '''
    Args are a search string for some subject, a list of existing ad IDs to skip,
    a limit to the number of results to return, and a url from which to start
    (in case, for example, a previous scrape was halted and is now being resumed).
    Returns (ads, all_ads_scraped); all_ads_scraped is whether or not an imposed limit was reached
    '''
    # Initialize variables for loop
    ad_dict = {}
    # Shallow copy for now
    ad_ids_to_skip = existing_ad_ids if existing_ad_ids is not None else set()

    if url is None: url =         f'https://www.kijiji.ca/b-city-of-toronto/{subject}/k0l1700273?ll=43.650843,-79.377573&dc=true'

    ads_parsed = 0
    
    while url:

        try:
            page = requests.get(url)
        except:
            logger.error(f'[Error] Unable to load {url}')
            return

        soup = BeautifulSoup(page.content, 'html.parser')

        ads = soup.find_all('div', {'class': 'regular-ad'})
        
        # Skip third-party ads; these third parties are typically small retailers
        third_party_ads = soup.find_all('div', {'class': 'third-party'})
        for ad in third_party_ads:
            ad_ids_to_skip.add(int(ad['data-ad-id']))

        # Parse ads until the limit is reached
        for ad in kijiji_ads:
            title = ad.find('a', {'class': 'title'}).text.strip()
            ad_id = int(ad['data-ad-id'])
            
            if ad_id not in ad_ids_to_skip:
                logging.info(f'New ad found! Ad id: {ad_id}')
                ad_info = parse_ad_summary(ad)
                ad_url = ad_info['url']
                ad_info['description'] = parse_ad_page(ad_url)
                ad_dict[ad_id] = ad_info
                if limit is not None:
                    ads_parsed += 1
                    if ads_parsed >= limit: return (ad_dict, url)
            else:
                logging.debug('Skip ad')
                
        url = soup.find('a', {'title' : 'Next'})
        if url:
            url_path = url['href']
            url = f'https://www.kijiji.ca{url_path}'

    return ad_dict if limit is None else (ad_dict, None)


# In[4]:


# Database models
class Listing(Base):
    __tablename__ = 'listings'
    
    def __init__(self, id, **kwargs):
        self.id = id
        self.set_attributes(kwargs)

    def set_attributes(self,kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    id = Column(BigInteger, primary_key=True)
    title = Column(String)
    img = Column(String)
    url = Column(String)
    details = Column(String)
    description = Column(String)
    date = Column(String)
    location = Column(String)
    price = Column(String)
    date_scraped = Column(DateTime, FetchedValue())

class Subject(Base):
    __tablename__ = 'subjects'
    
    def __init__(self, name):
        self.name = name
    
    id = Column(BigInteger, primary_key=True)
    name = Column(String)
    date_scraped = Column(DateTime, FetchedValue())

class SubjectListing(Base):
    __tablename__ = 'subject_listings'
    
    def __init__(self, subject_id, listing_id):
        self.subject_id = subject_id
        self.listing_id = listing_id
    
    subject_id = Column(BigInteger, primary_key=True)
    listing_id = Column(BigInteger, primary_key=True)


# In[112]:


def select_subject(subject, sess):
    return sess.query(Subject).filter_by(name=subject).first()

def update_date_scraped(subject, sess):
    subject_entry = select_subject(subject, sess)
    now = datetime.datetime.utcnow()
    subject_entry.date_scraped = now
    sess.merge(subject_entry)
    sess.commit()
    
def probe_subject(subject, sess):
    '''
    Returns (id, ads_to_skip), where id is the primary key for use in the 
    subject_listings table, and ads_to_skip is the list of ads already in the DB,
    or None if the subject has been scraped recently (< 1 day)
    '''
    subject_entry = select_subject(subject, sess)
    if not subject_entry:
        new_subject = Subject(subject)
        sess.add(new_subject)
        sess.commit()
        sess.refresh(new_subject)
        return (new_subject.id, set())
    else:
        now = datetime.datetime.utcnow()
        time_since_last_scrape = now - subject_entry.date_scraped
        stale = time_since_last_scrape.days > 1
        if stale:
            listings = sess.query(SubjectListing.listing_id)                 .join(Subject, Subject.id == SubjectListing.subject_id)                 .filter(Subject.name == subject)
            return subject_entry.id, set([listing.listing_id for listing in listings])
        else:
            return subject_entry.id, None

def write_ads_to_db(subject_id, ads, sess):  # Writes ads from given dictionary to given file
    listings = [Listing(id, **ad) for id, ad in ads.items()]
    for listing in listings:
        sess.merge(listing)
        sess.merge(SubjectListing(subject_id, listing.id))
        sess.commit()


# In[ ]:


def connect_db():
    '''
    Returns a db session
    '''
    db_url = os.getenv('DB_URL')
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()


# In[ ]:


def get_listings(subject):
    '''
    Returns a dataframe with all listings for a subject
    '''
    sess = connect_db()
    subject_id, ads_to_skip = probe_subject(subject, sess)
    print(subject_id, ads_to_skip)
    
    # Scrape unless it was done recently
    if ads_to_skip is not None:
        url = None
        all_ads_scraped = False
        while not all_ads_scraped:
            ads, url = scrape(subject, ads_to_skip, limit=100, url=url)
            all_ads_scraped = True if url is None else False
            write_ads_to_db(subject_id, ads, sess)
            for ad_id in ads:
                ads_to_skip.add(ad_id)
                
        update_date_scraped(subject, sess)
    
    # This could be done with the ORM
    query = f'''
    select listing_id, title, description from (
        select * from subjects
            inner join subject_listings on subjects.id = subject_listings.subject_id where subjects.name = '{subject}'
    ) as this_subject_listings
        inner join listings on this_subject_listings.listing_id = listings.id;
    '''
    
    return pd.read_sql(query, sess.get_bind(), index_col='listing_id')       


# In[119]:


get_listings('hummels')


# In[117]:


scrape('panasonic tcp50x5 plasma tv')


# Reads in some listings

# In[11]:


def listings(subject, con):
    # Idk if this is injectable
    #sql_query = 'SELECT Description FROM {} WHERE delivery_method='Cesarean';', subject
    birth_data_from_sql = pd.read_sql_query(sql_query,con)
    listings_file_csv = '../Kijiji-Scraper/plasma_tvs.csv'
    #df = pd.read_csv(listings_file_csv)
    listings_file_json = '../Kijiji-Scraper/bikes.json'
    df = pd.read_json(listings_file_json, orient='index')
    # Kijiji uids look like unix timestamps, and afaict there's no way do stop
    # pandas interpreting them as such while using orient='index'
    df.index = df.index.astype(np.int64) // 10**9
    descs = [row['Description'] for _, row in df.iterrows()]
    print(descs)

listings('test')


# Initialize spacy with the largest English CNN
# 
# https://spacy.io/models/en#en_core_web_lg

# In[1]:


import spacy
nlp_full = spacy.load('en_core_web_lg')
#nlp_tokenizer = spacy.load('en_core_web_lg', disable=['tagger'])


# In[31]:


listings_file_json = '../Kijiji-Scraper/plasma_tvs.json'
my_subject = 'TV'
df = pd.read_json(listings_file_json, orient='index')
# Kijiji uids look like unix timestamps, and afaict there's no way do stop
# pandas interpreting them as such while using orient='index'
#df.index = df.index.astype(np.int64) // 10**9
descs = [row['Description'] for _, row in df.iterrows()]


# In[32]:


def fix_capitalization(text):
    '''This function lowercases sentences that are in all- or nearly-all-caps'''
    sents = nl.tokenize.sent_tokenize(text)
    # First, figure out if a sentence is mostly caps, vs lowers and digits
    # Lowercasing mostly-caps sentences improves parsing, and using digits
    #   helps avoid targeting model names with this heuristic
    for i, sent in enumerate(sents):
        words = nl.tokenize.word_tokenize(sent)
        uppers = 0
        lowers_digits = 0
        for word in words:
            for letter in word:
                if letter.isupper():
                    uppers += 1 
                elif letter.islower() or letter.isdigit():
                    lowers_digits += 1
        if uppers > lowers_digits * 3:
            #print('SHAME')
            fixed_sent = sent.lower()
            sents[i] = fixed_sent
    return ' '.join(sents)


# Lowercase all-caps sentences then run the NLP pipeline

# In[33]:


capcleaned_descs = [fix_capitalization(desc) for desc in descs]
docs = [nlp_full(desc) for desc in capcleaned_descs]
#capcleaned_descs


# In[34]:


tagged_words_spacy = []
for doc in docs:
    tagged_words_spacy.append([(token.text, token.tag_) for token in doc])
    
my_subject_lower = my_subject.lower()
def is_brand_model_candidate(word, tag, subject_lower):
    return tag in ['NNP'] and word.lower() != subject_lower
brand_model_cands = []
for sent in tagged_words_spacy:
    brand_model_cands.append([word for (word, tag) in sent if is_brand_model_candidate(word, tag, my_subject_lower)])
#brand_model_cands


# In[35]:


#print(brand_model_words)
def generate_preferred_spelling_dict(words):
    spellings = {}
    for word in words:
        word_lower = word.lower()
        if word_lower in spellings:
            spellings[word_lower].append(word)
        else:
            spellings[word_lower] = [word]
    preferred_spellings = {}
    for (word, spelling_cands) in spellings.items():
        n_occurrences = len(spelling_cands)
        preferred_spelling = max(set(spelling_cands), key=spelling_cands.count)
        preferred_spellings[word] = (preferred_spelling, n_occurrences)
    return preferred_spellings

def generate_multiplicity_dict(words):
    multiplicities = {}
    for word in words:
        if word not in multiplicities:
            multiplicities[word] = words.count(word)
    return multiplicities 


# In[36]:


listing_noun_phrases = []
stop_tags = ['PRP', 'DT'] #'IN'
for doc in docs:
    #spans = [span for span in list(doc.noun_chunks) ]
    #tokens = [(token, token.tag_) for span in spans for token in span]
    #print(tokens)
    noun_phrases = []
    for np in doc.noun_chunks:
        if np.root.tag_ not in stop_tags:
            important_descriptors = [word for word in np if not word.tag_ in stop_tags and not word.text == np.root.text]
            noun_phrases.append((important_descriptors, np.root.text))
    listing_noun_phrases.append(noun_phrases)
#listing_noun_phrases


# In[37]:


listing_noun_phrase_subjects = [np for listing in listing_noun_phrases for (descriptors, np) in listing]
subject_preferred_spellings = generate_preferred_spelling_dict(listing_noun_phrase_subjects)
popular_descriptors = list(subject_preferred_spellings.items())


# In[38]:


import hunspell
hobj = hunspell.HunSpell('../dict/en_CA.dic', '../dict/en_CA.aff')
brand_blacklist = []
brand_whitelist = ['Panasonic', 'Samsung', 'Sharp', 'LG', 'Fujitsu', 'Philips', 'Sony', 'CCM', 'Norco', 'Raleigh', 'Shimano', 'Supercycle', 'Schwinn']
def contains_number(string):
     return any(char.isdigit() for char in string)
# Most brand names don't contain numbers, but many model names do
# Most brand names are not english words
def find_likely_brand_names(brands):
    whitelisted_brands = [brand for brand in brands if brand in brand_whitelist]
    if whitelisted_brands: return whitelisted_brands
    return [brand for brand in brands if not hobj.spell(brand) and not contains_number(brand)]
    


# In[39]:


def cands_directly_describing_subject(cands, subj_descriptors):
    return [cand for cand in cands if cand.lower() in subj_descriptors]


# In[40]:


# A dictionary of preferred spellings also contains word occurrence multiplicities
def highest_multiplicity_cand(cands, preferred_spellings):
    return max(cands, key=lambda cand: preferred_spellings[cand.lower()][1])


# In[41]:


brand_names = []
flattened_cand_list = [cand for cands in brand_model_cands for cand in cands]
preferred_brand_spellings = generate_preferred_spelling_dict(flattened_cand_list)
for doc, brand_cands, nps in zip(
    docs, brand_model_cands, listing_noun_phrases
):
    if not brand_cands:
        brand_names.append('')
        continue
        
    # See if one of the candidates is being used to directly describe the subject of
    #   the listing, rather than some other noun in the listing.
    subj_descriptors = [descriptors for (descriptors, subj) in nps if subj.lower() == my_subject_lower]
    flattened_subj_descriptors = [x.text.lower() for y in subj_descriptors for x in y]
    
    top_cands = find_likely_brand_names(brand_cands)
    if top_cands:
        top_top_cands = cands_directly_describing_subject(top_cands, flattened_subj_descriptors)
        if top_top_cands:
            top_cand = highest_multiplicity_cand(top_top_cands, preferred_brand_spellings)
        else:
            top_cand = highest_multiplicity_cand(top_cands, preferred_brand_spellings)
    else:
        top_cands = [cand for cand in brand_cands if cand.lower() in flattened_subj_descriptors]
        if top_cands:
            top_cand = highest_multiplicity_cand(top_cands, preferred_brand_spellings)
        else:
            top_cand = highest_multiplicity_cand(brand_cands, preferred_brand_spellings)
    
    brand_names.append(preferred_brand_spellings[top_cand.lower()][0])
#brand_names


# In[42]:


popular_descriptors.sort(key=lambda desc: desc[1][1], reverse=True)
#popular_descriptors


# In[43]:


popular_brands = [preferred_spelling for (key, preferred_spelling) in preferred_brand_spellings.items()]
popular_brands.sort(key=lambda brand: brand[1], reverse=True)
#popular_brands


# In[58]:


most_popular_descriptors = [descriptor for (descriptor, _) in popular_descriptors[:10]]
aggregate_indirect_descriptors = []
indirect_descriptor_phrases = {descriptor:[] for descriptor in most_popular_descriptors}
for listing in listing_noun_phrases:
    for descriptors, subject in listing:
        subject_lower = subject.lower()
        if subject_lower in most_popular_descriptors:
            subject_descriptions = []
            description_buffer = []
            for descriptor in descriptors:
                #if len(descriptor.text) == 1 and re.findall('[^A-Za-z0-9]', descriptor.text): continue
                description_buffer.append(descriptor.text)
                aggregate_indirect_descriptors.append(descriptor.text)
                #print(descriptor.text)
                # If the descriptor directly modifies the subject of the NP, take it
                # and all descriptors in the buffer (that presumably modify this new descriptor)
                if descriptor.head.text == subject:
                    subject_descriptions.append(description_buffer)
                    description_buffer = []
            indirect_descriptor_phrases[subject_lower].append(subject_descriptions)
            
            #print(subject)        
            #print(subject_descriptions)
            
preferred_descriptor_spellings = generate_preferred_spelling_dict(aggregate_indirect_descriptors)
#print(indirect_descriptor_phrases)
# for subject, listings in indirect_descriptor_phrases.items():
#     for i, listing in enumerate(listings):
#         for j, description in enumerate(listing):
#             print(description)
#             for k, descriptor in enumerate(description):
#                 #print(descriptor)
#                 preferred_spelling = preferred_descriptor_spellings[descriptor.lower()][0]
#                 if descriptor != preferred_spelling:
#                     indirect_descriptor_phrases[subject][i][j][k] = preferred_spelling
# print(indirect_descriptor_phrases)
top_indirect_descriptors = {descriptor:[] for descriptor in most_popular_descriptors}
for subject, listings in indirect_descriptor_phrases.items():
    flattened_indirect_descriptor_phrase_list = []
    for listing in listings:
        for description in listing:
            print(description)
            # This will unfortunately put spaces around hyphens, and that sort of thing
            text_description = ' '.join([preferred_descriptor_spellings[descriptor.lower()][0] for descriptor in description])
            flattened_indirect_descriptor_phrase_list.append(text_description)
    preferred_descriptions = list(generate_multiplicity_dict(flattened_indirect_descriptor_phrase_list).items())
    preferred_descriptions.sort(key=lambda desc: desc[1], reverse=True)
    top_indirect_descriptors[subject] = preferred_descriptions

for feature, descriptors in top_indirect_descriptors.items():
    print(f'{feature}:')
    for descriptor, mult in descriptors[:8]:
        print(f'\t{descriptor} ({mult})')


# In[45]:


for brand, mult in popular_brands[:15]:
    print(f'{brand} ({mult})')


# In[46]:


# for doc in docs[:3]:
#     spacy.displacy.render(doc, style='dep', jupyter=True)
#     print(doc)
#     print(doc.ents)


# In[47]:


#descs


# In[48]:


# import hunspell
# hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
# brand_model_cands


# In[49]:


#testdoc = nlp_full('42x42 cm black cat')
#spacy.displacy.render(testdoc, style='dep', jupyter = True)


# In[50]:


# vectorizer = sk.feature_extraction.text.CountVectorizer()


# In[51]:


# word_bag = vectorizer.fit_transform(important_words).toarray()
# bag_words = vectorizer.get_feature_names()
# # We don't care if a word appears multiple times in the same listing
# for (index, single_listing_word_multiplicity) in np.ndenumerate(word_bag):
#     if single_listing_word_multiplicity > 1: word_bag[index] = 1
# aggregate_multiplicities = word_bag.sum(axis=0)
# word_multiplicity = list(zip(bag_words, aggregate_multiplicities.tolist()))

# #word_multiplicity = np.column_stack((bag_words, aggregate_multiplicities))


# In[52]:


# # https://stackoverflow.com/a/2828121
# #def sort_words_by_multiplicity(words):
# #    return words[words[:,1].argsort()]

# #sorted_words = sort_words_by_multiplicity(word_multiplicity)
# #sorted_words = sorted_words[::-1]
# #sorted_words[:30]

# word_multiplicity.sort(key=lambda word_multiplicity: word_multiplicity[1])
# word_multiplicity[::-1]


# In[53]:


# def tag_words(text):
#       tokens = nl.word_tokenize(text)
#       return nl.pos_tag(tokens)


# In[54]:


# def ner_words(text):
#       tokens = nl.word_tokenize(text)
#       return nl.chunk.ne_chunk(tokens)


# In[55]:


# chunked_words = [nl.chunk.ne_chunk(desc) for desc in tagged_words]
# chunked_words


# In[56]:


# #print(brand_model_cands)
# treated_words = set()
# spellings = {}
# for listing_cands in brand_model_cands:
#     for cand in listing_cands:
#         cand_lower = cand.lower()
#         if cand.lower() in spellings:
#             spellings[cand_lower].append(cand)
#         else:
#             spellings[cand_lower] = [cand]
# spellings


# In[57]:


#preferred_spellings = {}
#single_spellings = set()
#for (word, spelling_cands) in spellings.items():
#    n_occurrences = len(spelling_cands)
#    preferred_spelling = max(set(spelling_cands), key=spelling_cands.count)
#    preferred_spellings[word] = (preferred_spelling, n_occurrences)
# preferred_spellings


# In[32]:


#hummel_ads = {'1202550115': {'title': 'Hummel Style Japanese Girl Chicken', 'img': '<img alt="Hummel Style Japanese Girl Chicken" src="https://i.ebayimg.com/00/s/ODAwWDQ1MA==/z/ySEAAOSwTA9X5tuP/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-style-japanese-girl-chicken/1202550115', 'details': '', 'date': '< 23 hours ago', 'location': 'City of Toronto', 'price': '-1.00', 'description': 'Hummel Style Japanese 5 x 4 inches no chips cracks or fleas'}, '1358609245': {'title': 'Hummel "Trumpet Boy"', 'img': '<img alt=\'Hummel "Trumpet Boy"\' src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/kJIAAOSw1xVbDIcB/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-trumpet-boy/1358609245', 'details': '', 'date': '20/09/2018', 'location': 'City of Toronto', 'price': '25.00', 'description': 'Early Goebel Hummel - Germany\n"Trumpet Boy" - #97\nStands 4 1/2" high'}, '1358607961': {'title': 'Hummel "Boy with Basket"', 'img': '<img alt=\'Hummel "Boy with Basket"\' src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/yloAAOSw8SpbDIXj/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-boy-with-basket/1358607961', 'details': '', 'date': '20/09/2018', 'location': 'City of Toronto', 'price': '25.00', 'description': 'Early Goebel Hummel - Germany\n"Village Boy" #51 3/0\nStands 4" high'}, '1260421734': {'title': 'hummel like figurines', 'img': '<img alt="hummel like figurines" src="https://i.ebayimg.com/00/s/NjQwWDQ4MA==/z/PNgAAOSw7GRZB~Te/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-like-figurines/1260421734', 'details': '', 'date': '18/09/2018', 'location': 'City of Toronto', 'price': '35.00', 'description': 'Hummel-Like Porcelain figurines , set of 6, mint condition'}, '1259656455': {'title': 'vintage Hummel like porcelain', 'img': '<img alt="vintage Hummel like porcelain" src="https://i.ebayimg.com/00/s/NjQwWDQ4MA==/z/8HcAAOSwlndZBNhB/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/vintage-hummel-like-porcelain/1259656455', 'details': '', 'date': '14/09/2018', 'location': 'City of Toronto', 'price': '25.00', 'description': 'vintage Hummel like porcelain figurine, 5" high, mint condition'}, '1340613666': {'title': 'Hummels - "Little Gardner"; "School Girl"; or "For Mother"', 'img': '<img alt=\'Hummels - "Little Gardner"; "School Girl"; or "For Mother"\' src="https://i.ebayimg.com/00/s/NTMzWDgwMA==/z/D~IAAOSwwz5arsuK/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummels-little-gardner-school-girl-or-for-mother/1340613666', 'details': '', 'date': '09/09/2018', 'location': 'City of Toronto', 'price': '50.00', 'description': 'Beautiful figurines. No chips or scratches. Complete with name tags attached. Excellent condition. Looking for $50.00 each, will accept reasonable offer'}, '1355433085': {'title': 'Hummel " Apple Tree Boy " Figurine', 'img': '<img alt=\'Hummel " Apple Tree Boy " Figurine\' src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/FFMAAOSwHf5a~GWV/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-apple-tree-boy-figurine/1355433085', 'details': '', 'date': '04/09/2018', 'location': 'City of Toronto', 'price': '50.00', 'description': 'For sale is a Hummel Figurine titled " Apple Tree Boy ". The base has a small chip that is barely noticeable,..the rest of this figurine is in perfect condition. It measures 6" tall. Older trade mark on bottom. Will ship if needed. Please visit my other items I have up for sale.'}, '419236774': {'title': 'Vintage Hummel "Happy Traveler" 109/0', 'img': '<img alt=\'Vintage Hummel "Happy Traveler" 109/0\' src="https://i.ebayimg.com/00/s/MTAwMFg3NTA=/$(KGrHqJ,!kwFBVKMfvwGBQbHRQrfH!~~48_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/vintage-hummel-happy-traveler-109-0/419236774', 'details': '', 'date': '01/09/2018', 'location': 'City of Toronto', 'price': '150.00', 'description': 'Pristine condition. TMK 3 (1960-1972), Sty-Bee. Retired in 1982. Price negotiable. After 5:00 p.m. and on weekends, call 416-825-3561.'}, '1380413234': {'title': 'vintage Hummel-like figurines', 'img': '<img alt="vintage Hummel-like figurines" src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/xt4AAOSwfbpbiFAi/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/vintage-hummel-like-figurines/1380413234', 'details': '', 'date': '30/08/2018', 'location': 'City of Toronto', 'price': '30.00', 'description': 'Vintage Hummel-like figurines\nMade in Japan - C7654\nBoy playing violin\nGirl playing drum\nexcellent condition'}, '1224890286': {'title': 'Hummel Plates, 3', 'img': '<img alt="Hummel Plates, 3" src="https://i.ebayimg.com/00/s/ODAwWDQ1MA==/z/PdQAAOSwa~BYVHVV/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-plates-3/1224890286', 'details': '', 'date': '26/08/2018', 'location': 'City of Toronto', 'price': '60.00', 'description': 'Price is for all 3. Can be purchased individually as well'}, '1224722568': {'title': 'Berta Hummel 1977 Limited Edition Christmas Plate', 'img': '<img alt="Berta Hummel 1977 Limited Edition Christmas Plate" src="https://i.ebayimg.com/00/s/ODAwWDQ1MA==/z/n9AAAOSwa~BYU4bb/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/berta-hummel-1977-limited-edition-christmas-plate/1224722568', 'details': '', 'date': '26/08/2018', 'location': 'City of Toronto', 'price': '140.00', 'description': 'Excellent condition, with original box and packaging, limited edition.'}, '1224660959': {'title': 'Hummel 1977 Plate - Apple Tree Boy', 'img': '<img alt="Hummel 1977 Plate - Apple Tree Boy" src="https://i.ebayimg.com/00/s/ODAwWDQ1MA==/z/T68AAOSwcUBYUyQz/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-1977-plate-apple-tree-boy/1224660959', 'details': '', 'date': '26/08/2018', 'location': 'City of Toronto', 'price': '150.00', 'description': 'No longer being made, vintage. In excellent condition. With original box and packaging.'}, '1379273218': {'title': 'Hummel Umbrella Boy and Girl porcelain dolls with soft body', 'img': '<img alt="Hummel Umbrella Boy and Girl porcelain dolls with soft body " src="https://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/4hQAAOSwK6NbgfVr/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-umbrella-boy-and-girl-porcelain-dolls-with-soft-body/1379273218', 'details': '', 'date': '25/08/2018', 'location': 'City of Toronto', 'price': '100.00', 'description': 'Hummel Umbrella Boy and Girl porcelain dolls with soft body. Great condition dolls have been kept wrapped in storage. Have documents for boy doll but not girl.\nPrice is negotiable'}, '1314353570': {'title': 'Vintage Berta Hummel Christmas Collector Plates - Mint!!', 'img': '<img alt="Vintage Berta Hummel Christmas Collector Plates - Mint!!" src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/W60AAOSwFyhaEOEc/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/vintage-berta-hummel-christmas-collector-plates-mint/1314353570', 'details': '', 'date': '23/08/2018', 'location': 'City of Toronto', 'price': '15.00', 'description': '$15 each. These are limited edition Berta Hummel Collector Christmas plates made in Germany by Schmid Brothers. The set has annual Christmas plates from 1971 through 1989. Each plate has a unique year and a scene portraying the authentic works of Sister Berta Hummel. (see all photos). The scenes are repnoruced on the finest Bavarian porcelain. These plates are in mint condition and most are in their orinal box. The plates are 8" in diameter and are great for serving special dishes and desserts during the festive season. They can be given as a gift or collected for their beauty.'}, '1314346772': {'title': '1982 Hummel Christmas Ornament - New - Excellent Condition!', 'img': '<img alt="1982 Hummel Christmas Ornament - New - Excellent Condition!" src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/SWgAAOSwPAxaENid/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/1982-hummel-christmas-ornament-new-excellent-condition/1314346772', 'details': '', 'date': '23/08/2018', 'location': 'City of Toronto', 'price': '15.00', 'description': 'This is a 1982 Hummel first annual collectors\' Christmas ornament. It is new in its original box. There are 3 separate scenes on the ornament: Gift Bearers, Angel\'s Music and A Gift for Jesus. This ornament is from the authentic ARS edition and is like new condition. The ornament is 4" across and a circumference of 10".'}, '1378715810': {'title': 'Mats Hummels game used 3 colour path autograph card', 'img': '<img alt="Mats Hummels game used 3 colour path autograph card" src="https://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/cOUAAOSwoYxbfvez/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/mats-hummels-game-used-3-colour-path-autograph-card/1378715810', 'details': '', 'date': '23/08/2018', 'location': 'City of Toronto', 'price': '65.00', 'description': 'This is a signed 3 colour patch autograph of Mats Hummels from the 2017 Immaculate collection. Please feel free to look at my other ads. Thanks for looking!'}, '1365444327': {'title': 'Collectibles Plate Collection', 'img': '<img alt="Collectibles Plate Collection" src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/FdQAAOSwh8NbMWWX/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/collectibles-plate-collection/1365444327', 'details': '', 'date': '22/08/2018', 'location': 'City of Toronto', 'price': '7.00', 'description': 'M.I. Hummel Plate Collection\n"Little Companions"\nAn edition limited to 14 full firing days.\nPlate no. YB 4571'}, '1378040238': {'title': 'Hummel Angel Figurine', 'img': '<img alt="Hummel Angel Figurine" src="https://i.ebayimg.com/00/s/MjQwWDMyMA==/z/WfoAAOSwO2lbey5i/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-angel-figurine/1378040238', 'details': '', 'date': '20/08/2018', 'location': 'City of Toronto', 'price': '15.00', 'description': 'Like new beautiful Hummel Angel Figurine.\nPraying before bedtime. Might be a nice shower gift.\nCollectible. Worth $100+'}, '1335229712': {'title': 'Hummel Apple Tree Girl', 'img': '<img alt="Hummel Apple Tree Girl" src="https://i.ebayimg.com/00/s/ODAwWDQ1MA==/z/zhMAAOSwIzFaj8J2/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-apple-tree-girl/1335229712', 'details': '', 'date': '13/08/2018', 'location': 'City of Toronto', 'price': '90.00', 'description': '4 inch figure. Full bee'}, '1374824000': {'title': 'Hummel special edition silver spoons', 'img': '<img alt="Hummel special edition silver spoons" src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/JQQAAOSws4lbaMC7/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-special-edition-silver-spoons/1374824000', 'details': '', 'date': '06/08/2018', 'location': 'City of Toronto', 'price': '25.00', 'description': 'Three spoons still in original boxes.\nBoy on phone.\nBoy reading book\nChild in hammock.\n1982 ARS special editions.\n$25.00 for all three.\nScarborough. location'}, '1361843722': {'title': 'Ceramic Figurine Erich Stauffer Hummel Style Girl Sewing', 'img': '<img alt="Ceramic Figurine Erich Stauffer Hummel Style Girl Sewing" src="https://i.ebayimg.com/00/s/ODAwWDYwMA==/z/Ns4AAOSwdm1bHWS6/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/ceramic-figurine-erich-stauffer-hummel-style-girl-sewing/1361843722', 'details': '', 'date': '05/08/2018', 'location': 'City of Toronto', 'price': '5.00', 'description': 'Vintage Ceramic Figurine Erich Stauffer Hummel Style Girl Sewing "Playing House"'}, '1361844209': {'title': '2PC Goebel Hummel Figurine "Little Music Makers"', 'img': '<img alt=\'2PC Goebel Hummel Figurine "Little Music Makers"\' src="https://i.ebayimg.com/00/s/NjAwWDgwMA==/z/AEcAAOSwuHJbHWUi/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/2pc-goebel-hummel-figurine-little-music-makers/1361844209', 'details': '', 'date': '05/08/2018', 'location': 'City of Toronto', 'price': '20.00', 'description': '2 piece Goebel Hummel Figurine "Little Music Makers"'}, '1361848787': {'title': 'HUMMEL 1980 - SPRING DANCE - ANNIVERSARY PLATE - SECOND EDITION', 'img': '<img alt="HUMMEL 1980 - SPRING DANCE - ANNIVERSARY PLATE - SECOND EDITION" src="https://i.ebayimg.com/00/s/ODAwWDYwMA==/z/M9UAAOSwzzVbHWkE/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-1980-spring-dance-anniversary-plate-second-edition/1361848787', 'details': '', 'date': '05/08/2018', 'location': 'City of Toronto', 'price': '30.00', 'description': 'HUMMEL-1980-SPRING-DANCE-ANNIVERSARY-PLATE-SECOND-EDITION'}, '1156862038': {'title': 'Hummel Playmates, Discontinued', 'img': '<img alt="Hummel Playmates, Discontinued" src="https://i.ebayimg.com/00/s/ODAwWDQ1MA==/z/xXgAAOSw7n9XEnzf/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/hummel-playmates-discontinued/1156862038', 'details': '', 'date': '03/08/2018', 'location': 'City of Toronto', 'price': '150.00', 'description': '4 inches high by 3.5 inches wide. Discontinued. With the "V" mark on the bottom.....very old.'}, '1322284610': {'title': 'Vintage  Porcelain  Doll', 'img': '<img alt="Vintage  Porcelain  Doll" src="https://i.ebayimg.com/00/s/ODAwWDM4OA==/z/X5wAAOSw1JVaPBw3/$_35.JPG"/>', 'url': 'http://www.kijiji.ca/v-art-collectibles/city-of-toronto/vintage-porcelain-doll/1322284610', 'details': '', 'date': '30/07/2018', 'location': 'City of Toronto', 'price': '20.00', 'description': 'Rare Vintage 10" Tall Hand Painted Porcelain Bisque Gretel Doll\nThis doll is from the Hummel series of Hansel & Gretel Figurines\nLovely detailed hand painted face with minor paint loss on the lips. In original clothes which is in very good condition ,,,the waist apron has lost a bit of its elasticity but otherwise clothes is all original and in excellent condition.\nShe stands approx. 10" tall when standing up.\nShe can stand or sit and all her joints arms and legs are moveable.\nNo chips or breaks or imperfections overall in excellent condition.\nI believe she is Hummel inspired\nIf you can see the ad the item is available for pick up downtown Toronto. Please call or txt 416-816-4020 if interested'}}

