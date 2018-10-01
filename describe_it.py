
# coding: utf-8

# In[1]:

import os
import sys
import re
import logging
from datetime import datetime, date, timedelta

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


import spacy
import hunspell

nlp = spacy.load('en_core_web_lg')

# Add personal possessive pronouns to stop-words collection
for ppp in ['my', 'your', 'their', 'his', 'her', 'our']:
    nlp.vocab[ppp].is_stop = True

spellcheck = hunspell.HunSpell('../dict/en_CA.dic', '../dict/en_CA.aff')


# In[2]:


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
    date_posted = Column(DateTime)
    location = Column(String)
    price = Column(String)
    date_scraped = Column(DateTime, FetchedValue())

class Subject(Base):
    __tablename__ = 'subjects'
    
    def __init__(self, name):
        self.name = name
    
    id = Column(BigInteger, primary_key=True)
    name = Column(String)
    date_last_scraped = Column(DateTime, FetchedValue())

class SubjectListing(Base):
    __tablename__ = 'subject_listings'
    
    def __init__(self, subject_id, listing_id):
        self.subject_id = subject_id
        self.listing_id = listing_id
    
    subject_id = Column(BigInteger, primary_key=True)
    listing_id = Column(BigInteger, primary_key=True)


# In[37]:


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

    raw_date = ''
    try:
        raw_date = html.find('span', {'class': 'date-posted'}).text.strip()
        # If the ad is less than 24h old, it's displayed as '< some amount of time'
        if raw_date[0] == '<':
            date = datetime.today().date()
        # Or "Yesterday"
        elif raw_date[0] == 'Y':
            date = datetime.today().date() - timedelta(days=1)
        else:
            date = datetime.strptime(raw_date, '%d/%m/%Y')
        ad_info['date_posted'] = date
    except:
        logging.error(f'Unable to parse Date data: \'{raw_date}\'')

    # The location field is affixed with the date for some reason
    try:
        location = html.find('div', {'class': 'location'}).text.strip()
        location = location.replace(raw_date, '')
        ad_info['location'] = location
    except:
        logging.error('Unable to parse Location data.')

    # In addition to normal prices, there can be 'Free' ($0), 'Please Contact' ($-1),
    # some other value ($-2), or empty ($-3)
    try:
        price = html.find('div', {'class': 'price'}).text.strip().split('\n')[0]
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
        logging.error('[Error] Unable to load ' + url)
        return None
    
    soup = BeautifulSoup(page.content, 'html.parser')
    
    try:
        return soup.find('div', {'itemprop': 'description'}).get_text()
    except:
        logging.error('Unable to parse ad description.')
        return None

def scrape(subject, existing_ad_ids = None, limit = None, url = None):
    '''
    Args are a search string for some subject, a list of existing ad IDs to skip,
    a limit to the number of results to return, and a url from which to start
    (in case, for example, a previous scrape was halted and is now being resumed).
    Returns (ads, already_scraped_ad_ids, all_ads_scraped); all_ads_scraped is whether
    or not an imposed limit was reached.
    '''
    # Initialize variables for loop
    ad_dict = {}
    # Shallow copy for now
    ad_ids_to_skip = existing_ad_ids if existing_ad_ids is not None else set()

    if url is None: url =         f'https://www.kijiji.ca/b-city-of-toronto/{subject}/k0l1700273?ll=43.650843,-79.377573&dc=true'

    ads_parsed = 0
    
    already_scraped_ad_ids = set()
    
    while url:

        try:
            page = requests.get(url)
        except:
            logging.error(f'[Error] Unable to load {url}')
            return

        soup = BeautifulSoup(page.content, 'html.parser')

        ads = soup.find_all('div', {'class': 'regular-ad'})
        
        # Skip third-party ads; these third parties are typically small retailers
        third_party_ads = soup.find_all('div', {'class': 'third-party'})
        third_party_ad_ids = set([int(ad['data-ad-id']) for ad in third_party_ads])
        
        # Parse ads until the limit is reached
        for ad in ads:
            title = ad.find('a', {'class': 'title'}).text.strip()
            ad_id = int(ad['data-ad-id'])
            
            if ad_id not in third_party_ad_ids:
                if ad_id not in ad_ids_to_skip:
                    logging.info(f'New ad found! Ad id: {ad_id}')
                    ad_info = parse_ad_summary(ad)
                    ad_url = ad_info['url']
                    description = parse_ad_page(ad_url)
                    if description is None: continue
                    ad_info['description'] = description 
                    ad_dict[ad_id] = ad_info
                    if limit is not None:
                        ads_parsed += 1
                        if ads_parsed >= limit: return ad_dict, already_scraped_ad_ids, url
                else:
                    logging.debug('Skip already-scraped ad')
                    already_scraped_ad_ids.add(ad_id)
            else:
                logging.debug('Skip third-party ad')
                
        url = soup.find('a', {'title' : 'Next'})
        if url:
            url_path = url['href']
            url = f'https://www.kijiji.ca{url_path}'

    return (ad_dict, already_scraped_ad_ids) if limit is None else (ad_dict, already_scraped_ad_ids, None)


# In[4]:


def select_subject(subject, sess):
    return sess.query(Subject).filter_by(name=subject).first()

def update_date_scraped(subject, sess):
    subject_entry = select_subject(subject, sess)
    now = datetime.utcnow()
    subject_entry.date_last_scraped = now
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
        subject_id = new_subject.id
    else:
        subject_id = subject_entry.id
        now = datetime.utcnow()
        time_since_last_scrape = now - subject_entry.date_last_scraped
        freshly_scraped = time_since_last_scrape.days < 365
        if freshly_scraped:
            return subject_id, None
    
    listings = sess.query(Listing.id)
    # each returned element from a single-column select is a single-element tuple
    return subject_id, set([listing[0] for listing in listings])

def write_ads_to_db(subject_id, ads, sess):
    logging.info('Writing new ads to DB')
    listings = [Listing(id, **ad) for id, ad in ads.items()]
    for listing in listings:
        sess.merge(listing)
        sess.merge(SubjectListing(subject_id, listing.id))
        sess.commit()

def write_subject_listing_relations_for_already_scraped_ads_to_db(subject_id, already_scraped_ad_ids, sess):
    logging.info('Updating subject with ads already scraped')
    for already_scraped_ad_id in already_scraped_ad_ids:
        sess.merge(SubjectListing(subject_id, already_scraped_ad_id))
        sess.commit()
        


# In[5]:


def connect_db():
    '''
    Returns a db session
    '''
    db_url = os.getenv('DB_URL')
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()


# In[6]:


def get_listings(subject):
    '''
    Returns a dataframe with all listings for a subject
    '''
    subject = subject.lower()
    sess = connect_db()
    subject_id, ads_to_skip = probe_subject(subject, sess)
    
    # Scrape unless it was done recently
    if ads_to_skip is not None:
        url = None
        all_ads_scraped = False
        # Every 50 ads, write to DB
        limit = 50
        while not all_ads_scraped:
            ads, already_scraped_ad_ids, url = scrape(subject, ads_to_skip, limit=limit, url=url)
            all_ads_scraped = True if url is None else False
            write_ads_to_db(subject_id, ads, sess)
            write_subject_listing_relations_for_already_scraped_ads_to_db(
                subject_id, already_scraped_ad_ids, sess)
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

def fix_capitalization(text):
    '''
    Lowercases sentences in a body of text that are either:
    * all-(or nearly-all-)caps
    Too many capitalized english words (very common in classifieds)
    '''
    sents = nl.tokenize.sent_tokenize(text)
    # First, figure out if a sentence is mostly caps, vs lowers and digits
    # Lowercasing mostly-caps sentences improves parsing, and using digits
    #   helps avoid targeting model names with this heuristic
    for i, sent in enumerate(sents):
        words = nl.tokenize.word_tokenize(sent)
        uppers = 0
        lowers_digits = 0
        capitalized_words = 0
        for word in words:
            for letter in word:
                if letter.isupper():
                    uppers += 1 
                elif letter.islower() or letter.isdigit():
                    lowers_digits += 1
            if word[0].isupper() and spellcheck.spell(word):
                capitalized_words += 1
                
        if uppers > lowers_digits * 3 or capitalized_words > 5:
            #print('SHAME')
            fixed_sent = sent.lower()
            sents[i] = fixed_sent
            
    return ' '.join(sents)

def replace_newlines_with_periods(descs):
    newline_with_optional_periods = re.compile('\.?\n')
    return [newline_with_optional_periods.sub('. ', desc) for desc in descs]

# I'm sure there's a way to generalize this regex,
# but I'm also sure nobody will be describing a four-dimensional feature
def normalize_measurements(descs):
    # start-of-line or whitespace
    SoL_or_WS = r'(^|\s)'
    # measurement
    m = r'(\d{1,9}|\d*\.\d+|\d+/\d+|\d+ \d+/\d+)'
    # dimensional separator
    DS = r'\s*[*xX×]\s*'
    unit = r'[-\s]*(\'|"|\w{1,2}\d?\.?\s+|in\.|inc?h?e?s?)\s*'
    unitless = r'(?:\s+|,)'
    # Unit and unitless regexes overlap, so they must be applied in that order
    dimension_regexes = []
    dimension_regexes.append(('32', re.compile(f'{SoL_or_WS}{m}{DS}{m}{DS}{m}{unit}'))) 
    dimension_regexes.append(('31', re.compile(f'{SoL_or_WS}{m}{DS}{m}{DS}{m}{unitless}')))
    dimension_regexes.append(('22', re.compile(f'{SoL_or_WS}{m}{DS}{m}{unit}')))
    dimension_regexes.append(('21', re.compile(f'{SoL_or_WS}{m}{DS}{m}{unitless}')))
    dimension_regexes.append(('12', re.compile(f'{SoL_or_WS}{m}{unit}')))
    dimension_regexes.append(('11', re.compile(f'{SoL_or_WS}{m}{unitless}')))

    #two_d_search = re.compile(r'(^|\s)(\d+)\s*[*x×]\s*(\d+)\s*(\w{1,2}\d?)')
    #two_d_replace = re.compile(r'^\1×\2\3')
    #three_d_search = re.compile(r'(^|\s)(\d+)\s*[*x×]\s*(\d+)\s*[*x×]\s*(\d+)\s*(\w{1,2}\d?)?')
    #two_d_replace = re.compile(r'\1×\2×\3\4')
    
    #measurements = {}
    #unnormalized_3d_measurements_in_descs = [three_d_search.findall(desc) for desc in descs]
    #unnormalized_2d_measurements_in_descs = [two_d_search.findall(desc) for desc in descs]
    #unnormalized_1d_measurements_in_descs = [one_d_search.findall(desc) for desc in descs]
   
    # Keyphrase with which to replace measurements
    # Will be affixed with dimensional info
    MK = '1029384756'
    
    original_descs = descs.copy()

    for dimension, regex in dimension_regexes:
        repl_template = f'{dimension}{MK}'
        for desc_i, desc in enumerate(descs):
            i = 0
            while True:
                subbed_desc = regex.sub(f' {repl_template}{i} ', desc, count = 1)
                if i > 20:
                    logging.error('too many measurements; probably a parsing error')
                    return
                if desc == subbed_desc: break
                desc = subbed_desc
            descs[desc_i] = desc

def generate_preferred_spelling_dict(words):
    '''
    For some set of words, returns a dict mapping each unique lowercased word to
    its most popular spelling and total occurrences of all spellings.
    '''
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
    '''
    Counts the number of occurrences of each word, case-sensitive.
    '''
    multiplicities = {}
    for word in words:
        if word not in multiplicities:
            multiplicities[word] = words.count(word)
    return multiplicities 


# In[48]:


def cands_directly_describing_subject(cands, subj_descriptors):
    return [cand for cand in cands if cand.lower() in subj_descriptors]


# In[49]:


# A dictionary of preferred spellings also contains word occurrence multiplicities
def highest_multiplicity_cand(cands, preferred_spellings):
    return max(cands, key=lambda cand: preferred_spellings[cand.lower()][1])


# ### Identify brand candidates

# In[50]:


def is_brand_model_candidate(word, tag, subject_lower):
    return tag in ['NNP'] and word.lower() != subject_lower

brand_blacklist = []
brand_whitelist = ['Panasonic', 'Samsung', 'Sharp', 'LG', 'Fujitsu', 'Philips', 'Sony', 'CCM', 'Norco', 'Raleigh', 'Shimano', 'Supercycle', 'Schwinn']
def contains_number(string):
     return any(char.isdigit() for char in string)
# Most brand names don't contain numbers, but many model names do
# Most brand names are not english words
def find_likely_brand_names(brands):
    whitelisted_brands = [brand for brand in brands if brand in brand_whitelist]
    if whitelisted_brands: return whitelisted_brands
    return [brand for brand in brands if not spellcheck.spell(brand) and not contains_number(brand)]

def reassociate_orphaned_descriptor(orphaned_descriptor, features_descriptors):
    most_occurrences = 0
    for _, feature_descriptors in features_descriptors.items():
        #print(feature_descriptors)
        for i, (feature_descriptor, mult) in enumerate(feature_descriptors):
            #print(orphaned_descriptor, feature_descriptor)
            if orphaned_descriptor == feature_descriptor:
                if mult > most_occurrences: most_occurrences = mult
    for _, feature_descriptors in features_descriptors.items():
        #print(feature_descriptors)
        for i, (feature_descriptor, mult) in enumerate(feature_descriptors):
            if mult == most_occurrences and orphaned_descriptor == feature_descriptor:
                feature_descriptors[i] = (feature_descriptor, mult + 1)
                return True
    return False

def top_features_and_descriptors(subject):
    my_subject = subject
    df = get_listings(my_subject)
    # Kijiji uids look like unix timestamps, and afaict there's no way do stop
    # pandas interpreting them as such while using orient='index'
    #df.index = df.index.astype(np.int64) // 10**9
    #return df
    descs = [row['description'] for _, row in df.iterrows()]

    # ## Pre-processing
    # * lowercasing all-caps and over-capped sentences
    # * replacing measurements with tokens identifying their dimensionality and whether or not they carry a unit

    descs = replace_newlines_with_periods(descs)

    normalize_measurements(descs)

    descs = [fix_capitalization(desc) for desc in descs]


    # lol, hardcode tv as the lemma for tvs
    my_subject_lemmas = [word.lemma_ if word.text != 'tvs' else 'tv' for word in nlp(my_subject)]
    docs = [nlp(desc) for desc in descs]


## Post-processing

    tagged_words_spacy = []
    for doc in docs:
        tagged_words_spacy.append([(token.text, token.tag_) for token in doc])
        
    my_subject_lower = my_subject.lower()
    brand_model_cands = []
    for sent in tagged_words_spacy:
        brand_model_cands.append([word for (word, tag) in sent if is_brand_model_candidate(word, tag, my_subject_lower)])

    listings_described_features = []
    listings_orphaned_descriptors = []
    stop_tags = ['PRP', 'DT'] #'IN'
    for doc in docs:
        described_features = []
        orphaned_descriptors = []
        for np in doc.noun_chunks:
            if np.root.tag_ not in stop_tags:
                interesting_descriptors = [
                    word for word in np 
                        if not word.tag_ in stop_tags 
                        and not word.is_stop
                        and not word.text == np.root.text
                ]
                if np.root.lemma_ in my_subject_lemmas:
                    orphaned_descriptors.append(interesting_descriptors)
                else:
                    described_features.append((interesting_descriptors, np.root.text))
        listings_described_features.append(described_features)
        listings_orphaned_descriptors.append(orphaned_descriptors)

    brand_names = []
    flattened_cand_list = [cand for cands in brand_model_cands for cand in cands]
    preferred_brand_spellings = generate_preferred_spelling_dict(flattened_cand_list)
    for doc, brand_cands, listing_described_features in zip(
        docs, brand_model_cands, listings_described_features
    ):
        if not brand_cands:
            brand_names.append('')
            continue
            
        # See if one of the candidates is being used to directly describe the subject of
        #   the listing, rather than some other noun in the listing.
        feature_descriptors = [descriptors for (descriptors, feature) in listing_described_features if feature.lower() == my_subject_lower]
        flattened_feature_descriptors = [x.text.lower() for y in feature_descriptors for x in y]
        
        top_cands = find_likely_brand_names(brand_cands)
        if top_cands:
            top_top_cands = cands_directly_describing_subject(top_cands, flattened_feature_descriptors)
            if top_top_cands:
                top_cand = highest_multiplicity_cand(top_top_cands, preferred_brand_spellings)
            else:
                top_cand = highest_multiplicity_cand(top_cands, preferred_brand_spellings)
        else:
            top_cands = [cand for cand in brand_cands if cand.lower() in flattened_feature_descriptors]
            if top_cands:
                top_cand = highest_multiplicity_cand(top_cands, preferred_brand_spellings)
            else:
                top_cand = highest_multiplicity_cand(brand_cands, preferred_brand_spellings)
        
        brand_names.append(preferred_brand_spellings[top_cand.lower()][0])
    #brand_names


    # In[82]:


    popular_brands = [preferred_spelling for (key, preferred_spelling) in preferred_brand_spellings.items()]
    popular_brands.sort(key=lambda brand: brand[1], reverse=True)
    #popular_brands


    # In[83]:


    features = [
        feature 
        for listing_described_features in listings_described_features
        for (descriptors, feature) in listing_described_features
    ]
    feature_preferred_spellings = generate_preferred_spelling_dict(features)
    popular_features = list(feature_preferred_spellings.items())
    popular_features.sort(key=lambda desc: desc[1][1], reverse=True)


    # In[84]:


    my_subject_lemmas


    # In[85]:


    most_popular_features = [feature for (feature, _) in popular_features[:10]]
    all_descriptors = set()
    feature_descriptors = {feature:[] for feature in most_popular_features}
    for listing_described_features in listings_described_features:
        for descriptors, feature in listing_described_features:
            feature = feature.lower()
            if feature in most_popular_features:
                feature_descriptions = []
                #already_handled = []
                for descriptor in descriptors:
                    if descriptor in all_descriptors or descriptor.is_stop: continue
                    if descriptor.head.text == feature:
                        full_description = []
                        # Not sure how valid the assumption is that the children will be
                        # in front of the main descriptor
                        for dependent_descriptor in descriptor.children:
                            if not dependent_descriptor.is_stop:
                                all_descriptors.add(dependent_descriptor)
                                full_description.append(dependent_descriptor.text)
                        all_descriptors.add(descriptor)
                        full_description.append(descriptor.text)
                        # This filters out a lot of stray punctuation
                        if not (len(full_description) == 1 and len(full_description[0]) == 1):
                            feature_descriptions.append(full_description)
                feature_descriptors[feature].append(feature_descriptions)
                
    flattened_orphaned_descriptors = [
        descriptor.text
             for listing_orphaned_descriptors in listings_orphaned_descriptors
                 for descriptors in listing_orphaned_descriptors
                     for descriptor in descriptors

    ]

    preferred_descriptor_spellings = generate_preferred_spelling_dict(
        [descriptor.text for descriptor in all_descriptors] +
        flattened_orphaned_descriptors
    )
    top_descriptors = {feature:[] for feature in most_popular_features}
    for feature, listings in feature_descriptors.items():
        flattened_indirect_descriptor_phrase_list = []
        for listing in listings:
            for description in listing:
                # This will unfortunately put spaces around hyphens, and that sort of thing
                text_description = ' '.join([preferred_descriptor_spellings[descriptor.lower()][0] for descriptor in description])
                flattened_indirect_descriptor_phrase_list.append(text_description)
        preferred_descriptions = list(generate_multiplicity_dict(flattened_indirect_descriptor_phrase_list).items())

        top_descriptors[feature] = preferred_descriptions
        
    for feature, descriptors in top_descriptors.items():
        descriptors.sort(key=lambda desc: desc[1], reverse=True)
        top_descriptors[feature] = descriptors[:5]

    true_orphans = []
    for orphaned_descriptor in flattened_orphaned_descriptors:
        if len(orphaned_descriptor) == 1: continue
        orphaned_descriptor = preferred_descriptor_spellings[orphaned_descriptor.lower()][0]
        if not reassociate_orphaned_descriptor(orphaned_descriptor, top_descriptors):
            true_orphans.append(orphaned_descriptor)

    preferred_orphan_descriptors = list(generate_multiplicity_dict(true_orphans).items())
    preferred_orphan_descriptors.sort(key=lambda desc: desc[1], reverse=True)

    top_descriptors['Type'] = preferred_orphan_descriptors[:5]

    # re-sort since some multiplicities may have been incremented
    for feature, descriptors in top_descriptors.items():
        descriptors.sort(key=lambda desc: desc[1], reverse=True)
        top_descriptors[feature] = descriptors

    return top_descriptors
