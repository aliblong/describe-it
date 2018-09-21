
# coding: utf-8

# In[2]:


import os
import sys
import regex as re
import logging

import requests
from bs4 import BeautifulSoup

import numpy as np
import nltk as nl
import pandas as pd
import sklearn as sk

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


# In[ ]:


def parse_ad(html): # Parses ad html trees and sorts relevant data into a dictionary
    ad_info = {}

    #description = html.find('div', {"class": "description"}).text.strip()
    #description = description.replace(html.find('div', {"class": "details"}).text.strip(), '')
    #print(description)
    try:
        ad_info["Title"] = html.find('a', {"class": "title"}).text.strip()
    except:
        logging.error('Unable to parse Title data.')

    try:
        ad_info["Image"] = str(html.find('img'))
    except:
        logging.error('Unable to parse Image data')

    try:
        ad_info["Url"] = 'http://www.kijiji.ca' + html.get("data-vip-url")
    except:
        logging.error('Unable to parse URL data.')

    try:
        ad_info["Details"] = html.find('div', {"class": "details"}).text.strip()
    except:
        logging.error('Unable to parse Details data.')

    try:
        description = html.find('div', {"class": "description"}).text.strip()
        description = description.replace(ad_info["Details"], '')
        ad_info["Description"] = description
    except:
        logging.error('Unable to parse Description data.')

    try:
        ad_info["Date"] = html.find('span', {"class": "date-posted"}).text.strip()
    except:
        logging.error('Unable to parse Date data.')

    try:
        location = html.find('div', {"class": "location"}).text.strip()
        location = location.replace(ad_info["Date"], '')
        ad_info["Location"] = location
    except:
        logging.error('Unable to parse Location data.')

    try:
        ad_info["Price"] = html.find('div', {"class": "price"}).text.strip()
    except:
        logging.error('Unable to parse Price data.')

    return ad_info


def WriteAds(ad_dict, con):  # Writes ads from given dictionary to given file
    


def ReadAds(outfile):  # Reads given file and creates a dict of ads in file
    import ast
    ad_dict = {}
    if os.path.exists(outfile):
        with open(outfile, 'r') as fh:
            ad_dict = json.loads(fh.read())

    return ad_dict

def scrape(url, old_ad_dict, exclude_list, filename, send_email):  # Pulls page data from a given kijiji url and finds all ads on each page
    # Initialize variables for loop
    email_title = None
    ad_dict = {}
    third_party_ad_ids = []

    while url:

        try:
            page = requests.get(url) # Get the html data from the URL
        except:
            print("[Error] Unable to load " + url)
            sys.exit(1)

        soup = BeautifulSoup(page.content, "html.parser")

        if not email_title: # If the email title doesnt exist pull it form the html data
            email_title = soup.find('div', {'class': 'message'}).find('strong').text.strip('"')
            email_title = to_upper(email_title)

        kijiji_ads = soup.find_all("div", {"class": "regular-ad"})  # Finds all ad trees in page html.

        third_party_ads = soup.find_all("div", {"class": "third-party"}) # Find all third-party ads to skip them
        for ad in third_party_ads:
            third_party_ad_ids.append(ad['data-ad-id'])


        exclude_list = to_lower(exclude_list) # Make all words in the exclude list lower-case
        #checklist = ['miata']
        for ad in kijiji_ads:  # Creates a dictionary of all ads with ad id being the keys.
            title = ad.find('a', {"class": "title"}).text.strip() # Get the ad title
            ad_id = ad['data-ad-id'] # Get the ad id
            if not [False for match in exclude_list if match in title.lower()]: # If any of the title words match the exclude list then skip
                #if [True for match in checklist if match in title.lower()]:
                if (ad_id not in old_ad_dict and ad_id not in third_party_ad_ids): # Skip third-party ads and ads already found
                    logging.info('New ad found! Ad id: ' + ad_id)
                    ad_dict[ad_id] = parse_ad(ad) # Parse data from ad
        url = soup.find('a', {'title' : 'Next'})
        if url:
            url = 'https://www.kijiji.ca' + url['href']

    if ad_dict != {}:  # If dict not emtpy, write ads to text file and send email.
        WriteAds(ad_dict, filename) # Save ads to file
        if send_email:
            MailAd(ad_dict, email_title) # Send out email with new ads

def to_lower(input_list): # Rturns a given list of words to lower-case words
    output_list = list()
    for word in input_list:
        output_list.append(word.lower())
    return output_list

def to_upper(title): # Makes the first letter of every word upper-case
    new_title = list()
    title = title.split()
    for word in title:
        new_word = ''
        new_word += word[0].upper()
        if len(word) > 1:
            new_word += word[1:]
        new_title.append(new_word)
    return ' '.join(new_title)

def main():
    parser = argparse.ArgumentParser(description='Scrape ads from a Kijiji URL')
    outfile_default = 'scraped_ads.json'
    parser.add_argument(
        '--url', '-u',
        dest='url',
        type=str,
        required=True,
        help='URL to scrape',
    )
    parser.add_argument(
        '--outfile', '-f',
        dest='outfile',
        type=str,
        default=outfile_default,
        help='filename to store ads in (default name is {outfile_default})'
    ),
    parser.add_argument(
        '--exclude', '-e',
        dest='exclude_list',
        nargs='*',
        type=str,
        default=[],
        help='ads containing one of the strings in this list are excluded'
    )
    parser.add_argument(
        '-send_email', '-s',
        dest='send_email',
        type=bool,
        default=False,
        help='Email the output to a hardcoded address in the script'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help='',
        dest='verbose',
        action='store_true'
    )
            #filename = args.pop(args.index('-f') + 1)
            #filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            #args.remove('-f')
    args = parser.parse_args()

    old_ad_dict = ReadAds(args.outfile)
    level = 'INFO' if args.verbose else 'ERROR'
    logging.basicConfig(level=level)
    logging.info('Ad database succesfully loaded.')
    scrape(args.url, old_ad_dict, args.exclude_list, args.outfile, args.send_email)


# Reads in some listings

# In[4]:


def connect_db():
    db_url = os.getenv('DB_URL')
    engine = create_engine( 'postgresql://{}', db_url)
    if not database_exists(engine.url):
        create_database(engine.url)
    return engine


# In[10]:


def listings(subject, con):
    # Idk if this is injectable
    sql_query = "SELECT Description FROM {} WHERE delivery_method='Cesarean';", subject
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

# In[10]:


import spacy
nlp_full = spacy.load('en_core_web_lg')
#nlp_tokenizer = spacy.load('en_core_web_lg', disable=['tagger'])


# In[88]:


listings_file_json = '../Kijiji-Scraper/plasma_tvs.json'
my_subject = 'tv'
df = pd.read_json(listings_file_json, orient='index')
# Kijiji uids look like unix timestamps, and afaict there's no way do stop
# pandas interpreting them as such while using orient='index'
#df.index = df.index.astype(np.int64) // 10**9
descs = [row['Description'] for _, row in df.iterrows()]


# In[89]:


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

# In[90]:


capcleaned_descs = [fix_capitalization(desc) for desc in descs]
docs = [nlp_full(desc) for desc in capcleaned_descs]
#capcleaned_descs


# In[91]:


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


# In[92]:


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


# In[93]:


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


# In[94]:


listing_noun_phrase_subjects = [np for listing in listing_noun_phrases for (descriptors, np) in listing]
subject_preferred_spellings = generate_preferred_spelling_dict(listing_noun_phrase_subjects)
popular_descriptors = list(subject_preferred_spellings.items())


# In[95]:


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
    


# In[96]:


def cands_directly_describing_subject(cands, subj_descriptors):
    return [cand for cand in cands if cand.lower() in subj_descriptors]


# In[97]:


# A dictionary of preferred spellings also contains word occurrence multiplicities
def highest_multiplicity_cand(cands, preferred_spellings):
    return max(cands, key=lambda cand: preferred_spellings[cand.lower()][1])


# In[98]:


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


# In[99]:


popular_descriptors.sort(key=lambda desc: desc[1][1], reverse=True)
#popular_descriptors


# In[100]:


popular_brands = [preferred_spelling for (key, preferred_spelling) in preferred_brand_spellings.items()]
popular_brands.sort(key=lambda brand: brand[1], reverse=True)
#popular_brands


# In[101]:


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


# In[102]:


for brand, mult in popular_brands[:15]:
    print(f'{brand} ({mult})')


# In[103]:


# for doc in docs[:3]:
#     spacy.displacy.render(doc, style='dep', jupyter=True)
#     print(doc)
#     print(doc.ents)


# In[104]:


#descs


# In[105]:


# import hunspell
# hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
# brand_model_cands


# In[106]:


#testdoc = nlp_full('42x42 cm black cat')
#spacy.displacy.render(testdoc, style='dep', jupyter = True)


# In[107]:


# vectorizer = sk.feature_extraction.text.CountVectorizer()


# In[108]:


# word_bag = vectorizer.fit_transform(important_words).toarray()
# bag_words = vectorizer.get_feature_names()
# # We don't care if a word appears multiple times in the same listing
# for (index, single_listing_word_multiplicity) in np.ndenumerate(word_bag):
#     if single_listing_word_multiplicity > 1: word_bag[index] = 1
# aggregate_multiplicities = word_bag.sum(axis=0)
# word_multiplicity = list(zip(bag_words, aggregate_multiplicities.tolist()))

# #word_multiplicity = np.column_stack((bag_words, aggregate_multiplicities))


# In[109]:


# # https://stackoverflow.com/a/2828121
# #def sort_words_by_multiplicity(words):
# #    return words[words[:,1].argsort()]

# #sorted_words = sort_words_by_multiplicity(word_multiplicity)
# #sorted_words = sorted_words[::-1]
# #sorted_words[:30]

# word_multiplicity.sort(key=lambda word_multiplicity: word_multiplicity[1])
# word_multiplicity[::-1]


# In[110]:


# def tag_words(text):
#       tokens = nl.word_tokenize(text)
#       return nl.pos_tag(tokens)


# In[111]:


# def ner_words(text):
#       tokens = nl.word_tokenize(text)
#       return nl.chunk.ne_chunk(tokens)


# In[112]:


# chunked_words = [nl.chunk.ne_chunk(desc) for desc in tagged_words]
# chunked_words


# In[113]:


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


# In[114]:


#preferred_spellings = {}
#single_spellings = set()
#for (word, spelling_cands) in spellings.items():
#    n_occurrences = len(spelling_cands)
#    preferred_spelling = max(set(spelling_cands), key=spelling_cands.count)
#    preferred_spellings[word] = (preferred_spelling, n_occurrences)
# preferred_spellings

