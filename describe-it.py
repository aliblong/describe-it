
# coding: utf-8

# In[64]:


import numpy as np
import nltk as nl
import pandas as pd
import sklearn as sk
import regex as re


# Reads in some listings

# In[25]:


listings_file_csv = '../Kijiji-Scraper/plasma_tvs.csv'
df = pd.read_csv(listings_file_csv)
#listings_file_json = 'Kijiji-Scraper/bikes.json'
#df = pd.read_json(listings_file_json, orient='index')
# Kijiji uids look like unix timestamps, and afaict there's no way do stop
# pandas interpreting them as such while using orient='index'
df.index = df.index.astype(np.int64) // 10**9
descs = [row['Description'] for _, row in df.iterrows()]


# In[26]:


import spacy
nlp_full = spacy.load('en_core_web_lg')
nlp_tokenizer = spacy.load('en_core_web_lg', disable=['tagger'])


# In[27]:


def fix_capitalization(text):
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
    


# In[28]:


capcleaned_descs = [fix_capitalization(desc) for desc in descs]
docs = [nlp_full(desc) for desc in capcleaned_descs]
capcleaned_descs


# In[29]:


tagged_words_spacy = []
for doc in docs:
    tagged_words_spacy.append([(token.text, token.tag_) for token in doc])
    
import itertools
#flattened_tagged_words = list(itertools.chain.from_iterable(tagged_words))
my_subject = 'bike'
my_subject_lower = my_subject.lower()
#print(tagged_words_spacy)
#tagged_words
#brand_model_candidates = [word for sent in tagged_words for (word, tag) in sent if tag in ['NNP', 'NN'] and not word.lower() == subject_lower]
#brand_model_candidates
def is_brand_model_candidate(word, tag, subject_lower):
    return tag in ['NNP'] and word.lower() != subject_lower
brand_model_cands = []
for sent in tagged_words_spacy:
    brand_model_cands.append([word for (word, tag) in sent if is_brand_model_candidate(word, tag, my_subject_lower)])
brand_model_cands


# In[110]:


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


# In[31]:


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
listing_noun_phrases


# In[32]:


listing_noun_phrase_subjects = [np for listing in listing_noun_phrases for (descriptors, np) in listing]
subject_preferred_spellings = generate_preferred_spelling_dict(listing_noun_phrase_subjects)
popular_descriptors = list(subject_preferred_spellings.items())


# In[33]:


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
    


# In[34]:


def cands_directly_describing_subject(cands, subj_descriptors):
    return [cand for cand in cands if cand.lower() in subj_descriptors]


# In[35]:


# A dictionary of preferred spellings also contains word occurrence multiplicities
def highest_multiplicity_cand(cands, preferred_spellings):
    return max(cands, key=lambda cand: preferred_spellings[cand.lower()][1])


# In[36]:


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
brand_names


# In[37]:


popular_descriptors.sort(key=lambda desc: desc[1][1], reverse=True)
popular_descriptors


# In[38]:


popular_brands = [preferred_spelling for (key, preferred_spelling) in preferred_brand_spellings.items()]
popular_brands.sort(key=lambda brand: brand[1], reverse=True)
popular_brands


# In[120]:


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

top_indirect_descriptors   


# In[61]:





# In[40]:


descs


# In[41]:


import hunspell
hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
brand_model_cands


# In[42]:


for doc in docs:
    spacy.displacy.render(doc, style='dep', jupyter=True)
    print(doc)
    print(doc.ents)


# In[121]:


testdoc = nlp_full('42x42 cm black cat')
spacy.displacy.render(testdoc, style='dep', jupyter = True)


# In[ ]:


vectorizer = sk.feature_extraction.text.CountVectorizer()


# In[ ]:


word_bag = vectorizer.fit_transform(important_words).toarray()
bag_words = vectorizer.get_feature_names()
# We don't care if a word appears multiple times in the same listing
for (index, single_listing_word_multiplicity) in np.ndenumerate(word_bag):
    if single_listing_word_multiplicity > 1: word_bag[index] = 1
aggregate_multiplicities = word_bag.sum(axis=0)
word_multiplicity = list(zip(bag_words, aggregate_multiplicities.tolist()))

#word_multiplicity = np.column_stack((bag_words, aggregate_multiplicities))


# In[ ]:


# https://stackoverflow.com/a/2828121
#def sort_words_by_multiplicity(words):
#    return words[words[:,1].argsort()]

#sorted_words = sort_words_by_multiplicity(word_multiplicity)
#sorted_words = sorted_words[::-1]
#sorted_words[:30]

word_multiplicity.sort(key=lambda word_multiplicity: word_multiplicity[1])
word_multiplicity[::-1]


# In[125]:





# In[ ]:


def tag_words(text):
      tokens = nl.word_tokenize(text)
      return nl.pos_tag(tokens)


# In[ ]:


def ner_words(text):
      tokens = nl.word_tokenize(text)
      return nl.chunk.ne_chunk(tokens)


# In[ ]:


chunked_words = [nl.chunk.ne_chunk(desc) for desc in tagged_words]
chunked_words


# In[ ]:


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


# In[ ]:


#preferred_spellings = {}
#single_spellings = set()
#for (word, spelling_cands) in spellings.items():
#    n_occurrences = len(spelling_cands)
#    preferred_spelling = max(set(spelling_cands), key=spelling_cands.count)
#    preferred_spellings[word] = (preferred_spelling, n_occurrences)
# preferred_spellings

