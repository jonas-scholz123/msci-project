import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from utils import load_one_transcript, load_all_transcripts, load_pretrained_glove
from predictDA import get_all_annotated_transcripts
from tqdm import tqdm
from analyse_transcripts import enhance_transcript_df
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader import NOUN
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from itertools import product

from flair.data import Sentence
from flair.tokenization import SegtokSentenceSplitter
from flair.models import MultiTagger

pd.options.display.width = 0
pd.options.display.max_rows = 500

import config
#%%
def cosine_similarity(vec1, vec2):
    if vec1 is not None and vec2 is not None:
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))
    return 0

def make_similarity_matrix(labels, features):
    return np.inner(features, features)

def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    #fontsize_pt = plt.rcParams['ytick.labelsize']
    #dpi = 72.27
    #matrix_height_pt = fontsize_pt * len(features)
    #matrix_height_in = matrix_height_pt / dpi
    #print("plotting matrix of size ", matrix_height_in)
    plt.figure(figsize = (20, 20))

    sns.set(font_scale=1.2)
    g = sns.clustermap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.ax_heatmap.set_title("Clustered Semantic Textual Similarity")
    plt.savefig(config.paths["figures"] + "similarity.pdf")

def in_word_net(word, Lem):
    '''
    checks whether the lemmatized given word is found in wordNet, not
    only exact matches are kept!

    PARAMS:
        str word: word to check
        WordNetLemmatizer Lem
    '''
    return len(wn.synsets(word, NOUN)) > 0


def add_by_ner(keywords, sentence):
    for entity in sentence.get_spans('ner'):
        if entity.labels[0].value in ["CARDINAL", "ORDINAL", "TIME", "PERCENT"]:
            continue
        keywords.add(entity.text)
    return keywords

def add_by_pos(keywords, sentence, Lem):
    for entity in sentence.get_spans('pos'):
        pos = entity.labels[0].value
        if pos.startswith("NN"):
            keywords.add(Lem.lemmatize(entity.text))
    return keywords

#def add_by_pos(keywords, tokens, Lem, topic_pos_tags, all_nouns):
#    for word, pos in nltk.pos_tag(tokens):
#        if (pos in topic_pos_tags and len(word) > 1): #one letter words are always false positives
#            if pos in ["NN", "NNS"]:
#                #wordnet does not have plurals, roughly remove plurals
#                word = Lem.lemmatize(word)
#                if word in all_nouns:
#                    keywords.add(word)
#            else:
#                # either NNP (proper noun) or number, might not be in wordNet
#                keywords.add(word)
#    return keywords

def add_all_nouns(keywords, tokens, Lem):
    for word in tokens:
        word = Lem.lemmatize(word)
        if wn.synsets(word, NOUN):
            keywords.add(word)
    return keywords

def add_bi_grams(keywords, tokens, Lem):
    # for two word combinations, such as "neural net", check if its in synset
    for words in zip(tokens, tokens[1:]):
        combined = "_".join(words)
        if in_word_net(combined, Lem):
            keywords.add(combined)
    return keywords

#def add_by_ner(keywords, tokens):
#
#    # if recognised as a named entity, add to key words
#    text = Text(" ".join(tokens))
#    text.language = "en" #set language, otherwise sometimes gets confused
#    for e in text.entities:
#        word = "_".join(text.words[e.start:e.end])
#        if not ("'" in word or "’" in word): #weirdly, words with apostrophes mess with NER
#            keywords.add(word)
#    return keywords

def remove_manual_filter_words(keywords, manual_filter_words, Lem):
    to_remove = []
    for w in keywords:
        if Lem.lemmatize(w.lower()) in manual_filter_words:
            to_remove.append(w)
    for w in to_remove: keywords.remove(w) #remove manually filtered words
    return keywords

def remove_partial_words(keywords):
    #remove e.g. "neural" and "net" if talking about neural_nets
    to_remove = set() #unique
    for kw in keywords:
        if "_" in kw:
            partial_kws = kw.split("_")
        else:
            continue

        for partial_kw in partial_kws:
            if partial_kw in keywords:
                to_remove.add(partial_kw)
    for w in to_remove: keywords.remove(w)
    return keywords


def add_topic_words(tdf):
    #TODO: implement tf-idf when we have all datasets to remove common words
    #TODO: add documentation for how to install NER/POS libraries
    '''
    Extracts and adds (in a new column) words that represent the topic of a
    given sentence. This will later be used for segmentation.

    Uses POS (part of speech) tagging, NER (named entity recognition) and
    wordNet to figure out topic words
    =======================================================================
    PARAMS:
        pd.DataFrame tdf: transcript dataframe
    RETURNS:
        pd.DataFrame tdf: annotated dataframe
    '''
    #TODO: Analogies destroy topic, maybe can't do anything about that

    #singular person words that were wrongly classified as topic words, are filtered out manually
    #Note some words, like "time" are unlikely to be the topic of conversation
    #and more used in sentences such as "last time you did X" or "I remember"
    #a time when..."
    # other words are just nouns that are filler words such as "guy, thing, man"
    # some words are mostly used in combination with "of" such as "the process of"
    # "kind of" "sort of " are filtered
    manual_filter_words = set(["get", "thing", "man", "go", "okay", "“", "Don",
                               "nobody", "are", "wow", "woah", "whoa", "perfect",
                               "way", "guy", "stuff", "day", "iteration", "bit",
                               "inch", "meter", "millimeter", "centimeter", "yard",
                               "kilometer", "mile", "foot", "time",
                               "Does", "process", "lot", "kind", "sort"])
    #Don't is classified as Don (name)


    filler_das = ['Appreciation', 'Agree/Accept', 'Acknowledge (Backchannel)',
        'Repeat-phrase', 'Yes answers', 'Response Acknowledgement',
        'Affirmative non-yes answers', 'Backchannel in question form',
        'Negative non-no answers', 'Uninterpretable', 'Signal-non-understanding',
        'Hold before answer/agreement', 'Action-directive', 'Thanking']


    splitter = SegtokSentenceSplitter()
    sentences = [
        Sentence(
            text=u,
            use_tokenizer = splitter._tokenizer,
            start_position=0
        )
        for u in tdf["utterance"]
    ]


    tagger = MultiTagger.load(['pos', 'ner-ontonotes-fast'])
    tagger.predict(sentences)

    topic_pos_tags = set(["NN", "NNP", "NNS"]) #nouns, proper nouns, plural nouns

    stop_words = set(stopwords.words('english'))
    Lem = WordNetLemmatizer()

    #get all nouns in wordNet
    all_nouns = set([s.name().split(".")[0] for s in wn.all_synsets('n')])

    sentence_keywords = []
    for sent, text in zip(sentences, tdf["utterance"]):
        keywords = set() #set because want unique
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words] #filter out stop words
        #keywords = add_by_pos(keywords, tokens, Lem, topic_pos_tags, all_nouns)
        #keywords = add_by_ner(keywords, tokens)
        keywords = add_by_pos(keywords, sent, Lem)
        keywords = add_by_ner(keywords, sent)
        keywords = add_bi_grams(keywords, tokens, Lem)
        # can use this with tf-idf later, can get rid of POS
        #keywords = add_all_nouns(keywords, tokens, Lem)
        keywords = remove_manual_filter_words(keywords, manual_filter_words, Lem)
        keywords = remove_partial_words(keywords)
        sentence_keywords.append(keywords)

    tdf["key_words"] = sentence_keywords
    tdf.loc[~tdf["key_words"].astype(bool), "key_words"] = None
    tdf.loc[tdf["da_label"].isin(filler_das), "key_words"] = None
    return tdf

def get_end_of_topic(key_words, topic_set, start_index, max_gap, min_sim, kw_glove):
    '''
    goes through following keywords (within max_gap) and checks if matches
    are found, if so, calls itself recursively to check matching keyword etc.

    if no matches are found within the next max_gap sentences, returns the
    starting point
    '''
    for j, next_kws in key_words.loc[start_index + 1:].iteritems():
        if j - start_index  > max_gap:
            return start_index
        matches = get_matches(topic_set, next_kws, min_sim, kw_glove)
        if matches:
            return get_end_of_topic(key_words, topic_set, j, max_gap, min_sim, kw_glove)
    return key_words.index[-1] #only reaches this point at end of convo

def add_topics(tdf, max_gap, min_sim, glove):
    topics = defaultdict(list)

    key_words = tdf[~tdf["key_words"].isnull()]["key_words"]
    all_keywords = set().union(*[k for k in key_words if k])
    kw_glove = {kw : glove[kw] for kw in all_keywords if kw in glove}

    for i, kws in key_words.iteritems():
        for topic_word in kws:
            # Always take maximum topic, range, don't check again if topic
            # already marked for a range containing i
            if (topics[topic_word] and
                topics[topic_word][-1][0] < i and topics[topic_word][-1][1] > i):
                continue
            topic_set = set([topic_word])
            end = get_end_of_topic(key_words, topic_set, i, max_gap, min_sim, kw_glove)
            if end != i:
                topics[topic_word].append((i, end))

    tdf["topics"] = np.empty((len(tdf), 0)).tolist() #set topics to empty list

    tdf["topics"] = ""
    #append topics
    for topic, index_pairs in topics.items():
        for start, end in index_pairs:
            tdf.loc[start : end, "topics"] += topic + ", "

    # no topic fields are filled w previous topics
    tdf.replace("", np.nan, inplace = True)
    tdf["topics"].fillna(method="ffill", inplace = True)


def renumerate(sequence, start=None):
    ''' reverse enumerate'''
    if start is None:
        start = len(sequence) - 1
    n = start
    for elem in sequence[::-1]:
        yield n, elem
        n -= 1

def get_matches(current_kws, next_kws, min_sim, kw_glove):

    matches = set()
    for kw1, kw2 in product(current_kws, next_kws):
        if (kw1==kw2 or
            cosine_similarity(kw_glove.get(kw1), kw_glove.get(kw2)) > min_sim):
            matches.add(kw1)
            matches.add(kw2)
            #print("added to matches: ", kw1, kw2)
    if len(matches) == 0:
        return False
    return matches

if __name__ == "__main__":

    max_gap = 10 #max number of sentences between two topic_word matches for it to no longer be one topic
    min_sim = 0.65


    #load glove:
    glove = load_pretrained_glove("../embeddings/glove.840B.300d.txt")

    transcript_dfs = get_all_annotated_transcripts(force_rebuild=False)
    for transcript_df in transcript_dfs:
        enhance_transcript_df(transcript_df)
        transcript_df["utterance"] = transcript_df["utterance"].str.replace(" ' ", "'")
        transcript_df["utterance"] = transcript_df["utterance"].str.replace(" ’ ", "’")
    tdf = transcript_dfs[2]

#a = tdf.loc[75, "utterance"].split(" ")
    tdf = add_topic_words(tdf)
    add_topics(tdf, max_gap, min_sim, glove)

    tdf.head(500)
#%%

all_keywords = []

for sentence in sentences:
    keywords = set()
    #print(sentence.to_tagged_string())
    #print(sentence.to_dict(tag_type="ner"))
    for entity in sentence.get_spans('ner'):
        if entity.labels[0].value in ["CARDINAL", "ORDINAL", "TIME", "PERCENT"]:
            continue
        keywords.add(entity.text)
        #print("E", entity)

    for entity in sentence.get_spans('pos'):
        if entity.labels[0].value.startswith("NN"):
            keywords.add(entity.text)

    all_keywords.append(keywords)

all_keywords
