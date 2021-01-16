import tensorflow_hub as hub
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils import load_one_transcript, load_all_transcripts, load_pretrained_glove
from predictDA import get_all_annotated_transcripts
from tqdm import tqdm
from gensim.models import word2vec
from analyse_transcripts import enhance_transcript_df
import pandas as pd
import config
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader import NOUN
from nltk.stem.wordnet import WordNetLemmatizer
from polyglot.text import Text

pd.options.display.width = 0
pd.options.display.max_rows = 100

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
#%%
if __name__ == "__main__":

    #%%
    #load glove:
    glove = load_pretrained_glove("../embeddings/glove.840B.300d.txt")

    #%%
    transcript_dfs = get_all_annotated_transcripts(force_rebuild=False)
    for transcript_df in transcript_dfs:
        enhance_transcript_df(transcript_df)
        transcript_df["utterance"] = transcript_df["utterance"].str.replace(" ' ", "'")
        transcript_df["utterance"] = transcript_df["utterance"].str.replace(" ’ ", "’")
    tdf = transcript_dfs[2]
    filler_das = ['Appreciation', 'Agree/Accept', 'Acknowledge (Backchannel)',
        'Repeat-phrase', 'Yes answers', 'Response Acknowledgement',
        'Affirmative non-yes answers', 'Backchannel in question form',
        'Negative non-no answers', 'Uninterpretable', 'Signal-non-understanding',
        'Hold before answer/agreement', 'Action-directive', 'Thanking']

    text = " ".join(tdf["utterance"])

    def in_word_net(word, Lem):
        '''
        checks whether the lemmatized given word is found in wordNet, not
        only exact matches are kept!

        PARAMS:
            str word: word to check
            WordNetLemmatizer Lem
        '''
        return len(wn.synsets(word, NOUN)) > 0

    def add_by_pos(keywords, tokens, Lem, topic_pos_tags, all_nouns):
        for word, pos in nltk.pos_tag(tokens):
            if (pos in topic_pos_tags and len(word) > 1): #one letter words are always false positives
                if pos in ["NN", "NNS"]:
                    #wordnet does not have plurals, roughly remove plurals
                    word = Lem.lemmatize(word)
                    if word in all_nouns:
                        keywords.add(word)
                else:
                    # either NNP (proper noun) or number, might not be in wordNet
                    keywords.add(word)
        return keywords

    def add_bi_grams(keywords, tokens, Lem):
        # for two word combinations, such as "neural net", check if its in synset
        for words in zip(tokens[::2], tokens[1::2]):
            combined = "_".join(words)
            if in_word_net(combined, Lem):
                keywords.add(combined)
        return keywords

    def add_by_ner(keywords, tokens):

        # if recognised as a named entity, add to key words
        text = Text(" ".join(tokens))
        text.language = "en" #set language, otherwise sometimes gets confused
        for e in text.entities:
            word = "_".join(text.words[e.start:e.end])
            if not ("'" in word or "’" in word): #weirdly, words with apostrophes mess with NER
                keywords.add(word)
        return keywords

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
        #TODO: add documentation for how to install NER/POS libraries
        #TODO: clean this up, put everything in smaller functions etc
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
        manual_filter_words = set(["get", "thing", "man", "go", "okay", "“", "Don",
                                   "nobody", "are", "wow", "woah", "whoa", "perfect",
                                   "way", "guy", "stuff", "day", "iteration", "bit",
                                   "inch", "meter", "millimeter", "centimeter", "yard",
                                   "kilometer", "mile", "foot"])
        #Don't is classified as Don (name)


        topic_pos_tags = set(["NN", "NNP", "NNS"]) #nouns, proper nouns, plural nouns

        stop_words = set(stopwords.words('english'))
        Lem = WordNetLemmatizer()

        #get all nouns in wordNet
        all_nouns = set([s.name().split(".")[0] for s in wn.all_synsets('n')])

        sentence_keywords = []
        for entry in tdf["utterance"]:
            keywords = set() #set because want unique
            tokens = nltk.word_tokenize(entry)
            tokens = [t for t in tokens if t not in stop_words] #filter out stop words
            keywords = add_by_pos(keywords, tokens, Lem, topic_pos_tags, all_nouns)
            keywords = add_by_ner(keywords, tokens)
            keywords = add_bi_grams(keywords, tokens, Lem)
            keywords = remove_manual_filter_words(keywords, manual_filter_words, Lem)
            keywords = remove_partial_words(keywords)
            sentence_keywords.append(keywords)

        tdf["key_words"] = sentence_keywords
        tdf.loc[~tdf["key_words"].astype(bool), "key_words"] = None
        tdf.loc[tdf["da_label"].isin(filler_das), "key_words"] = None

        # temporarily set all entries without keywords to have the same keywords as previous:
        current_keywords = ["NO KEYWORDS YET"]
        for i, keywords in enumerate(tdf["key_words"]):
            if keywords:
                current_keywords = keywords
            else:
                tdf.at[i, "key_words"] = current_keywords
        return tdf


    tdf = add_topic_words(tdf)

    tdf.head(50)

    #%%
    keywords = tdf["key_words"]
    min_sim = 0.7
    min_topic_length = 10

    from itertools import product

    def renumerate(sequence, start=None):
        if start is None:
            start = len(sequence) - 1
        n = start
        for elem in sequence[::-1]:
            yield n, elem
            n -= 1

    def get_matches(current_kws, next_kws, min_sim, kw_glove):

        matches = []
        for kw1, kw2 in product(current_kws, next_kws):
            if (kw1==kw2 or
                cosine_similarity(kw_glove.get(kw1), kw_glove.get(kw2)) > min_sim):

                matches.append((kw1, kw2))
        if len(matches) == 0:
            return False
        return matches

    get_matches(set(["baby", "day", "world", "tesla"]), set(["child", "tesla"]), 0.6, kw_glove)
    get_matches(["child" , "world"], ["child", "life", "time"], 0.7, kw_glove)
    boundaries = []
    topics = []

    #ASSUMPTION: only one topic at a time
    all_keywords = set().union(*keywords)
    kw_glove = {kw : glove[kw] for kw in all_keywords if kw in glove}
    #
    prev_kws = keywords[0]
    prev_topic = keywords[0]
    skips = 0

    keyword_iter = iter(enumerate(keywords[0:500]))
    for i, current_kws in keyword_iter: #TODO: when finalising, only let list go to end - min_topic_length

        if prev_kws == current_kws:
            continue #when exact same key words, continue

        if skips > 0:
            #print(skips)
            skips -= 1
            #print("skipping over ", current_kws)
            prev_kws = current_kws
            continue


        #print("next: ", next_kws)
        #print("prev: ", prev_kws)

        #Now check potential boundaries:
        for j, next_kws in renumerate(keywords[i: i + min_topic_length], i + min_topic_length - 1):#go backwards because we can cancel early
            #print("looking for match with: ", next_kws, j)
            matches = get_matches(prev_kws, next_kws, min_sim, kw_glove)
            if matches:
                #print(matches)
                #print("match found, ", j-i, " positions ahead")
                skips = j - i #skip this many iterations
                break


        if not matches:
            #print("no match found: adding boundary at ", i)
            boundaries.append(i)

        prev_kws = current_kws
#%%
c = 0
tdf[boundaries[c]:boundaries[c+1]]
#%%
boundaries[0:5]
tdf.head(50)
