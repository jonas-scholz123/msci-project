import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_all_transcripts, load_pretrained_glove
from predictDA import get_all_annotated_transcripts
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader import NOUN
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from itertools import product

from flair.data import Sentence
from flair.models import MultiTagger

import config


class TopicExtractor:

    def __init__(self):
        print("Initialising topic models, this takes a while.")
        self.tagger = MultiTagger.load(['pos-fast', 'ner-ontonotes-fast'])
        # init lematizer
        self.Lem = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.filler_das = config.topics["filler_das"]
        self.manual_filter_words = config.topics["manual_filter_words"]

        #things needed for topic segmentation
        self.max_gap = config.topics["max_gap"]
        self.min_sim = config.topics["min_sim"]
        #load glove: ~ 9s
        self.glove = load_pretrained_glove("../embeddings/glove.840B.300d.txt")

    def in_word_net(self, word):
        '''
        checks whether the lemmatized given word is found in wordNet, not
        only exact matches are kept!

        PARAMS:
            str word: word to check
            WordNetLemmatizer Lem
        '''
        return len(wn.synsets(self.Lem.lemmatize(word), NOUN)) > 0

    def add_by_ner(self, keywords, sentence):
        for entity in sentence.get_spans('ner-ontonotes-fast'):
            if entity.labels[0].value in ["CARDINAL", "ORDINAL", "TIME", "PERCENT"]:
                continue
            keywords.add(self.Lem.lemmatize(entity.text.lower().replace(" ", "_")))
        return keywords

    def add_by_pos(self, keywords, sentence):
        for entity in sentence.get_spans('pos-fast'):
            pos = entity.labels[0].value
            if pos.startswith("NN"):
                keywords.add(self.Lem.lemmatize(entity.text.lower()))
        return keywords

    def add_all_nouns(self, keywords, tokens):
        for word in tokens:
            word = self.Lem.lemmatize(word)
            if wn.synsets(word, NOUN):
                keywords.add(word)
        return keywords

    def add_bi_grams(self, keywords, tokens):
        # for two word combinations, such as "neural net", check if its in synset
        for words in zip(tokens, tokens[1:]):
            combined = "_".join(words)
            if self.in_word_net(combined):
                keywords.add(self.Lem.lemmatize(combined.lower()))
        return keywords

    def remove_manual_filter_words(self, keywords):
        to_remove = []
        for w in keywords:
            if self.Lem.lemmatize(w.lower()) in self.manual_filter_words:
                to_remove.append(w)
        for w in to_remove: keywords.remove(w) #remove manually filtered words
        return keywords

    def remove_partial_words(self, keywords):
        # remove e.g. "neural" and "net" if talking about neural_nets
        to_remove = set()  # unique
        for kw in keywords:
            if "_" in kw:
                partial_kws = kw.split("_")
            else:
                continue

            for partial_kw in partial_kws:
                if partial_kw in keywords:
                    to_remove.add(partial_kw)
        for w in to_remove:
            keywords.remove(w)
        return keywords

    def remove_len_1_tokens(self, keywords):
        to_remove = set()
        for kw in keywords:
            if len(kw) < 2:
                to_remove.add(kw)
        for w in to_remove:
            keywords.remove(w)
        return keywords

    def add_key_words(self, tdf):
        # TODO: implement tf-idf when have all datasets to remove common words
        # TODO: add documentation for how to install NER/POS libraries
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
        # TODO: Analogies destroy topic, maybe can't do anything about that

        # turn all utterances into Sentence objects for flair
        sentences = [Sentence(text=u) for u in tdf["utterance"]]
        # Predict ~10s (6s using fast)
        self.tagger.predict(sentences)

        sentence_keywords = []
        for sent, text in zip(sentences, tdf["utterance"]):
            keywords = set()  # set because want unique
            tokens = nltk.word_tokenize(text)
            # filter out stop words
            tokens = [t for t in tokens if t not in self.stop_words]
            keywords = self.add_by_pos(keywords, sent)
            keywords = self.add_by_ner(keywords, sent)
            keywords = self.add_bi_grams(keywords, tokens)
            # can use this with tf-idf later, can get rid of POS
            # keywords = add_all_nouns(keywords, tokens, Lem)
            keywords = self.remove_manual_filter_words(keywords)
            keywords = self.remove_partial_words(keywords)
            keywords = self.remove_len_1_tokens(keywords)
            sentence_keywords.append(keywords)

        tdf["key_words"] = sentence_keywords
        tdf.loc[~tdf["key_words"].astype(bool), "key_words"] = None
        tdf.loc[tdf["da_label"].isin(self.filler_das), "key_words"] = None
        return tdf

    def get_end_of_topic(self, key_words, topic_set, start_index):
        '''
        goes through following keywords (within max_gap) and checks if matches
        are found, if so, calls itself recursively to check matching keyword.

        if no matches are found within the next max_gap sentences, returns the
        starting point
        '''
        for j, next_kws in key_words.loc[start_index + 1:].iteritems():
            if j - start_index > self.max_gap:
                return start_index
            matches = self.get_matches(topic_set, next_kws)
            if matches:
                return self.get_end_of_topic(key_words, topic_set, j)
        return key_words.index[-1]  # only reaches this point at end of convo

    def add_topics(self, tdf):
        topics = defaultdict(list)

        key_words = tdf[~tdf["key_words"].isnull()]["key_words"]
        self.all_keywords = set().union(*[k for k in key_words if k])
        self.kw_glove = {kw: self.glove[kw]
                         for kw in self.all_keywords if kw in self.glove}

        for i, kws in key_words.iteritems():
            for topic_word in kws:
                # Always take maximum topic, range, don't check again if topic
                # already marked for a range containing i
                if (topics[topic_word]
                    and topics[topic_word][-1][0] < i
                        and topics[topic_word][-1][1] > i):
                    continue
                topic_set = set([topic_word])
                end = self.get_end_of_topic(key_words, topic_set, i)
                if end != i:
                    topics[topic_word].append((i, end))

        tdf["topics"] = [list() for _ in range(len(tdf))]

        # append topics
        for topic, index_pairs in topics.items():
            for start, end in index_pairs:
                tdf.loc[start: end, "topics"] = tdf.loc[start: end,
                                                        "topics"].apply(
                                                        lambda x: x + [topic])

        # empty topic fields are filled w previous topics
        tdf["topics"] = tdf["topics"].apply(lambda x: np.nan if len(x) == 0
                                            else x)
        tdf["topics"].fillna(method="ffill", inplace=True)
        return tdf

    def get_matches(self, current_kws, next_kws):

        matches = set()
        for kw1, kw2 in product(current_kws, next_kws):
            if (kw1 == kw2
                or self.cosine_similarity(self.kw_glove.get(kw1),
                                          self.kw_glove.get(kw2))
                    > self.min_sim):
                matches.add(kw1)
                matches.add(kw2)
                # print("added to matches: ", kw1, kw2)
        if len(matches) == 0:
            return False
        return matches

    def cosine_similarity(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def process_batch(self, dfs):
        '''
        Wrapper of topic classification for multiple DFs, to cache models etc.
        '''

        if not dfs:
            return []

        # things needed for topics extraction
        # load flair model ~ 11s
        dfs = [self.process(tdf) for tdf in tqdm(dfs)]
        return dfs

    def process(self, tdf):
        if tdf is None:
            return
        return self.add_topics(self.add_key_words(tdf))


def make_similarity_matrix(labels, features):
    return np.inner(features, features)


def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    # fontsize_pt = plt.rcParams['ytick.labelsize']
    # dpi = 72.27
    # matrix_height_pt = fontsize_pt * len(features)
    # matrix_height_in = matrix_height_pt / dpi
    # print("plotting matrix of size ", matrix_height_in)
    plt.figure(figsize=(20, 20))

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
    te = TopicExtractor()

    dfs = get_all_annotated_transcripts()
    tdf = dfs[2]
    dfs = [tdf]

    te.process(dfs)
    # glove = load_pretrained_glove("../embeddings/glove.840B.300d.txt")
    # %%
    # import pandas as pd
    # tdf = pd.read_pickle("../processed_transcripts/joe_rogan_elon_musk.pkl")
    # tdf.head(50)
    # all_kws = list(set([kw for kws in tdf["key_words"] if kws for kw in kws]))
#
    # kws_embeddings = {kw : glove[kw] for kw in all_kws if kw in glove.keys()}
    # gloved_kws = [kw for kw in all_kws if kw]
    # sim_m = make_similarity_matrix(list(kws_embeddings.keys()), list(kws_embeddings.values()))
