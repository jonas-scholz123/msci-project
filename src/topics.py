import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_all_transcripts, load_pretrained_glove
from predictDA import get_all_annotated_transcripts
from tqdm import tqdm
import nltk
from pprint import pprint
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader import NOUN
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from itertools import product
from queue import Queue
from sklearn.decomposition import PCA
import pandas as pd

from flair.data import Sentence
from flair.models import MultiTagger

import config


class TopicNode():

    def __init__(self, topic, start, end):
        self.topic = topic
        self.start = start
        self.end = end
        self.neighbours = set()
        self.visited = False


class TopicNetwork():
    def __init__(self):
        self.nodes = []
        self.edges = set()

    def add_node(self, topic, start, end):
        n = TopicNode(topic, start, end)
        self.nodes.append(n)
        return n

    def connect_nodes(self, n1, n2):
        self.edges.add((n1, n2))
        n1.neighbours.add(n2) #directed


class TopicExtractor:

    def __init__(self):
        self.models_loaded = False
        # init lematizer
        self.Lem = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.filler_das = config.topics["filler_das"]
        self.manual_filter_words = config.topics["manual_filter_words"]

        #things needed for topic segmentation
        self.max_gap = config.topics["max_gap"]
        self.min_sim = config.topics["min_sim"]
        self.min_topic_length = config.topics["min_topic_length"]
        self.min_overlap = 0.2
        #load glove: ~ 9s
        print("loading glove, this takes a while")
        self.glove = load_pretrained_glove("../embeddings/glove.840B.300d.txt")
        self.pca = None

    def init_models(self):
        print("Initialising topic models, this takes a while.")
        self.tagger = MultiTagger.load(['pos-fast', 'ner-ontonotes-fast'])

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

        if not self.models_loaded:
            self.init_models()
            self.models_loaded = True
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

    def get_end_of_topic(self, key_words, topic_set, orig_word, start_index):
        '''
        goes through following keywords (within max_gap) and checks if matches
        are found, if so, calls itself recursively to check matching keyword.

        if no matches are found within the next max_gap sentences, returns the
        starting point
        '''
        for j, next_kws in key_words.loc[start_index + 1:].iteritems():
            if j - start_index > self.max_gap:
                return start_index, topic_set
            # topic is all matching keywords with original word
            matches = self.get_matches(set([orig_word]), next_kws)
            if matches:
                printable = False
                topic_set = topic_set.union(matches)
                return self.get_end_of_topic(key_words, topic_set,
                                             orig_word, j)
        return key_words.index[-1], topic_set  # only reaches this point at end of convo

    def add_topics(self, tdf, return_topic_ranges=False):
        #self = te
        topics = defaultdict(list)
        topic_ranges = defaultdict(list)

        key_words = tdf[~tdf["key_words"].isnull()]["key_words"]
        self.all_keywords = set().union(*[k for k in key_words if k])
        self.kw_glove = {kw: self.glove[kw]
                         for kw in self.all_keywords if kw in self.glove}
        for i, kws in key_words.iteritems():
            for topic_word in kws:
                # Always take maximum topic, range, don't check again if topic
                # already marked for a range containing i
                if (topics[topic_word]
                    and topics[topic_word][-1][0] <= i
                        and topics[topic_word][-1][1] >= i):
                    continue

                topic_set = set([topic_word])
                end, topic_set = self.get_end_of_topic(key_words,
                                                       topic_set, topic_word, i)
                if end - i >= self.min_topic_length:
                    topic_ranges[(i, end)].append(topic_set)
                    for tw in topic_set:
                        topics[tw].append((i, end))
        clustered_topics = self.get_clustered_topics(topic_ranges)
        tdf["topics"] = [list() for _ in range(len(tdf))]

        # append topics
        for (start, end), topics in clustered_topics.items():
            for topic in topics:
                tdf.loc[start:end, "topics"] = tdf.loc[start:end, "topics"].apply(
                                                    lambda x: x + [topic])

        # empty topic fields are filled w previous topics
        tdf["topics"] = tdf["topics"].apply(lambda x: np.nan if len(x) == 0
                                            else x)
        tdf["topics"].fillna(method="ffill", inplace=True)
        isna = tdf['topics'].isna()
        # replace nans with empty lists again
        tdf.loc[isna, 'topics'] = pd.Series([[]] * isna.sum()).values
        if return_topic_ranges:
            return tdf, clustered_topics
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
        if len(matches) == 0:
            return False
        return matches

    def get_clustered_topics(self, topic_ranges):
        #self = te

        tr_list = list(topic_ranges.items())
        net = TopicNetwork()

        for tr, topics in tr_list:
            for t in topics:
                net.add_node(t, tr[0], tr[1])

        for i, n1 in enumerate(net.nodes):
            topic = n1.topic
            for n2 in net.nodes[i + 1:]:
                if n1.end < n2.start:
                    break
                t1 = n1.topic
                t2 = n2.topic

                matches = self.get_matches(t1, t2)
                topic_length = len(t1) + len(t2)
                if matches and len(matches) >= self.min_overlap * topic_length:
                    net.connect_nodes(n1, n2)

        true_topics = defaultdict(list)
        for n in net.nodes:
            if n.visited:
                continue
            t, tr = self.cluster_topic_nodes(n)
            true_topics[tr].append(t)
        return true_topics

    def cluster_topic_nodes(self, n):
        q = Queue()
        topic = n.topic
        start = n.start
        end = n.end
        q.put(n)

        while not q.empty():
            n.visited = True
            matches = self.get_matches(topic, n.topic)
            if not matches:
                n = q.get()
                continue
            # topic = topic.union(n.topic)
            topic = topic.union(matches)
            end = max(end, n.end)
            for next in n.neighbours:
                if not next.visited:
                    q.put(next)
            n = q.get()
        return topic, (start, end)

    def get_all_topics(self, tdf):
        all_topics = set()
        for topics in tdf["topics"]:
            for t in topics:
                if type(t) == set:
                    all_topics.add(frozenset(t))
        return all_topics

    def fit_n_d_embeddings(self, tdf, n):
        all_topics = self.get_all_topics(tdf)

        pca = PCA(n)
        embeds = {}
        for t in all_topics:
            topic_embeds = []
            for word in t:
                embed = self.glove.get(word)
                if embed is not None:
                    topic_embeds.append(embed)
            if topic_embeds:
                embeds[frozenset(t)] = np.array(topic_embeds).mean(axis=0)
        self.pca = pca.fit(list(embeds.values()))
        return list(embeds.values())

    def get_n_d_embedding(self, topic, n):
        if self.pca is None:
            print("No pca fitted yet, run fit_n_d_embeddings()")
            return
        embeds = [self.glove[w] for w in topic if w in self.glove.keys()]

        if embeds:
            mean = np.array(embeds).mean(axis=0)
            return self.pca.transform([mean])[0][0]
        return False

    def cosine_similarity(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0
        vec1 = np.asarray(vec1, dtype = np.int32)
        vec2 = np.asarray(vec2, dtype = np.int32) #8 bits overflow

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
    tdf = pd.read_pickle("../processed_transcripts/joe_rogan_elon_musk.pkl")
    tdf, tr = te.add_topics(tdf, return_topic_ranges=True)
