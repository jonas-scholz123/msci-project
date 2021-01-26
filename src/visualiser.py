import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.patches import Ellipse, Circle, Arrow, ConnectionPatch
import numpy.random as rnd
import numpy as np
import pandas as pd
import config
from matplotlib import cm
import matplotlib as mpl

from utils import load_pretrained_glove

from collections import defaultdict

class Visualiser():

    def __init__(self, max_nodes):

        #Sentence to node
        self.s2n = {}

        #stores all nodes
        self.nodes = []

        #position data
        #self.y = 0
        #self.delta_y = 1 #depth gained per timestep
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.delta_y = 2 #height difference between topics

        self.width = 10

        self.node_radius = 0.3

        self.fig = plt.figure(0, figsize = (150, 2))
        self.ax = self.fig.add_subplot(111, aspect='equal')

        self.nr_objects = np.zeros(max_nodes)
        self.lowest_free_y = np.zeros(max_nodes)

    def add_node(self, x, y):
        node = Circle((x, y), self.node_radius)
        self.ax.add_artist(node)
        node.set_facecolor("white")
        node.set_edgecolor("black")
        self.nodes.append(node)

        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)
        return node

    def add_section(self, x0, x1, s, color):
        y = self.lowest_free_y[x0:x1].max() * self.delta_y

        n0 = self.add_node(x0, y)
        n1 = self.add_node(x1, y)
        self.connect_nodes(n0, n1, color = color)

        fontsize = max(min(8, (x1-x0)/len(s) * 8), 4)
        plt.annotate(s, ((x1+x0)/2, y + self.delta_y/3), color=color,
            fontsize=fontsize, ha="center")

    def set_axes(self):
        self.ax.set_xlim(self.min_x - 5, self.max_x + 5)
        self.ax.set_ylim(self.min_y - 1, self.max_y + 3)

    def show_fig(self):
        #self.fig.show()
        self.set_axes()
        plt.show()

    def save_fig(self, name):
        fpath = config.paths["figures"] + name + "_topics.pdf"
        self.set_axes()
        plt.savefig(fpath)

    def connect_nodes(self, n0, nf, color = None):
        x0, y0 = n0.center
        xf, yf = nf.center

        self.nr_objects[int(x0) : int(xf)] += 1

        for i, lfy in enumerate(self.lowest_free_y[int(x0) : int(xf)], int(x0)):
            if lfy >= y0/self.delta_y:
                self.lowest_free_y[i] += self.delta_y

        con = ConnectionPatch(n0.center, nf.center, "data", "data", zorder = 0)
        con.set_linewidth(4)
        if color:
            con.set_edgecolor(color)
        self.ax.add_artist(con)

def idx_to_ranges(idx):
    starts = [idx[0]]
    ends = []

    prev = idx[0]
    for j, index in enumerate(idx[1:]):
        if index == prev + 1:
            prev = index
            continue
        else:
            prev = index
            ends.append(idx[j])
            starts.append(index)
    ends.append(idx[-1])
    return list(zip(starts, ends))

def get_sorted_topics(topic_ranges):
    topic_occurences = {}
    for t, trs in topic_ranges.items():
        cum_sum = 0
        for tr in trs:
            cum_sum += tr[1] - tr[0]
        topic_occurences[t] = cum_sum
    return sorted(list(topic_occurences.items()), key=lambda x: x[1], reverse=True)

def get_topic_colors(topic_occurences):
    topic_colors = {}
    color_count = 0
    nr_colors = 10
    for topic, _ in topic_occurences:
        color_count += 1
        color = "C" + str(color_count % nr_colors)
        topic_colors[topic] = color
    return topic_colors

def cluster_sub_topics(inverse_topic_ranges, kw_glove):
    new_itr = defaultdict(list)
    itr_tuples = sorted(list(inverse_topic_ranges.items()),
        key=lambda x:(x[0][0], -x[0][1])) #sort to have low first, high second index
        #this means we have the largest range items first
    skips = 0
    for i, (tr, words) in enumerate(itr_tuples):
        if skips > 0:
            skips -= 1
            continue

        start = tr[0]
        [new_itr[tr].append(w) for w in words]


        for j, (next_tr, next_words) in enumerate(itr_tuples[i + 1:], i+1):

            #if same starting position, given that end position is smaller because of sorting
            #if next_tr[1] <= tr[1] and next_tr[0] >= tr[0]:
            if next_tr[1] == tr[1]:
                [new_itr[tr].append(w) for w in next_words]
                skips += 1
            else:
                #no more same starting positions, break and calculate skips
                #skips = j - i - 1
                break


    return {k : list(set(v)) for k, v in new_itr.items()}


def invert_dict_of_lists(d):
    new_dict = defaultdict(list)
    for key, l in d.items():
        for listitem in l:
            new_dict[listitem].append(key)

    return new_dict

def make_1d_embeddings(kws_embeddings):
    pca = PCA(1)
    embeddings_1d = pca.fit_transform(kws_embeddings)
    return np.array([e[0] for e in embeddings_1d]), pca

def get_colormap(embeddings_1d):

    #normalise
    #embeddings_1d += abs(embeddings_1d.min())
    #embeddings_1d /= embeddings_1d.max()

    #tups = list(zip(kws_embeddings.keys(), embeddings_1d))
    #sorted_tups = sorted(tups, key = lambda x: x[1])


    norm = mpl.colors.Normalize(embeddings_1d.min(), embeddings_1d.max())
    cmap = cm.tab20c
    return cm.ScalarMappable(norm=norm, cmap=cmap)

import pandas as pd
#%%
glove = load_pretrained_glove("../embeddings/glove.840B.300d.txt")
#%%
#transcript_name = "joe_rogan_elon_musk"
transcript_name = "sam_harris_nicholas_christakis"
tdf = pd.read_pickle("../processed_transcripts/" + transcript_name + ".pkl")
tdf
pd.options.display.max_rows = None
tdf["topic_count"] = tdf["topics"].apply(lambda x: len(x) if type(x) == list else 0)

topic_ranges = {}

topic_words = set(sum([t for t in tdf["topics"] if type(t) == list], []))
topic_strings = tdf["topics"].apply(lambda x:
        "," + ",".join(x) + "," if type(x) == list else "")
for tw in topic_words:
    topic_ranges[tw] = idx_to_ranges(
        topic_strings[topic_strings.str.contains("," + tw + ",")].index) #"," ensures full word

inverse_topic_ranges = invert_dict_of_lists(topic_ranges)
inverse_topic_ranges = cluster_sub_topics(inverse_topic_ranges)

tcs = get_topic_colors(get_sorted_topics(topic_ranges))
vis = Visualiser(len(tdf))
min_topic_length = 1

from sklearn.decomposition import PCA

all_kws = list(set([w for words in inverse_topic_ranges.values() for w in words]))
kws_embeddings = {kw : glove[kw] for kw in all_kws if kw in glove.keys()}

embeddings_1d, pca = make_1d_embeddings(list(kws_embeddings.values()))
kws_embeddings1d = {kw : e for kw, e in zip(list(kws_embeddings.keys()), embeddings_1d)}

scalar_to_color = get_colormap(embeddings_1d)
for tr, words in inverse_topic_ranges.items():
    if tr[1] - tr[0] < min_topic_length:
        continue
    joined = ", ".join(words)

    word_embeddings = [kws_embeddings[w] for w in words if w in kws_embeddings.keys()]
    if word_embeddings:
        embedding_avg_1d = pca.transform(np.array(word_embeddings).mean(axis=0).reshape(1, -1))[0][0]
        color = scalar_to_color.to_rgba(embedding_avg_1d)
    else:
        color = 'grey'
    vis.add_section(tr[0], tr[1], joined, color)

vis.save_fig(transcript_name)
vis.show_fig()


#import pke
#
## initialize keyphrase extraction model, here TopicRank
#extractor = pke.unsupervised.TopicRank()
#
## load the content of the document, here document is expected to be in raw
## format (i.e. a simple text file) and preprocessing is carried out using spacy
#extractor.load_document(input=' '.join(tdf['utterance']), language='en')
#
## keyphrase candidate selection, in the case of TopicRank: sequences of nouns
## and adjectives (i.e. `(Noun|Adj)*`)
#extractor.candidate_selection()
#
## candidate weighting, in the case of TopicRank: using a random walk algorithm
#extractor.candidate_weighting()
#
## N-best selection, keyphrases contains the 10 highest scored candidates as
## (keyphrase, score) tuples
#keyphrases = extractor.get_n_best(n=30)

#keyphrases
