import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils import load_one_transcript

#%%
class Network:
    def __init__(self, cutoff):
        self.edges = []
        self.nodes = []
        self.clusters = []
        self.cutoff = cutoff

    def add_node(self, text, embedding):
        self.nodes.append(Node(text, embedding))

    def make_connections(self):
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes[i+1:]):
                self.add_edge(n1, n2)

    def get_clusters(self):
        for n in self.nodes:
            if not n.visited:
                self.clusters.append([]) #append empty list, then add nodes while crawling
                self.crawl(n)

    def crawl(self, n):
        n.visited = True
        self.clusters[-1].append(n) #add to latest cluster
        for neighbour_n in n.connected:
            if not neighbour_n.visited:
                self.crawl(neighbour_n)


    def add_edge(self, n1, n2):
        weight = np.inner(n1.embedding, n2.embedding)
        if weight > self.cutoff:
            self.edges.append(Edge((n1, n2), weight))
            n1.connected.append(n2)
            n2.connected.append(n1)

class Node:
    def __init__(self, text, embeddings):
        self.text = text
        self.embedding = embeddings

        self.visited = False # for clustering
        self.connected = []

    def __repr__(self):
        return "Text: " + self.text

class Edge:
    def __init__(self, nodes, weight):
        self.nodes = nodes
        self.weight = weight

    def __repr__(self):
        return ("(nodes: (" + str(self.nodes[0]) + str(self.nodes[1]) + "), " +
            str(self.weight) + ")")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))

def make_similarity_matrix(labels, features):
    return np.inner(features, features)

def plot_similarity(labels, features, rotation):
    plt.figure(figsize = (20, 12))
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.clustermap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.ax_heatmap.set_title("Clustered Semantic Textual Similarity")

#some test sentences
sentences = [
    "Hi, when are you back?",
    "I'll be back in half an hour.",
    "Traffic is awful!"
]
sentences = [
    # Smartphones
    "I like my phone",
    "My phone is not good.",
    "Your cellphone looks great.",

    # Weather
    "Will it snow tomorrow?",
    "Recently a lot of hurricanes have hit the US",
    "Global warming is real",

    # Food and health
    "An apple a day, keeps the doctors away",
    "Eating strawberries is healthy",
    "Is paleo better than keto?",

    # Asking about age
    "How old are you?",
    "what is your age?",
]

if __name__ == "__main__":
    #load transcript
    #%%
    print("loading sentence embedder, this takes a while.")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Done")
    #%%
    fpath = "../transcripts/joe_rogan_elon_musk.txt"
    transcript = load_one_transcript(fpath, chunked = False)[:100]
    sentences = [s[0] for s in transcript]
    #embed using google sentence encoder
    embeddings = embed(sentences)

    #approach 1: use density based clustering algorithm to cluster sentences into topics
    sim_m = make_similarity_matrix(sentences, embeddings)

    cluster_labels = DBSCAN(eps = 1.4, min_samples = 1).fit_predict(sim_m) #eps is sensitivity

    clusters = defaultdict(list)
    for cl, sentence in zip(cluster_labels, sentences):
        clusters[cl].append(sentence)

    pprint(list(clusters.values())) #these are sentence clusters
    plot_similarity(sentences, embeddings, 90) #visualised here

    # approach 2: use network, every sentence is node, connections above cutoff
    # similarity, clusters in graph are clusters of topics
    # TODO: filter backchannels, agreements etc (DAs without content)
    # TODO: decay factor for time delay, penalise large time difference of sentences
    # TODO: adjust cutoff

    #network = Network(cutoff = 0.20)

    #for s, e in zip(sentences, embeddings):
    #    network.add_node(s, e)

    #network.make_connections()
    #network.get_clusters()

    #pprint(network.clusters)
    #pprint(network.edges)
