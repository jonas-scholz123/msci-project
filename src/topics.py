import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils import load_one_transcript, load_all_transcripts
from predictDA import get_all_annotated_transcripts
from tqdm import tqdm
from analyse_transcripts import enhance_transcript_df
import pandas as pd
import config

pd.options.display.width = 0
pd.options.display.max_rows = None

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

#some test sentences
sentences = [
    "Hi, when are you back?",
    "I'll be back in half an hour.",
    "Traffic is awful!"
]
test_sentences = [
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

    plot_similarity(test_sentences, embed(test_sentences), 90)
    transcript_dfs = get_all_annotated_transcripts(force_rebuild=False)
    for transcript_df in transcript_dfs:
        enhance_transcript_df(transcript_df)
    #%%
    tdf = transcript_dfs[2]
    filler_das = ['Appreciation', 'Agree/Accept', 'Acknowledge (Backchannel)',
        'Repeat-phrase', 'Yes answers', 'Response Acknowledgement',
        'Affirmative non-yes answers', 'Backchannel in question form',
        'Negative non-no answers', 'Uninterpretable', 'Signal-non-understanding',
        'Hold before answer/agreement', 'Action-directive', 'Thanking']

    sentences = tdf["utterance"]
    valid_sentences = tdf[~tdf["da_label"].isin(filler_das)]["utterance"]

    import pke
    from nltk.stem.snowball import SnowballStemmer
    extractor = pke.unsupervised.TopicRank()

    extractor.load_document(" ".join(valid_sentences))

    extractor.candidate_selection(pos={"NOUN", "PROPN"})

    stem = SnowballStemmer("english").stem

    sentence_keywords = []

    for entry in tdf["utterance"]:
        keywords = []
        for word in entry.split():
            if stem(word.lower()) in extractor.candidates:
                keywords.append(word)
        sentence_keywords.append(keywords)

    tdf["key_words"] = sentence_keywords

    tdf

#%%
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

transcript_dfs = get_all_annotated_transcripts(force_rebuild=False)
for transcript_df in transcript_dfs:
    enhance_transcript_df(transcript_df)

tdf = transcript_dfs[2]

filler_das = ['Appreciation', 'Agree/Accept', 'Acknowledge (Backchannel)',
    'Repeat-phrase', 'Yes answers', 'Response Acknowledgement',
    'Affirmative non-yes answers', 'Backchannel in question form',
    'Negative non-no answers', 'Uninterpretable', 'Signal-non-understanding',
    'Hold before answer/agreement', 'Action-directive', 'Thanking']

stopwords = set(stopwords.words())


sentence_keywords = []
for entry in tdf["utterance"]:
    keywords = []
    for word, pos in nltk.pos_tag(nltk.word_tokenize(entry)):
        #sometimes 1 letter words are misclassified as nouns and added
        if (pos in ["NN", "NNP", "NNS", "CD"] and len(word) > 1
            and word not in stopwords):

            keywords.append(word)
    sentence_keywords.append(keywords)

tdf["key_words"] = sentence_keywords

stc = "Archangel 12, the precursor to the sr71"
tok = nltk.word_tokenize(stc)
nltk.pos_tag(tok)
tdf

tdf.loc[~tdf["key_words"].astype(bool), "key_words"] = None
tdf
tdf.loc[tdf["da_label"].isin(filler_das), "key_words"] = None
tdf.head(100)

# temporarily set all entries without keywords to have the same keywords as previous:
current_keywords = ["NO KEYWORDS YET"]
for i, keywords in enumerate(tdf["key_words"]):
    if keywords:
        current_keywords = keywords
    else:
        tdf.at[i, "key_words"] = current_keywords

#%%


























    #embed using google sentence encoder
    embeddings = embed(sentences)
    valid_embeddings = embed(valid_sentences)

    #approach 1: use density based clustering algorithm to cluster sentences into topics
    sim_m = make_similarity_matrix(sentences, embeddings)

    #cluster_labels = DBSCAN(eps = 0.5, min_samples = 2, metric = "cosine").fit_predict(embeddings) #eps is sensitivity
    cluster_labels = DBSCAN(eps = 5, min_samples = 2).fit_predict(sim_m) #eps is sensitivity

    tdf["topic_label"] = cluster_labels + 1
    valid_sentence = ~tdf["da_label"].isin(filler_das)
    tdf["topic_label"] *= valid_sentence #if filler sentence has topic "0"
    plot_similarity(valid_sentences[:100], valid_embeddings[:100], 90) #visualised here

    (tdf["topic_label"] != 0).sum()/len(tdf)

    #%% Linkage approach

    #%% OPTICS APPROACH

    from sklearn.cluster import OPTICS

    clustering = OPTICS(min_samples=1, metric="cosine").fit(embeddings)

    (clustering.labels_ == 0).sum()

    clustering.labels_

    #%%
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize = (20, 20))
    plt.rcParams.update({'font.size':3})
    for text, topic, (x, y) in zip(tdf["utterance"], tdf["topic_label"], embeddings_2d):
        plt.scatter(x, y, c = "C" + str(topic))
        #plt.text(x, y + 0.1, text)


    plt.show()


    #%% k-means approach
    from sklearn.cluster import KMeans

    kmeans = KMeans(init="k-means++", n_clusters = 10, n_init=4)
    kmeans.fit(embeddings)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    k_means_2d = pca.transform(codebook)

    plt.figure(figsize = (20, 20))
    plt.rcParams.update({'font.size':3})

    for (x, y), text in zip(embeddings_2d, sentences):
        plt.scatter(x, y, c = "grey")
        plt.text(x, y + 0.1, text)

    for x,y in k_means_2d:
        plt.scatter(x, y, c = "red")

    plt.savefig("pca.pdf")
    plt.show()

    #%%

    reduced_data = PCA(n_components=2).fit_transform(embeddings)
    kmeans = KMeans(init="k-means++", n_clusters=30, n_init=4)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(20, 20))
    plt.clf()
    plt.imshow(Z, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired, aspect="auto", origin="lower")

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                color="w", zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(config.paths["figures"] + "kmeans.pdf")
    plt.show()

    #%%
    clusters = defaultdict(list)
    for cl, sentence in zip(tdf["topic_label"], tdf["utterance"]):
        if cl == 0: continue
        clusters[cl].append(sentence)

    pprint(list(clusters.values())) #these are sentence clusters

    n_topics = len(clusters)
    n_topics
    #plot_similarity(sentences, embeddings, 90) #visualised here

    #%%

    # approach 2: use network, every sentence is node, connections above cutoff
    # similarity, clusters in graph are clusters of topics
    # TODO: filter backchannels, agreements etc (DAs without content)
    # TODO: decay factor for time delay, penalise large time difference of sentences
    # TODO: adjust cutoff

    network = Network(cutoff = 0.4)

    for s, e in zip(valid_sentences, valid_embeddings):
        network.add_node(s, e)

    network.make_connections()
    network.get_clusters()


    pprint(network.clusters)
    len(network.clusters)
    #pprint(network.edges)

#%%

#stcs = test_sentences
#embeds = embed(stcs)

from sklearn.cluster import AgglomerativeClustering

#clustering = AgglomerativeClustering(n_clusters = None, distance_threshold = 0.2,
    #linkage="complete", affinity="precomputed").fit(sim_m[:100, :100])

from scipy.cluster import hierarchy
from scipy.spatial import distance


linkage = hierarchy.linkage(sim_m, method='average')

from scipy.cluster.hierarchy import fcluster
topic_labels = fcluster(linkage, sim_m.shape[0]//10, criterion="maxclust")
tdf["topic_label"] = topic_labels

#%%
for i in tdf["topic_label"].unique():
    if (tdf["topic_label"] == i).sum() > 5:
        print("TOPIC: ", i)
        print(tdf[tdf["topic_label"] == i]["utterance"].values)

#%%
sns.clustermap(sim_m[:100, :100], row_linkage=row_linkage, col_linkage=col_linkage)#, figsize=(20, 20))
plt.show()
#%%

from minisom import MiniSom

map_dim = 25

es = valid_embeddings[:100]
stcs = valid_sentences[:100]
som = MiniSom(map_dim, map_dim, 512, sigma=1.0, random_seed=1)
som.train_batch(es, num_iteration=len(es)*500, verbose=True)

#%%

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in es]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, (map_dim, map_dim))

#%%
plt.figure(figsize=(10, 10))
texts = []

#for t, vec in zip(stcs, es):
#    winning_position = som.winner(vec)
#    texts.append(plt.text(
#        winning_position[0],
#        winning_position[1],
#        t))

es = pd.DataFrame(es)
for c in np.unique(cluster_index):
    plt.scatter(es[cluster_index == c][0],
                es[cluster_index == c][1], label = "cluster: " + str(c), alpha=.7)

plt.rcParams.update({'font.size':3})
#plt.xlim([0, map_dim])
#plt.ylim([0, map_dim])
plt.savefig(config.paths["figures"] + "som.pdf")
plt.plot()
