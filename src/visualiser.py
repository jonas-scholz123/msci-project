import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.patches import Ellipse, Circle, Arrow, ConnectionPatch
from classifier import Classifier
import numpy.random as rnd


fpath = "../data/podcasts/joe_rogan_elon_musk_may_2020.txt"

classifier = Classifier()

classifier.read_file(fpath)

classifier.classify_emotion()

classifier.classify_disagreement()

classifier.iterate_topic_words()

classifier.classify_topical_similarity()

classifier.make_branches()

classifier.make_dissimilarity_matrix()

classifier.mds_topic_similarity()

import numpy as np

class Visualiser():

    def __init__(self):

        #Sentence to node
        self.s2n = {}

        #stores all nodes
        self.nodes = []

        #position data
        self.y = 0
        self.delta_y = 1 #depth gained per timestep
        self.min_x = 0
        self.max_x = 0

        self.width = 10

        self.node_radius = 0.3

        self.fig = plt.figure(0, figsize = (10, 10))
        self.ax = self.fig.add_subplot(111, aspect='equal')

    def add_node(self, x, r = 0.3, normalised = True):
        if normalised:
            x = self.width * x
        self.set_min_max(x)
        node = Circle((x, self.y), self.node_radius)
        self.ax.add_artist(node)
        node.set_facecolor("white")
        node.set_edgecolor("black")
        self.nodes.append(node)

        self.y -= self.delta_y

    def set_min_max(self, x):
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)

    def show_fig(self):
        self.ax.set_xlim(self.min_x - 5, self.max_x + 5)
        self.ax.set_ylim(self.y - 5, 2)
        #self.fig.show()
        plt.show()

    def connect_nodes(self, n0, nf):
        x0, y0 = n0.center
        xf, yf = nf.center

        con = ConnectionPatch(n0.center, nf.center, "data", "data", zorder = 0)
        self.ax.add_artist(con)

#%%
vis = Visualiser()

#for i in range(25):
#    x = rnd.randint(-5, 5)
#    vis.add_node(x)
#    vis.connect_nodes(vis.nodes[i - 1], vis.nodes[i])
#vis.show_fig()

vis = Visualiser()

vis.add_node(0)
vis.width = 2* len(classifier.branches) * classifier.topic_width

for branch, x in zip(classifier.branches, classifier.x_branches):
    for sentence in branch.values:
        vis.add_node(x)
        vis.connect_nodes(vis.nodes[-1], vis.nodes[-2])
vis.show_fig()
