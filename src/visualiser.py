import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import pandas as pd
from topics import TopicExtractor
import config
from matplotlib import cm
import matplotlib as mpl


class Visualiser():

    def __init__(self, max_nodes):

        # Sentence to node
        self.s2n = {}

        #stores all nodes
        self.nodes = []

        # position data
        # self.y = 0
        # self.delta_y = 1 #depth gained per timestep
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.delta_y = 2 # height difference between topics

        self.width = 10

        self.node_radius = 0.3

        self.fig = plt.figure(0, figsize=(150, 2))
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
        self.connect_nodes(n0, n1, color=color)

        fontsize = max(min(8, (x1-x0)/len(s) * 8), 4)
        plt.annotate(s, ((x1+x0)/2, y + self.delta_y/3), color=color,
                     fontsize=fontsize, ha="center")

    def set_axes(self):
        self.ax.set_xlim(self.min_x - 5, self.max_x + 5)
        self.ax.set_ylim(self.min_y - 1, self.max_y + 3)

    def show_fig(self):
        self.set_axes()
        plt.show()

    def save_fig(self, name):
        fpath = config.paths["figures"] + name + "_topics.pdf"
        self.set_axes()
        plt.savefig(fpath)

    def connect_nodes(self, n0, nf, color=None):
        x0, y0 = n0.center
        xf, yf = nf.center

        self.nr_objects[int(x0): int(xf)] += 1

        for i, lfy in enumerate(self.lowest_free_y[int(x0): int(xf)], int(x0)):
            if lfy >= y0/self.delta_y:
                self.lowest_free_y[i] += self.delta_y

        con = ConnectionPatch(n0.center, nf.center, "data", "data", zorder=0)
        con.set_linewidth(4)
        if color:
            con.set_edgecolor(color)
        self.ax.add_artist(con)

# def invert_dict_of_lists(d):
#     new_dict = defaultdict(list)
#     for key, l in d.items():
#         for listitem in l:
#             new_dict[listitem].append(key)
#
#     return new_dict


def get_colormap(embeddings_1d):
    norm = mpl.colors.Normalize(embeddings_1d.min(), embeddings_1d.max())
    cmap = cm.tab20c
    return cm.ScalarMappable(norm=norm, cmap=cmap)


te = TopicExtractor()
# transcript_name = "joe_rogan_elon_musk"
transcript_name = "sam_harris_nicholas_christakis"
tdf = pd.read_pickle("../processed_transcripts/" + transcript_name + ".pkl")
pd.options.display.max_rows = None
embeds1d = np.array(te.fit_n_d_embeddings(tdf, 1))
scalar_to_color = get_colormap(embeds1d)
# %%
tdf, topic_ranges = te.add_topics(tdf, return_topic_ranges=True)
vis = Visualiser(len(tdf))
min_topic_length = 5

for tr, topics in topic_ranges.items():
    if tr[1] - tr[0] < min_topic_length:
        continue

    for topic in topics:
        joined = ", ".join(list(topic))

        try:
            embedding_1d = te.get_n_d_embedding(topic, 1)
            if embedding_1d is False:
                color = 'grey'
            color = scalar_to_color.to_rgba(embedding_1d)
        except:
            color = 'grey'
        vis.add_section(tr[0], tr[1], joined, color)

vis.save_fig(transcript_name)
vis.show_fig()
