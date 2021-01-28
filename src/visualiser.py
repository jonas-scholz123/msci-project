import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.patches import Circle, ConnectionPatch
from utils import load_all_processed_transcripts
import numpy as np
import pandas as pd
from topics import TopicExtractor
import config
from matplotlib import cm
import matplotlib as mpl
from tqdm import tqdm
from collections import defaultdict


class Visualiser():

    def __init__(self, max_nodes):

        # Sentence to node
        self.s2n = {}

        # stores all nodes
        self.nodes = []

        # position data
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.delta_y = 3  # height difference between topics

        self.width = 10

        self.node_radius = 0.3

        self.fig = plt.figure(0, figsize=(max_nodes//10, 2))
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
        y = self.lowest_free_y[x0:x1].max()

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
            self.lowest_free_y[i] = y0 + self.delta_y

        con = ConnectionPatch(n0.center, nf.center, "data", "data", zorder=0)
        con.set_linewidth(4)
        if color:
            con.set_edgecolor(color)
        self.ax.add_artist(con)


def get_topic_ranges(tdf):
    trs = defaultdict(list)

    for i, topics in tdf["topics"].iteritems():
        for topic in topics:
            existing_range = trs.get(frozenset(topic))
            if (existing_range is not None and i >= existing_range[-1][0]
                    and i <= existing_range[-1][1]):
                continue
            for j, next_topics in tdf.loc[i + 1:, "topics"].iteritems():
                if topic in next_topics:
                    continue
                trs[frozenset(topic)].append((i, j - 1))
                break
    return trs


def get_colormap(embeddings_1d):
    norm = mpl.colors.Normalize(embeddings_1d.min(), embeddings_1d.max())
    cmap = cm.Dark2
    return cm.ScalarMappable(norm=norm, cmap=cmap)


if __name__ == "__main__":
    te = TopicExtractor()

    tdfs, tnames = load_all_processed_transcripts(return_fnames=True)
    for tdf, transcript_name in tqdm(zip(tdfs, tnames)):

        pd.options.display.max_rows = None
        embeds1d = np.array(te.fit_n_d_embeddings(tdf, 1))
        scalar_to_color = get_colormap(embeds1d)
        topic_ranges = get_topic_ranges(tdf)
        vis = Visualiser(len(tdf))
        min_topic_length = 5
        # sort topics by tr, so that bigger ones placed at the bottom

        tr_items = [(topic, tr) for topic, trs in topic_ranges.items()
                    for tr in trs]

        for topic, tr in tr_items:
            if tr[1] - tr[0] < min_topic_length:
                continue
            joined = ", ".join(list(topic))
            embedding_1d = te.get_n_d_embedding(topic, 1)
            if embedding_1d is False:
                color = 'lightgrey'
            color = scalar_to_color.to_rgba(embedding_1d)
            vis.add_section(tr[0], tr[1], joined, color)

        vis.save_fig(transcript_name)
        vis.show_fig()
