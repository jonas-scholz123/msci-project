"""
    This file is incomplete and was going to contain the analysis of the topic
    information we extract.
"""


import pandas as pd
import os
import config
from utils import load_pretrained_conceptnet
import numpy as np
from scipy.spatial import ConvexHull


class TopicAnalyser:
    def __init__(self, tdf, embeddings):
        self.tdf = tdf
        self.embeddings = embeddings

    def calculate_volume(self):
        """
        Doesnt work, exponential complexity
        """
        key_words = self.tdf[~self.tdf["key_words"].isnull()]["key_words"]
        all_keywords = set().union(*[k for k in key_words if k])
        all_kw_embeds = np.asarray(
            [self.embeddings[kw] for kw in all_keywords if kw in self.embeddings]
        )

        all_kw_embeds = all_kw_embeds[:401]

        hull = ConvexHull(all_kw_embeds)

        hull.volume
        return volume


if __name__ == "__main__":
    tdf_dir = config.paths["tdfs"]

    pkl_files = []
    for root, dirs, files in os.walk(tdf_dir):
        for name in files:
            if name.endswith(".pkl"):
                pkl_files.append(os.path.join(root, name))

    tdf = pd.read_pickle(pkl_files[0])
    embeds = load_pretrained_conceptnet()
    ta = TopicAnalyser(tdf, embeds)
