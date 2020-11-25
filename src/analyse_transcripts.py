import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from utils import load_one_transcript, load_all_transcripts
from predictDA import make_annotated_transcript

import config

from mappings import get_id2tag, get_tag2full_label

def make_adjacency_matrix(tag_sequences, n, normalise = "next"):

    adj_matrix = np.zeros((n, n))

    for tag_sequence in tag_sequences:

        top_n_tags = unique_tags[:n]
        for i, tag in enumerate(tag_sequence[:-1]):
            row = tag2id[tag]

            next_tag = tag_sequence[i + 1]
            if tag not in top_n_tags or next_tag not in top_n_tags : continue
            col = tag2id[next_tag]
            adj_matrix[row, col] += 1

    #normalise:
    if normalise in ["next", "both"]:
        #gives prob of following da's given that current da is col
        for i, row in enumerate(adj_matrix):
            if row.sum() == 0: continue
            adj_matrix[i] = np.round(row/row.sum(), 2)

    if normalise in ["prev", "both"]:
        #gives prob of previous da's given that current da is col
        for i, col in enumerate(adj_matrix.T):
            if col.sum() == 0: continue
            adj_matrix.T[i] = np.round(col/col.sum(), 2)

        adj_matrix = adj_matrix.T

    return adj_matrix

max_nr_utterances = config.data["max_nr_utterances"]
corpus = config.corpus["corpus"]
detail_level = config.corpus["detail_level"]

id2tag = get_id2tag(corpus, detail_level)
tag2id = {t: id for id, t in id2tag.items()}

tag2full = get_tag2full_label(corpus, detail_level)
full2tag = {full : t for t, full in tag2full.items()}

n_tags = len(id2tag)

transcripts = load_all_transcripts()
annotated_transcripts = [make_annotated_transcript(t) for t in transcripts]
tag_sequences = [[u[1] for u in a_t] for a_t in annotated_transcripts]

unique_tags = np.unique(np.asarray(list(tag2full.values())))

tag_counter = {t : 0 for t in unique_tags}
for tag_sequence in tag_sequences:
    for t in tag_sequence:
        tag_counter[t] += 1

tag_counter_tuples = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)

unique_tags = [t[0] for t in tag_counter_tuples]

tag2id = {t: i for i, t in enumerate(unique_tags)}
id2tag = {i: t for i, t in enumerate(unique_tags)}

adj_matrix = make_adjacency_matrix(tag_sequences, n, normalise="prev")
top_n_tags = unique_tags[:n]
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(adj_matrix, xticklabels=top_n_tags, yticklabels=top_n_tags, annot=True)
plt.show()
