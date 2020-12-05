import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from datetime import datetime

from tqdm import tqdm

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

def timestamp_to_datetime(timestamp):
    return datetime.strptime(timestamp, "%H:%M:%S")


def get_fractional_time(current_timestamp, biggest_time):
    try:
        current_seconds = (biggest_time - timestamp_to_datetime(current_timestamp)).total_seconds()
        total_seconds = (biggest_time - timestamp_to_datetime("00:00:00")).total_seconds()
        return 1 - current_seconds/total_seconds
    except:
        return current_timestamp

def enhance_transcript_df(df):
    #calc relative time
    biggest_time = timestamp_to_datetime(df.loc[df.shape[0] - 1, "timestamp"])
    df["relative_time"] = (
        df["timestamp"].apply(get_fractional_time, args = (biggest_time, )))

    #explicitly find speaker change positions
    speaker_change = np.zeros(df.shape[0], dtype=np.int32)
    speakers = df["speaker"]
    for i, speaker in speakers[:-1].iteritems():
        if speaker != speakers[i + 1]:
            speaker_change[i + 1] = 1
    df["speaker_change"] = speaker_change

    # length of utterances in seconds
    timestamps = df["timestamp"]
    lengths = np.zeros(df.shape[0])
    for i, timestamp in timestamps[:-1].iteritems():
        next_timestamp = timestamps[i + 1]
        lengths[i] = (timestamp_to_datetime(next_timestamp) - timestamp_to_datetime(timestamp)).total_seconds()
    df["utterance_length"] = lengths

    return df

max_nr_utterances = config.data["max_nr_utterances"]
corpus = config.corpus["corpus"]
detail_level = config.corpus["detail_level"]

id2tag = get_id2tag(corpus, detail_level)
tag2id = {t: id for id, t in id2tag.items()}

tag2full = get_tag2full_label(corpus, detail_level)
full2tag = {full : t for t, full in tag2full.items()}

n_tags = len(id2tag)

transcripts = load_all_transcripts()
transcript_dfs = [make_annotated_transcript(t) for t in tqdm(transcripts)]
for transcript_df in transcript_dfs:
    enhance_transcript_df(transcript_df)
merged_transcript_dfs = pd.concat(transcript_dfs)

#%%
tag_sequences = [t_df["da_label"] for t_df in transcript_dfs]

unique_tags = np.unique(np.asarray(list(tag2full.values())))

tag_counter = {t : 0 for t in unique_tags}
for tag_sequence in tag_sequences:
    for t in tag_sequence:
        tag_counter[t] += 1

tag_counter_tuples = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)

unique_tags = [t[0] for t in tag_counter_tuples]

tag2id = {t: i for i, t in enumerate(unique_tags)}
id2tag = {i: t for i, t in enumerate(unique_tags)}


#%%
#tag_counter_tuples
##inspect low frequency DAs
#indices = []
#for i, t in enumerate(merged_annotated_transcripts):
#    if t[1] == "Wh-Question":
#        print(t)
#        indices.append(i)
#t = 5
#merged_annotated_transcripts[indices[t] - 3 : indices[t] + 3]
#%%

n = 10
adj_matrix = make_adjacency_matrix(tag_sequences, n, normalise="prev")
for i in range(adj_matrix.shape[1]):
    mean = adj_matrix[:, i].mean()
    adj_matrix[:, i] = adj_matrix[:, i] - mean
top_n_tags = unique_tags[:n]
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(adj_matrix, xticklabels=top_n_tags, yticklabels=top_n_tags, annot=True)
plt.show()
#%%
n = 10
adj_matrix = make_adjacency_matrix(tag_sequences, n, normalise="prev")
for i in range(adj_matrix.shape[1]):
    mean = adj_matrix[:, i].mean()
    adj_matrix[:, i] = adj_matrix[:, i] - mean
top_n_tags = unique_tags[:n]
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(adj_matrix, xticklabels=top_n_tags, yticklabels=top_n_tags, annot=True)
plt.show()
#%%

transcript_df = transcript_dfs[3]
transcript_df
speaker_change_idx = transcript_df[transcript_df["speaker_change"] == 1].index
speaker_change_prev_idx = speaker_change_idx - 1

speaker_change_da = transcript_df.iloc[speaker_change_idx]["da_label"].values
speaker_change_prev_da = transcript_df.iloc[speaker_change_prev_idx]["da_label"].values

top_n_tags = unique_tags[:n]
for da, prev_da in zip(speaker_change_da, speaker_change_prev_da):
    row = tag2id[da]
    col = tag2id[prev_da]

    next_tag = tag_sequence[i + 1]
    if da not in top_n_tags or prev_da not in top_n_tags : continue
    adj_matrix[row, col] += 1

normalise = "prev"
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


plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(adj_matrix, xticklabels=top_n_tags, yticklabels=top_n_tags, annot=True)
plt.show()
#%%


das = ["Yes-No-Question", "Wh-Question"]
das = ["Yes-No-Question", "Yes answers"]
#das = ["Conventional-opening", "Conventional-closing"]
#das = ["Statement-opinion", "Statement-non-opinion", "Agree/Accept"]
#das = ["Acknowledge (Backchannel)","Statement-opinion", "Statement-non-opinion"]
#da = "Wh-Question"
#da = "Conventional-closing"
#da = "Conventional-opening"
#%% CDF
plt.figure(figsize=(10,8))
for da in das:
    total_count = transcript_df[transcript_df["da_label"] == da]["da_label"].count()
    da_df = transcript_df[transcript_df["da_label"] == da]

    t = transcript_df["relative_time"]

    da_df

    counter = 0
    y = np.zeros(transcript_df.shape[0])

    prev_i = 0
    for i, entry in da_df.iterrows():
        counter += 1
        cumulative_probability = counter/total_count
        y[prev_i:i+1] = cumulative_probability
        prev_i = i + 1
    y[prev_i:] = 1

    plt.plot(t, y, "", label = da)

plt.legend()
plt.show()
#%% PDF

transcript_df = transcript_dfs[3]
das = ["Yes-No-Question", "Yes answers"]
#das = ["Yes-No-Question", "Wh-Question", "Statement-opinion", "Statement-non-opinion"]
#das = ["Conventional-opening", "Conventional-closing"]
#das = ["Statement-opinion", "Statement-non-opinion", "Agree/Accept"]
#das = ["Statement-opinion", "Statement-non-opinion"]
#da = "Wh-Question"
import scipy as sp
plt.figure(figsize=(10,8))
for da in das:
    total_count = transcript_df[transcript_df["da_label"] == da]["da_label"].count()
    da_df = transcript_df[transcript_df["da_label"] == da]

    t = transcript_df["relative_time"]

    da_df

    counter = 0
    y = np.zeros(transcript_df.shape[0])

    for i, entry in da_df.iterrows():
        y[i] = 1

    y1 = sp.ndimage.gaussian_filter(y, 50)
    y1 = y1/max(y1)
    plt.plot(t, y1, "", label = da)

plt.legend()
plt.show()



#%% speaker changes over time

y = sp.ndimage.gaussian_filter(transcript_df["speaker_change"].values/10, 30)
#y= transcript_df["speaker_change"].values
t = transcript_df["relative_time"].values
plt.plot(t, y)

#%% speaker dominance
transcript_df = transcript_dfs[2]
speakers = transcript_df["speaker"].unique()

total_utterances = transcript_df.shape[0]
counts = []
for speaker in speakers:
    counts.append((transcript_df["speaker"] == speaker).sum()/total_utterances)

plt.bar(speakers, counts)

#%% speaker dominance by time

transcript_df = transcript_dfs[3]
speakers = transcript_df["speaker"].unique()

total_utterances = transcript_df.shape[0]
counts = []
for speaker in speakers:
    counts.append((transcript_df["speaker"] == speaker).sum()/total_utterances)

plt.bar(speakers, counts)
#%% #utterance length over time

plt.figure(figsize=(10,8))
for transcript_df in transcript_dfs[1:]:
    full_length_df = transcript_df[transcript_df["utterance_length"] != 0]
    y = sp.ndimage.gaussian_filter(full_length_df["utterance_length"].values, 15)

    plt.plot(full_length_df["relative_time"].values, y, label=" ".join(transcript_df["speaker"].unique()))

plt.legend()
plt.show()
