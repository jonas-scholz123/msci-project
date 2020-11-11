import pandas as pd
import os
import tensorflow.compat.v1 as tf
import re
import numpy as np
import pickle

from kumar import DAModel, pad_sequences

def read_transcript(fpath):
    #open file
    with open(fpath, "r") as f:
        podcast = f.readlines()

    #organise content
    title_rows = podcast[::3]
    speakers = [row.split(":")[0] for row in title_rows]
    times = [":".join(re.findall("[0-9]+", row)) for row in title_rows]
    text = podcast[1::3]
    text = [t[0:-1] for t in text]

    #store content in dataframe
    podcast_df = pd.DataFrame([speakers, times, text]).transpose()
    podcast_df.columns = ["speaker", "time", "utterances"]

    return podcast_df

podcast_df = read_transcript("../transcripts/joe_rogan_kanye_west.txt")

podcast_df

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

data = [[[1, 5, 2, 188, 216], [29, 1, 4, 16]], [[1, 9, 4, 188, 116], [229, 1, 4, 116]],
    [[1, 9, 4, 188, 116], [229, 1, 4, 116]]]
labels = [[0, 0], [0, 0], [0, 0]]
clip = 2

with tf.Session(config=config) as sess:
    model = DAModel()

    saver = tf.train.Saver()
    if os.path.exists("../trained_model/kumar/model.index"):
        print("MODEL FOUND, LOADING CHECKPOINT")
        saver.restore(sess, "../trained_model/kumar/model")


    _, dialogue_lengths = pad_sequences(data, 0)
    X, utterance_lengths = pad_sequences(data, 0, nlevels=2)
    y, _ = pad_sequences(labels, 0)

    y_pred = sess.run(model.logits,
        feed_dict = {
          model.word_ids: X,
          model.utterance_lengths: utterance_lengths,
          model.dialogue_lengths: dialogue_lengths,
          model.labels: y,
          model.clip: clip
        }
    )


#%% translate

with open("../helper_files/mrda_id_to_tag.pkl", "rb") as f:
    id2tag = pickle.load(f)

tags = [[id2tag[np.argmax(y_pred[j, i])] for i in range(len(data[j]))] for j in range(len(data))]

tags
