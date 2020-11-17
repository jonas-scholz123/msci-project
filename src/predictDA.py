import numpy as np
import pickle
import re
import os

import tensorflow as tf

from utils import get_embedding_matrix, pad_nested_sequences, \
    split_into_chunks, load_mrda_data, get_tokenizer, \
    make_model_readable_data, make_model_readable_X, load_all_transcripts

from tf2crf_model import get_tf2crf_model

from mappings import get_id2tag

max_nr_utterances = 100
max_nr_words = 200

transcripts = load_all_transcripts(chunked=True)
#get id2tag map and inverse
id2tag = get_id2tag('mrda', detail_level = 0)
tag2id = {t : id for id, t in id2tag.items()}
n_tags = len(tag2id.keys())

tokenizer = get_tokenizer(rebuild_from_all_texts=False) #TODO set to false for final model
word2id = tokenizer.word_index

X = make_model_readable_X(transcripts, tokenizer,max_nr_utterances, max_nr_words)

# import pretrained GloVe embeddings
embedding_matrix = get_embedding_matrix("../data/embeddings/glove.840B.300d.txt",
    word2id, force_rebuild=False) #set force rebuild to False when not changing total vocabulary

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

model = get_tf2crf_model(embedding_matrix, max_nr_utterances, max_nr_words, n_tags)

checkpoint_path = "../trained_model/tf2crf/ckpt_tf2crf.hdf5"
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

y_hat = model.predict(X, batch_size=1)

y_hat.shape

y_hat = [[id2tag[id] for id in predict_batch] for predict_batch in y_hat]

u_joined_y_hat = []
for t, y_hat_batch in zip(transcripts, y_hat):
    u_joined_y_hat.append(tuple(zip(t, y_hat_batch)))

u_joined_y_hat
