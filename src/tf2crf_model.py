import numpy as np
import pickle
import re
import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from keras import Sequential
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from keras_contrib.metrics import crf_viterbi_accuracy, crf_marginal_accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tf2crf import CRF

from utils import get_embedding_matrix, pad_nested_sequences, \
    split_into_chunks, load_mrda_data, get_id2tag, get_tokenizer, \
    make_model_readable_data

dropout_rate = 0.5
EMBEDDING_DIM = 300
max_nr_utterances = 100
max_nr_words = 200

#get mrda data TODO: get other datasets in same format
conversations, labels = load_mrda_data(chunk_size = max_nr_utterances)
all_utterances = sum(conversations, [])

#get id2tag map and inverse
id2tag = get_id2tag()
tag2id = {t : id for id, t in id2tag.items()}
n_tags = len(tag2id.keys())

tokenizer = get_tokenizer(all_utterances)
word2id = tokenizer.word_index

X,y = make_model_readable_data(conversations, labels, tokenizer, tag2id,
        max_nr_utterances, max_nr_words)

# import pretrained GloVe embeddings
embedding_matrix = get_embedding_matrix("../data/embeddings/glove.840B.300d.txt", word2id)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
#%%

checkpoint_path = "../trained_model/checkpoint_bilstm_crf_dropout.hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True)

embedding_layer = Embedding(len(word2id) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_nr_words, trainable=False)

utterance_encoder = Sequential()
utterance_encoder.add(embedding_layer)
utterance_encoder.add(Bidirectional(LSTM(300, return_sequences=True)))
utterance_encoder.add(AveragePooling1D(max_nr_words))
utterance_encoder.add(Dropout(dropout_rate))
utterance_encoder.add(Flatten())
utterance_encoder.summary()

crf = CRF(dtype='float32')

x_input = Input(shape = (max_nr_utterances, max_nr_words))
h = TimeDistributed(utterance_encoder)(x_input)
h = Bidirectional(LSTM(300, return_sequences=True))(h)
h = Dropout(dropout_rate)(h)
h = Dense(n_tags, activation=None)(h)
crf_output = crf(h)
model = Model(x_input, crf_output)
#h = Dense(n_tags, activation="softmax")(h)
#model = Model(x_input, h)

model.summary()
model.compile("adam", loss=crf.loss, metrics=[crf.accuracy])

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

model.fit(X, y, batch_size=5, epochs=10, validation_split=0.2,
    callbacks=[model_checkpoint_callback])
