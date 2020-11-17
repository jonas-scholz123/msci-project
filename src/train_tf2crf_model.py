import numpy as np
import pickle
import re
import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from keras import Sequential
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, AveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop
from keras_contrib.metrics import crf_viterbi_accuracy, crf_marginal_accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tf2crf import CRF

from utils import get_embedding_matrix, pad_nested_sequences, \
    split_into_chunks, load_mrda_data, get_tokenizer, \
    make_model_readable_data, chunk, load_switchboard_data

from mappings import get_id2tag

from reformat_training_data import read_mrda_training_data

from tf2crf_model import get_tf2crf_model

dropout_rate = 0.5
EMBEDDING_DIM = 300
max_nr_utterances = 100
max_nr_words = 200

corpus = 'swda' # 'mrda' or 'swda'
detail_level = 0

data_name = corpus + "_detail_" + str(detail_level)

#get mrda data TODO: get other datasets in same format
#conversations, labels = load_mrda_data(chunk_size = max_nr_utterances)
conversations, labels = read_mrda_training_data(detail_level=detail_level)
conversations, labels = load_switchboard_data()
all_utterances = sum(conversations, [])

conversations = chunk(conversations, max_nr_utterances)
labels = chunk(labels, max_nr_utterances)

#get id2tag map and inverse
id2tag = get_id2tag(corpus, detail_level=detail_level)
tag2id = {t : id for id, t in id2tag.items()}
n_tags = len(tag2id.keys())

tokenizer = get_tokenizer(rebuild_from_all_texts=True) #TODO set to false for final model
word2id = tokenizer.word_index

X,y = make_model_readable_data(conversations, labels, tokenizer,
        max_nr_utterances, max_nr_words)


y2 = to_categorical(y)
y2.shape
X.shape
X3 = X.reshape(-1, X.shape[-1])
y3 = y2.reshape(-1, y2.shape[-1])
# import pretrained GloVe embeddings
embedding_matrix = get_embedding_matrix("../data/embeddings/glove.840B.300d.txt",
    word2id, force_rebuild=False) #set force rebuild to False when not changing total vocabulary

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

#%%
model = get_tf2crf_model(embedding_matrix, max_nr_utterances, max_nr_words, n_tags)

checkpoint_path = "../trained_model/tf2crf/ckpt_" + data_name + ".hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True)

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

model.fit(X, y, batch_size=5, epochs=50, validation_split=0.2,
    callbacks=[model_checkpoint_callback])

#%%
#hidden_layer = 128
#
#model = Sequential()
#model.add(Embedding(embedding_matrix.shape[0], EMBEDDING_DIM, input_length=max_nr_words, weights=[embedding_matrix], mask_zero=False))
#model.add(LSTM(hidden_layer, dropout=0.3, return_sequences=True, kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform'))
#model.add(TimeDistributed(Dense(hidden_layer, input_shape=(max_nr_words, hidden_layer))))
#model.add(GlobalMaxPooling1D())
#model.add(Dense(n_tags, activation='softmax'))
#
#optimizer = RMSprop(lr=0.001, decay=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
#model.fit(X3, y3, batch_size=500, epochs=50, validation_split=0.2)
