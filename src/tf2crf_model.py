import numpy as np
import pickle
import re
import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from keras import Sequential
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
#from keras_contrib.layers import CRF
#from keras_contrib.losses import crf_loss, crf_nll
from keras_contrib.metrics import crf_viterbi_accuracy, crf_marginal_accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

#from tensorflow_addons.layers.crf import CRF
#from tensorflow_addons.text.crf import crf_log_likelihood

from tf2crf import CRF

from utils import load_pretrained_matrix, pad_nested_sequences, split_into_chunks

dropout_rate = 0.2
EMBEDDING_DIM = 300
max_nr_utterances = 100
max_nr_words = 200

#open tag2id mapping for labels and create inverse
with open('../helper_files/mrda_id_to_tag.pkl', 'rb') as f:
    id2tag = pickle.load(f)
tag2id = {t : id for id, t in id2tag.items()}
n_tags = len(tag2id.keys())

with open('../data/clean/mrda_utterances.tsv', 'r') as f:
    lines = f.readlines()

conversations = [[u for u in c.split('\t')[1:]] for c in lines]
chunked_conversations = [split_into_chunks(c, max_nr_utterances) for c in conversations]
chunked_conversations = sum(chunked_conversations, [])
conversations = chunked_conversations #TODO test

all_utterances = [line.split("\t")[1:] for line in lines]
all_utterances = sum(all_utterances, [])

with open('../data/clean/mrda_labels.tsv', 'r') as f:
    lines = f.readlines()

labels = [line.split("\t")[1:] for line in lines]

#fix parsing of '\n' tag
for l in labels[:-1]:
    l[-1] = l[-1][:-1]


chunked_labels = [split_into_chunks(l, max_nr_utterances) for l in labels]
chunked_labels = sum(chunked_labels, [])
labels = chunked_labels #TODO test

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_utterances)
word2id = tokenizer.word_index

conversation_sequences = [tokenizer.texts_to_sequences(c) for c in conversations]
#max_nr_utterances = max([len(s) for s in conversation_sequences])
#max_nr_words = max([len(u) for u in all_utterances])

print("maximum number of utterances per conversation: ", max_nr_utterances) #TODO must be at least podcast length!!
print("maximum number of words per utterance: ", max_nr_words)

X = pad_nested_sequences(conversation_sequences, max_nr_utterances, max_nr_words)
y = [to_categorical(l, num_classes=n_tags) for l in labels]
y = pad_sequences(y, max_nr_utterances, padding= "post")

# import pretrained GloVe embeddings
embedding_matrix = load_pretrained_matrix("../dialogue-understanding/glove-end-to-end/glove/glove.840B.300d.txt", word2id)

#%%

checkpoint_path = "../trained_model/checkpoint_bilstm_crf_dropout.hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True)

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
with tf.device('/CPU:0'):
    embedding_layer = Embedding(len(word2id) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_nr_words, trainable=False)

    utterance_encoder = Sequential()
    utterance_encoder.add(embedding_layer)
    utterance_encoder.add(Dropout(dropout_rate))
    utterance_encoder.add(Bidirectional(LSTM(300, return_sequences=True)))
    utterance_encoder.add(Dropout(dropout_rate))
    utterance_encoder.add(AveragePooling1D(max_nr_words))
    utterance_encoder.add(Dropout(dropout_rate))
    utterance_encoder.add(Flatten())
    utterance_encoder.summary()


    #model = Sequential()
    #model.add(TimeDistributed(utterance_encoder, input_shape = (max_nr_utterances, max_nr_words)))
    #model.add(Bidirectional(LSTM(256, return_sequences = True)))
    #model.add(Dense(n_tags))
    #model.add(CRF(n_tags))
    #model.add(crf)
    #model.summary()


    crf = CRF(dtype='float32')

    x_input = Input(shape = (max_nr_utterances, max_nr_words))
    h = TimeDistributed(utterance_encoder)(x_input)
    h = Dropout(dropout_rate)(h)
    h = Bidirectional(LSTM(300, return_sequences=True))(h)
    h = Dropout(dropout_rate)(h)
    h = Dense(n_tags, activation=None)(h)
    h = Dropout(dropout_rate)(h)
    crf_output = crf(h)
    model = Model(x_input, crf_output)
    #h = Dense(n_tags, activation="softmax")(h)
    #crf_output = crf(h)
    #model = Model(x_input, h)


    #model.compile("adam", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()
    model.compile("adam", loss=crf.loss, metrics=[crf.accuracy])

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    model.fit(X, y, batch_size=10, epochs=10, validation_split=0.2,
        callbacks=[model_checkpoint_callback])
