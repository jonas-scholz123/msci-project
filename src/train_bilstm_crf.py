import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.preprocessing.text import Tokenizer
import tensorflow as tf

from utils import get_embedding_matrix, get_tokenizer, make_model_readable_data,\
    chunk, load_corpus_data
from mappings import get_id2tag
from bilstm_crf import get_bilstm_crf_model

import config

#load config data
max_nr_utterances = config.data["max_nr_utterances"]
max_nr_words = config.data["max_nr_words"]
corpus = config.corpus["corpus"]
detail_level = config.corpus["detail_level"]
batch_size = config.model["batch_size"]

conversations, labels = load_corpus_data(corpus, detail_level)

conversations = chunk(conversations, max_nr_utterances)
labels = chunk(labels, max_nr_utterances)

n_tags = len(get_id2tag(corpus, detail_level=detail_level))

tokenizer = get_tokenizer(rebuild_from_all_texts=False) #TODO set to false for final model
word2id = tokenizer.word_index

X,y = make_model_readable_data(conversations, labels, tokenizer,
        max_nr_utterances, max_nr_words)

# import pretrained GloVe embeddings
embedding_matrix = get_embedding_matrix("../data/embeddings/glove.840B.300d.txt",
    word2id, force_rebuild=False) #set force rebuild to False when not changing total vocabulary


model = get_bilstm_crf_model(embedding_matrix, max_nr_utterances, max_nr_words, n_tags)

data_name = corpus + "_detail_" + str(detail_level)
checkpoint_path = "../trained_model/bilstm_crf/ckpt_" + data_name + ".hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True)

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

model.fit(X, y, batch_size=batch_size, epochs=3, validation_split=0.1,
    callbacks=[model_checkpoint_callback])
