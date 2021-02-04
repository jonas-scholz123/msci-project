import config
from sklearn.model_selection import train_test_split
from bilstm_crf import get_bilstm_crf_model
from mappings import get_id2tag
from utils import (
    get_embedding_matrix,
    get_tokenizer,
    make_model_readable_data,
    chunk,
    load_corpus_data,
)
import tensorflow as tf
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# load config data
max_nr_utterances = config.data["max_nr_utterances"]
max_nr_words = config.data["max_nr_words"]
corpus = config.corpus["corpus"]
detail_level = config.corpus["detail_level"]
batch_size = config.model["batch_size"]
test_fraction = config.model["test_fraction"]
validation_fraction = config.model["validation_fraction"]

conversations, labels = load_corpus_data(corpus, detail_level)

conversations = chunk(conversations, max_nr_utterances)
labels = chunk(labels, max_nr_utterances)

n_tags = len(get_id2tag(corpus, detail_level=detail_level))

# TODO set to false for final model
tokenizer = get_tokenizer(rebuild_from_all_words=False)
word2id = tokenizer.word_index

X, y = make_model_readable_data(
    conversations, labels, tokenizer, max_nr_utterances, max_nr_words
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# import pretrained GloVe embeddings
embedding_matrix = get_embedding_matrix(
    "../data/embeddings/glove.840B.300d.txt", word2id, force_rebuild=False
)  # set force rebuild to False when not changing total vocabulary


model = get_bilstm_crf_model(embedding_matrix, max_nr_utterances, max_nr_words, n_tags)

data_name = corpus + "_detail_" + str(detail_level)
checkpoint_path = "../trained_model/bilstm_crf/ckpt_" + data_name + ".hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True
)

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=3,
    validation_data=(X_test, y_test),
    callbacks=[model_checkpoint_callback],
)
