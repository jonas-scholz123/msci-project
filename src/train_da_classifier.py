import config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from da_classifier import BiRNN_CRF_factory
from mappings import get_id2tag, get_tag2full_label
from utils import (
    get_embedding_matrix,
    get_tokenizer,
    make_model_readable_data,
    chunk,
)
from dataloader import load_corpus_data
import tensorflow as tf
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams.update({"figure.autolayout": True})

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# load config data
max_nr_utterances = config.data["max_nr_utterances"]
max_nr_words = config.data["max_nr_words"]
corpus = config.corpus["corpus"]
detail_level = config.corpus["detail_level"]
batch_size = config.model["batch_size"]
test_fraction = config.model["test_fraction"]
validation_fraction = config.model["validation_fraction"]
rnn_type = "gru"


def make_heatmap(data, labels, title, save_path, display_values=True):
    plt.figure(figsize=(8, 6))
    g = sns.heatmap(
        data,
        xticklabels=labels,
        yticklabels=labels,
        # cmap="YlOrRd",
        # cmap="hot",
        cmap="Blues",
        annot=display_values,
        square=True,
    )
    g.set_title(title, fontsize=16)
    g.set_xlabel("Predicted Label", fontsize=14)
    g.set_ylabel("True Label", fontsize=14)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

conversations, labels = load_corpus_data(corpus, detail_level)

conversations = chunk(conversations, max_nr_utterances)
labels = chunk(labels, max_nr_utterances)

n_tags = len(get_id2tag(corpus, detail_level=detail_level))

tokenizer = get_tokenizer(rebuild_from_all_words=False)
word2id = tokenizer.word_index

X, y = make_model_readable_data(
    conversations, labels, tokenizer, max_nr_utterances, max_nr_words
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# import pretrained embeddings
# set force rebuild to False when not changing total vocabulary
embedding_matrix = get_embedding_matrix(word2id, force_rebuild=False)

# model = get_bilstm_crf_model(embedding_matrix, n_tags)
model_factory = BiRNN_CRF_factory(embedding_matrix, n_tags, rnn_type)
model = model_factory.get()

data_name = corpus + "_detail_" + str(detail_level)
checkpoint_path = "../trained_model/bi" + rnn_type + "_crf/ckpt_" + data_name + ".hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True
)

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
# %%
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=3,
    validation_data=(X_test, y_test),
    callbacks=[model_checkpoint_callback],
)
# %%
# accuracy of biGru = 84.6% \pm 0.4%
# accuracy of biLSTM = 82.2% \pm 0.4%

accuracy = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

id2tag = get_id2tag(corpus)
tag2full = get_tag2full_label(corpus, detail_level)

labels = [tag2full.get(t) for t in id2tag.values()]

cm = confusion_matrix(y_test.flatten(), y_pred.flatten(), normalize="true")

#%%


flat_y = y_test.flatten()

ids, freqs = np.unique(flat_y, return_counts=True)
labels = [tag2full.get(id2tag[id]) for id in ids]

truncate = 10
important_cm = cm[0:truncate, 0:truncate]
important_cm = 100 * important_cm
important_cm = np.around(important_cm, 2).astype(np.int16)
# important_cm[important_cm < 1] = 0

plt.tight_layout()

make_heatmap(
    important_cm,
    labels[:truncate],
    "Normalised BiGru-CRF Confusion Matrix in %",
    "../figures/confusion_" + rnn_type + ".pdf",
)
