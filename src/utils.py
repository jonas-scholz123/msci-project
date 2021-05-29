import operator
import pickle
import os
import nltk
from tqdm import tqdm
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from dataloader import load_pretrained_conceptnet, get_all_words
import config


def get_embedding_matrix(word2id, force_rebuild=False):
    fpath = "../helper_files/embedding_matrix.pkl"
    if not force_rebuild and os.path.exists(fpath):
        with open(fpath, "rb") as f:
            matrix = pickle.load(f)
    else:
        # glv_vector = load_pretrained_glove(path)
        glv_vector = load_pretrained_conceptnet()
        dim = config.data["embedding_dim"]
        matrix = np.zeros((len(word2id) + 1, dim))

        for word, label in word2id.items():
            try:
                matrix[label] = glv_vector[word]
            except KeyError:
                continue
        with open(fpath, "wb") as matrix_file:
            pickle.dump(matrix, matrix_file)
    return matrix


def get_tokenizer(rebuild_from_all_words=False):

    tokenizer = Tokenizer(filters="")
    preloaded_exists = os.path.exists("../helper_files/tokenizer.pkl")
    if rebuild_from_all_words or not preloaded_exists:
        print("Building tokenizer from all words, this might take a while...")
        all_words = get_all_words()
        tokenizer.fit_on_texts(all_words)
        with open("../helper_files/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

    else:
        print("Found prebuilt tokenizer, loading...")
        with open("../helper_files/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    print("Done!")
    return tokenizer


def pad_nested_sequences(sequences, max_nr_sentences, max_nr_words):
    X = np.zeros((len(sequences), max_nr_sentences, max_nr_words), dtype="int32")
    for i, sequence in enumerate(sequences):
        for j, utterance in enumerate(sequence):
            if j < max_nr_words:
                if len(utterance) > max_nr_words:
                    # print("WARNING: utterance too long, will be truncated,
                    # increase max_nr_words!")
                    utterance = utterance[:max_nr_words]
                X[i, j, : len(utterance)] = utterance
    return X


def merge_offset_arrays(base, offset, step):
    """
    Merges offset array (where offset array is same shape as base except a chunk
    of size step missing both at the beginning and end) into the base array as follows:

    a = np.zeros((12)) #array of 0s
    b = np.zeros((8)) + 1 #array of 1s

    merge_offset_arrays(a, b, step=2)

    >>> array([0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.])
    """
    idx = step // 2
    new_idx = 0

    while new_idx < offset.shape[0] - step:
        new_idx = idx + step
        base[idx + step : new_idx + step] = offset[idx:new_idx]
        idx = new_idx + step
    return base


def convert_tag_to_id(iterable, tag2id):
    return [tag2id[tag] for tag in iterable]


def make_model_readable_data(
    conversations, labels, tokenizer, max_nr_utterances, max_nr_words
):
    X = make_model_readable_X(conversations, tokenizer, max_nr_utterances, max_nr_words)
    y = make_model_readable_y(labels, max_nr_utterances)
    return X, y


def make_model_readable_X(conversations, tokenizer, max_nr_utterances, max_nr_words):
    conversations = [
        [" ".join(nltk.word_tokenize(u)) for u in c] for c in conversations
    ]  # separates full stops etc
    conversation_sequences = [tokenizer.texts_to_sequences(c) for c in conversations]
    return pad_nested_sequences(conversation_sequences, max_nr_utterances, max_nr_words)


def make_model_readable_y(labels, max_nr_utterances):
    y = [[int(t_id) for t_id in t_ids] for t_ids in labels]
    return pad_sequences(y, max_nr_utterances, padding="post")


def turn_tags_to_id(labels, tag2id):
    return [[tag2id[lab] for lab in utterance_labels] for utterance_labels in labels]


def check_coverage(vocab, embeddings_index):
    # checks what fraction of words in vocab are in the embeddings
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except KeyError:
            oov[word] = vocab[word]
            i += vocab[word]

    print("Found embeddings for {:.2%} of vocab".format(float(len(a)) / len(vocab)))
    print("Found embeddings for  {:.2%} of all text".format(float(k) / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
