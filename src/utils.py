import numpy as np
import pickle
import os

def load_pretrained_glove(path):
    f = open(path, encoding='utf-8')
    print("Loading GloVe model, this can take some time...")
    glv_vector = {}
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    print("Completed loading GloVe model.")
    return glv_vector

def load_pretrained_matrix(path, word2id, force_new = False):
    fpath = "../helper_files/embedding_matrix.pkl"
    if not force_new and os.path.exists(fpath):
        with open(fpath, "rb") as f:
            matrix = pickle.load(f)
    else:
        glv_vector = load_pretrained_glove(path)
        dim = len(glv_vector[list(glv_vector.keys())[0]])
        matrix = np.zeros((len(word2id) + 1, dim))

        for word, id in word2id.items():
            try:
                matrix[id] = glv_vector[word]
            except KeyError:
                continue
        with open(fpath, "wb") as f:
            matrix = pickle.dump(matrix, f)
    return matrix

def pad_nested_sequences(sequences, max_nr_sentences, max_nr_words):
    # TODO: is it important to have 0s at the back?
    X = np.zeros((len(sequences), max_nr_sentences, max_nr_words), dtype="int32")
    for i, sequence in enumerate(sequences):
        for j, utterance in enumerate(sequence):
            if j < max_nr_words:
                X[i, j, :len(utterance)] = utterance

    with open('./training_matrix.pkl', 'wb') as f:
        pickle.dump(X, f)
    return X

def split_into_chunks(l, chunk_size):
    return [l[i:i+chunk_size] for i in range(0, len(l), chunk_size)]
