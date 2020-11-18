import numpy as np
import pickle
import os
import nltk
import operator
import re
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from swda import CorpusReader
from mappings import get_id2tag

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

def get_embedding_matrix(path, word2id, force_rebuild = False):
    fpath = "../helper_files/embedding_matrix.pkl"
    if not force_rebuild and os.path.exists(fpath):
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
            pickle.dump(matrix, f)
    return matrix

def pad_nested_sequences(sequences, max_nr_sentences, max_nr_words):
    # TODO: is it important to have 0s at the back?
    X = np.zeros((len(sequences), max_nr_sentences, max_nr_words), dtype="int32")
    for i, sequence in enumerate(sequences):
        for j, utterance in enumerate(sequence):
            if j < max_nr_words:
                if len(utterance) > max_nr_words:
                    print("WARNING: utterance too long, will be truncated, increase max_nr_words!")
                    utterance = utterance[:max_nr_words]
                X[i, j, :len(utterance)] = utterance
    return X

def chunk(l_of_ls, chunk_size):
    chunked = [split_into_chunks(l, chunk_size) for l in l_of_ls]
    return sum(chunked, [])

def split_into_chunks(l, chunk_size):
    return [l[i:i+chunk_size] for i in range(0, len(l), chunk_size)]

#def get_id2tag(corpus):
    #open tag2id mapping for labels and create inverse
    #with open('../helper_files/' + corpus + '_id_to_tag.pkl', 'rb') as f:
        #id2tag = pickle.load(f)
    #return id2tag

def get_tokenizer(rebuild_from_all_texts = False):

    tokenizer = Tokenizer(filters="")
    preloaded_exists = os.path.exists("../helper_files/tokenizer.pkl")
    if rebuild_from_all_texts or not preloaded_exists:
        print("Building tokenizer from all words, this might take a while...")
        all_texts = get_all_texts()
        tokenizer.fit_on_texts(all_texts)
        with open("../helper_files/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

    else:
        print("Found prebuilt tokenizer, loading...")
        with open("../helper_files/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    print("Done")
    return tokenizer

def convert_tag_to_id(l, tag2id):
    return [tag2id[tag] for tag in l]

def make_model_readable_data(conversations, labels, tokenizer,
    max_nr_utterances, max_nr_words):
    #TODO CHECK THAT EVERYTHING IS PADDED IN THE RIGHT PLACE
    X = make_model_readable_X(conversations, tokenizer, max_nr_utterances, max_nr_words)
    y = make_model_readable_y(labels, max_nr_utterances)
    return X, y

def make_model_readable_X(conversations, tokenizer, max_nr_utterances, max_nr_words):
    conversations = [[" ".join(nltk.word_tokenize(u)) for u in c] for c in conversations] #separates full stops etc
    conversation_sequences = [tokenizer.texts_to_sequences(c) for c in conversations]
    return pad_nested_sequences(conversation_sequences, max_nr_utterances, max_nr_words)

def make_model_readable_y(labels, max_nr_utterances):
    y = [[int(t_id) for t_id in t_ids] for t_ids in labels]
    return pad_sequences(y, max_nr_utterances, padding= "post")


def get_all_texts():
    all_texts = []
    all_texts += sum(load_mrda_data()[0], [])
    all_texts += sum(load_swda_data()[0], [])
    all_texts += sum(load_all_transcripts(chunked = False), [])
    all_texts = [" ".join(nltk.word_tokenize(s)) for s in all_texts]
    return all_texts

def turn_tags_to_id(labels, tag2id):
    return [[tag2id[l] for l in utterance_labels] for utterance_labels in labels]

def load_corpus_data(corpus, detail_level=0):
    print("loading corpus: ", corpus)
    if corpus == 'swda':
        data = load_swda_data()
    elif corpus == 'mrda':
        data = load_mrda_data(detail_level)
    print("Done!")
    return data

def load_swda_data():

    if not os.path.exists("../helper_files/swda_data.pkl"):
        corpus = CorpusReader('../data/switchboard-corpus/swda')
        excluded_tags = ['x', '+']
        conversations = []
        labels = []
        print('Loading swda transcripts, this might take a while')
        for transcript in corpus.iter_transcripts():
            utterances, utterance_labels = process_transcript_txt(transcript, excluded_tags)
            conversations.append(utterances)
            labels.append(utterance_labels)

        with open("../helper_files/swda_data.pkl", "wb") as f:
            pickle.dump((conversations, labels), f)
    else:
        with open("../helper_files/swda_data.pkl", "rb") as f:
            conversations, labels = pickle.load(f)

    return conversations, labels

def process_transcript_txt(transcript, excluded_tags=None):
    # Special characters for ignoring i.e. <laughter>
    special_chars = {'<', '>', '(', ')', '#'}

    utterances = []
    labels = []

    id2tag = get_id2tag("swda", detail_level=None)
    tag2id = {t : id for id, t in id2tag.items()}

    for utt in transcript.utterances:

        utterance = []
        for word in utt.text_words(filter_disfluency=True):

            # Remove the annotations that filter_disfluency does not (i.e. <laughter>)
            if all(char not in special_chars for char in word):
                utterance.append(word)

        # Join words for complete sentence
        utterance_sentence = " ".join(utterance)

        # Print original and processed utterances
        # print(utt.transcript_index, " ", utt.text_words(filter_disfluency=True), " ", utt.damsl_act_tag())
        # print(utt.transcript_index, " ", utterance_sentence, " ", utt.damsl_act_tag())

        # Check we are not adding an empty utterance (i.e. because it was just <laughter>)
        if len(utterance) > 0 and utt.damsl_act_tag() not in excluded_tags:
            utterances.append(" ".join(nltk.word_tokenize(utterance_sentence.lower()))) # this separates ?, ! etc from words
            labels.append(tag2id[utt.damsl_act_tag()])
    return utterances, labels

def load_mrda_data(detail_level = 0):
    training_dir = "../data/mrda_corpus/train"
    test_dir = "../data/mrda_corpus/test"
    val_dir = "../data/mrda_corpus/val"

    labels_list = []
    utterances_list = []

    for dir in [training_dir, test_dir, val_dir]:
        filenames = os.listdir(dir)

        for fname in filenames:
            fpath = dir + "/" + fname

            with open(fpath, "r") as f:
                lines = f.readlines()

            split_lines = [l.split("|") for l in lines]
            utterances = [l[1] for l in split_lines]

            tags = [l[detail_level + 2].replace("\n", "") for l in split_lines]
            id2tag = get_id2tag("mrda", detail_level)
            tag2id = {t : id for id, t in id2tag.items()}
            ids = [tag2id[tag] for tag in tags]

            utterances_list.append(utterances)
            labels_list.append(ids)
    return utterances_list, labels_list


def load_all_transcripts(transcript_dir = "../transcripts/", chunked = True,
    chunk_size = 100):

    transcripts = []
    for fpath in os.listdir(transcript_dir):
        with open(transcript_dir + fpath, 'r') as f:
            transcript = f.read()
        transcript = transcript.split("\n")

        conversation = transcript[1::3]
        conversation = " ".join(conversation).lower().replace("...", "")
        conversation = np.asarray(nltk.word_tokenize(conversation))

        sentence_boundary_indices = []
        for i, token in enumerate(conversation):
            if token in [".", "?", "!", ";"]:
                sentence_boundary_indices.append(i + 1)

        utterances = [" ".join(u) for u in np.split(conversation, sentence_boundary_indices)]
        transcripts.append(utterances)

    if chunked:
        return chunk(transcripts, chunk_size)

    return transcripts

def check_coverage(vocab,embeddings_index):
    #checks what fraction of words in vocab are in the embeddings
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
