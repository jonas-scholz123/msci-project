import numpy as np
import pickle
import os
import nltk

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

    tokenizer = Tokenizer()

    if rebuild_from_all_texts:
        all_texts = get_all_texts()
        tokenizer.fit_on_texts(all_texts)
        with open("../helper_files/tokenizer.pkl", "wb") as f:
            tokenizer = pickle.dump(tokenizer, f)

    if os.path.exists("../helper_files/tokenizer.pkl"):
        print("Found tokenizer, loading...")
        with open("../helper_files/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

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
    conversation_sequences = [tokenizer.texts_to_sequences(c) for c in conversations]
    return pad_nested_sequences(conversation_sequences, max_nr_utterances, max_nr_words)

def make_model_readable_y(labels, max_nr_utterances):
    y = [[int(t_id) for t_id in t_ids] for t_ids in labels]
    return pad_sequences(y, max_nr_utterances, padding= "post")

def load_mrda_data(chunked = True, chunk_size = 100):

    with open('../data/clean/mrda_utterances.tsv', 'r') as f:
        lines = f.readlines()

    conversations = [[u for u in c.split('\t')[1:]] for c in lines]
    if chunked:
        chunked_conversations = [split_into_chunks(c, chunk_size) for c in conversations]
        chunked_conversations = sum(chunked_conversations, [])
        conversations = chunked_conversations #TODO test

    with open('../data/clean/mrda_labels.tsv', 'r') as f:
        lines = f.readlines()

    labels = [line.split("\t")[1:] for line in lines]

    #fix parsing of '\n' tag
    for l in labels[:-1]:
        l[-1] = l[-1][:-1]

    if chunked:
        chunked_labels = [split_into_chunks(l, chunk_size) for l in labels]
        chunked_labels = sum(chunked_labels, [])
        labels = chunked_labels #todo test

    return conversations, labels

def load_all_transcripts(transcript_dir = "../transcripts/", chunked = True,
    chunk_size = 100):

    transcripts = []
    for fpath in os.listdir(transcript_dir):
        with open(transcript_dir + fpath, 'r') as f:
            transcripts.append(f.readlines())

    if chunked:
        chunked_transcripts = [split_into_chunks(t, chunk_size) for t in transcripts]
        chunked_transcripts = sum(chunked_transcripts, [])
        transcripts = chunked_transcripts

    return transcripts

def get_all_texts():
    all_texts = []
    all_texts += sum(load_mrda_data(chunked = False)[0], [])
    all_texts += sum(load_switchboard_data()[0], [])
    all_texts += sum(load_all_transcripts(chunked = False), [])
    return all_texts

def turn_tags_to_id(labels, tag2id):
    return [[tag2id[l] for l in utterance_labels] for utterance_labels in labels]

def load_switchboard_data():

    corpus = CorpusReader('../data/switchboard-corpus/swda')

    excluded_tags = ['x', '+']
    conversations = []
    labels = []
    for transcript in corpus.iter_transcripts():
        utterances, utterance_labels = process_transcript_txt(transcript, excluded_tags)
        conversations.append(utterances)
        labels.append(utterance_labels)

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
