import numpy as np
import pickle
import os
import nltk
import operator
import re
from tqdm import tqdm
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from swda import CorpusReader
from mappings import get_id2tag

def load_pretrained_glove(path):
    pkl_path = "../helper_files/glove.840B.300d.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            glove = pickle.load(f)
    else:
        print("Loading GloVe model, this can take some time...")
        f = open(path, encoding='utf-8')
        print("Loading GloVe model, this can take some time...")
        glove = {}
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float')
                glove[word] = coefs
            except ValueError:
                continue
        f.close()
        print("Completed loading GloVe model.")
    return glove

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

def merge_offset_arrays(base, offset, step):
    """
    Merges offset array (where offset array is same shape as base except a chunk
    of size step missing both at the beginning and end) into the base array as follows:

    a = np.zeros((12)) #array of 0s
    b = np.zeros((8)) + 1 #array of 1s

    merge_offset_arrays(a, b, step=2)

    >>> array([0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.])
    """
    idx = step//2
    new_idx = 0

    while new_idx < offset.shape[0] - step:
        new_idx = idx + step
        base[idx + step : new_idx + step] = offset[idx : new_idx]
        idx = new_idx + step
    return base

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

    print("Done!")
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
    transcripts = load_all_transcripts(chunked = False)
    transcript_texts = [entry[0] for t in transcripts for entry in t]
    all_texts += transcript_texts

    all_texts = [(" ".join(nltk.word_tokenize(s))).lower() for s in all_texts]
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
        entries = load_one_transcript(transcript_dir + fpath,
            chunked = chunked, chunk_size=chunk_size)
        transcripts.append(entries)

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


def generate_confusion_matrix(data, predictions, metadata, verbose=False):
    # ONLY SHOWS FIRST 5!
    # Get label data
    labels = data['labels']

    # Get metadata
    index_to_label = metadata['index_to_label']
    label_to_index = metadata['label_to_index']
    num_labels = metadata['num_labels']


    # Create empty confusion matrix
    confusion_matrix = np.zeros(shape=(num_labels, num_labels), dtype=int)

    # For each prediction
    for i in range(len(predictions)):
        # Get prediction with highest probability
        prediction = np.argmax(predictions[i])

        # Add to matrix
        confusion_matrix[label_to_index[labels[i]]][prediction] += 1

    if verbose:
        # Print confusion matrix
        print("------------------------------------")
        print("Confusion Matrix:")
        print('{:15}'.format(" "), end='')
        for j in range(confusion_matrix.shape[1]):
            print('{:15}'.format(index_to_label[j]), end='')
        print()
        for j in range(confusion_matrix.shape[0]):
            print('{:15}'.format(index_to_label[j]), end='')
            print('\n'.join([''.join(['{:10}'.format(item) for item in confusion_matrix[j]])]))

    return confusion_matrix


def plot_history(history, title='History'):
    # Create figure and title
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    fig.suptitle(title, fontsize=14)

    # Plot accuracy
    acc = fig.add_subplot(121)
    acc.plot(history['accuracy'])
    #acc.plot(history['val_acc'])
    acc.set_ylabel('Accuracy')
    acc.set_xlabel('Epoch')

    # Plot loss
    loss = fig.add_subplot(122)
    loss.plot(history['loss'])
    loss.plot(history['val_loss'])
    loss.set_ylabel('Loss')
    loss.set_xlabel('Epoch')
    loss.legend(['Train', 'Test'], loc='upper right')

    # Adjust layout to fit title
    fig.tight_layout()
    fig.subplots_adjust(top=0.15)

    return fig


def plot_confusion_matrix(matrix, classes,  title='', matrix_size=10, normalize=False, color='black', cmap='viridis'):

    # Number of elements of matrix to show
    if matrix_size:
        matrix = matrix[:matrix_size, :matrix_size]
        classes = classes[:matrix_size]

    # Normalize input matrix values
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        value_format = '.2f'
    else:
        value_format = 'd'

    # Create figure with two axis and a colour bar
    fig, ax = plt.subplots(ncols=1, figsize=(5, 5))

    # Generate axis and image
    ax, im = plot_matrix_axis(matrix, ax, classes, title, value_format, color=color, cmap=cmap)

    # Add colour bar
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    color_bar = fig.colorbar(im, cax=colorbar_ax)
    # Tick color
    color_bar.ax.yaxis.set_tick_params(color=color)
    # Tick labels
    plt.setp(plt.getp(color_bar.ax.axes, 'yticklabels'), color=color)
    # Edge color
    color_bar.outline.set_edgecolor(color)

    # Set layout
    fig.tight_layout()

    return fig

def _find_timestamp(s):
    timestamps = re.findall(r'\d{2}:\d{2}:\d{2}|\d{2}:\d{2}', s)
    if timestamps:
        timestamp = timestamps[0]
        if len(timestamp) == 5: #if of shape "01:37"
            timestamp = "00:" + timestamp
        return timestamp
    return None

def _find_speaker(text, all_speakers):
    for speaker in all_speakers:
        potential_speaker = re.findall(speaker, text.lower())
        if potential_speaker:
            return potential_speaker[0]
    return None

def get_speakers():
    transcript_paths = os.listdir("../transcripts")
    transcript_names = [p.split(".")[0] for p in transcript_paths]
    speakers = ([" ".join(n.split("_")[0:2]) for n in transcript_names] +
                    [" ".join(n.split("_")[2:4]) for n in transcript_names])
    return list(set(speakers))

def load_one_transcript(fpath, chunked = True, chunk_size = 100):
    with open(fpath, 'r') as f:
        transcript = f.read().replace("...", "")
    transcript = transcript.split("\n")

    all_speakers = get_speakers()
    speakers = []
    timestamps = []
    conversation = []

    current_timestamp = ""
    current_speaker = ""

    entries = []
    for line in transcript:
        if line == "":
            continue

        if _find_timestamp(line):
            current_timestamp = _find_timestamp(line)

        if _find_speaker(line, all_speakers):
            current_speaker = _find_speaker(line, all_speakers)

        if not (_find_speaker(line, all_speakers) or _find_timestamp(line)):
            split_line = re.split("([.?;!])", line)
            utterance_texts = split_line[::2][:-1]
            utterance_signs = split_line[1::2]

            for i in range(len(utterance_texts)):
                utterance = "".join([utterance_texts[i], utterance_signs[i]])
                utterance = " ".join(nltk.word_tokenize(utterance))
                entries.append((utterance, current_speaker, current_timestamp))

    if chunked:
        entries = split_into_chunks(entries, chunk_size)

    return entries

if __name__ == '__main__':
    transcripts = load_all_transcripts("../transcripts/", chunked = True)
    transcripts[0][0]
