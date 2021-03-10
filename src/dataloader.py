import os
import pickle
import nltk
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

import config
from swda import CorpusReader
from mappings import get_id2tag


def chunk(l_of_ls, chunk_size):
    chunked = [split_into_chunks(l, chunk_size) for l in l_of_ls]
    return sum(chunked, [])


def split_into_chunks(l, chunk_size):
    return [l[i : i + chunk_size] for i in range(0, len(l), chunk_size)]


def load_pretrained_glove(path):
    pkl_path = "../helper_files/glove.840B.300d.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            glove = pickle.load(f)
    else:
        print("Loading GloVe model, this can take some time...")
        f = open(path, encoding="utf-8")
        print("Loading GloVe model, this can take some time...")
        glove = {}
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float")
                glove[word] = coefs
            except ValueError:
                continue
        f.close()
        print("Completed loading GloVe model.")
    return glove


class ConceptNetDict:
    def __init__(self):
        path = config.paths["embeddings"] + "en_mini_conceptnet.h5"
        self.df = pd.read_hdf(path, "data")

    def __getitem__(self, idx):
        return self.df.loc[idx].values

    def __contains__(self, idx):
        return self.get(idx) is not None

    def get(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            return


def load_pretrained_conceptnet():
    return ConceptNetDict()

def extract_words(tdf):
    words = set()
    tdf["utterance"].str.lower().str.split().apply(words.update)
    return words

def get_all_words():
    all_words = set()
    all_words = all_words.union(
        set(nltk.word_tokenize(" ".join(sum(load_mrda_data()[0], [])).lower()))
    )
    all_words = all_words.union(
        set(nltk.word_tokenize(" ".join(sum(load_swda_data()[0], [])).lower()))
    )

    t_words = set()
    for entry in tqdm(os.walk("../processed_transcripts/")):
        folder = entry[0]
        for fname in entry[-1]:
            if not fname.endswith(".pkl"):
                continue
            fpath = folder + "/" + fname
            t_words = t_words.union(extract_words(pd.read_pickle(fpath)))

    all_words = list(all_words.union(t_words))
    return all_words


def load_corpus_data(corpus, detail_level=0):
    print("loading corpus: ", corpus)
    if corpus == "swda":
        data = load_swda_data()
    elif corpus == "mrda":
        data = load_mrda_data(detail_level)
    print("Done!")
    return data


def load_swda_data():

    if not os.path.exists("../helper_files/swda_data.pkl"):
        corpus = CorpusReader("../data/switchboard-corpus/swda")
        excluded_tags = ["x", "+"]
        conversations = []
        labels = []
        print("Loading swda transcripts, this might take a while")
        for transcript in corpus.iter_transcripts():
            utterances, utterance_labels = process_transcript_txt(
                transcript, excluded_tags
            )
            conversations.append(utterances)
            labels.append(utterance_labels)

        with open("../helper_files/swda_data.pkl", "wb") as f:
            pickle.dump((conversations, labels), f)
    else:
        with open("../helper_files/swda_data.pkl", "rb") as f:
            conversations, labels = pickle.load(f)

    return conversations, labels


def load_mrda_data(detail_level=0):
    training_dir = "../data/mrda_corpus/train"
    test_dir = "../data/mrda_corpus/test"
    val_dir = "../data/mrda_corpus/val"

    labels_list = []
    utterances_list = []

    for directory in [training_dir, test_dir, val_dir]:
        filenames = os.listdir(directory)

        for fname in filenames:
            fpath = directory + "/" + fname

            with open(fpath, "r") as f:
                lines = f.readlines()

            split_lines = [line.split("|") for line in lines]
            utterances = [line[1] for line in split_lines]

            tags = [line[detail_level + 2].replace("\n", "") for line in split_lines]
            id2tag = get_id2tag("mrda", detail_level)
            tag2id = {t: id for id, t in id2tag.items()}
            ids = [tag2id[tag] for tag in tags]

            utterances_list.append(utterances)
            labels_list.append(ids)
    return utterances_list, labels_list


def process_transcript_txt(transcript, excluded_tags=None):
    # Special characters for ignoring i.e. <laughter>
    special_chars = {"<", ">", "(", ")", "#"}

    utterances = []
    labels = []

    id2tag = get_id2tag("swda", detail_level=None)
    tag2id = {t: id for id, t in id2tag.items()}

    for utt in transcript.utterances:

        utterance = []
        for word in utt.text_words(filter_disfluency=True):

            # Remove the annotations that filter_disfluency does not (i.e.
            # <laughter>)
            if all(char not in special_chars for char in word):
                utterance.append(word)

        # Join words for complete sentence
        utterance_sentence = " ".join(utterance)

        # Print original and processed utterances
        # print(utt.transcript_index, " ", utterance_sentence, " ", utt.damsl_act_tag())

        # Check we are not adding an empty utterance (i.e. because it was just
        # <laughter>)
        if len(utterance) > 0 and utt.damsl_act_tag() not in excluded_tags:
            # this separates ?, ! etc from words
            utterances.append(" ".join(nltk.word_tokenize(utterance_sentence.lower())))
            labels.append(tag2id[utt.damsl_act_tag()])
    return utterances, labels


def load_all_transcripts(
    transcript_dir="../transcripts/",
    chunked=True,
    chunk_size=100,
    return_fnames=False,
    max_nr=None,
):

    transcripts = []
    fnames = []

    counter = 0

    for fpath in os.listdir(transcript_dir):
        if fpath.startswith("spotify"):
            entries = load_spotify_transcript(
                transcript_dir + fpath, chunked=chunked, chunk_size=chunk_size
            )
        else:
            entries = load_one_transcript(
                transcript_dir + fpath, chunked=chunked, chunk_size=chunk_size
            )
        counter += 1
        transcripts.append(entries)
        fnames.append(fpath.split(".")[-2])
        if max_nr is not None and counter >= max_nr:
            break

    if return_fnames:
        return transcripts, fnames
    return transcripts


def load_all_processed_transcripts(return_fnames=False):
    directory = config.paths["tdfs"]
    tdfs = []
    fnames = []
    for fname in os.listdir(directory):
        if fname.endswith("pkl"):
            fpath = directory + fname
            tdf = pd.read_pickle(fpath)
            tdfs.append(tdf)
            fnames.append(fname.split(".")[0])
    if return_fnames:
        return tdfs, fnames
    return tdfs


def find_timestamp(string):
    timestamps = re.findall(r"\d{2}:\d{2}:\d{2}|\d{2}:\d{2}", string)
    if timestamps:
        timestamp = timestamps[0]
        if len(timestamp) == 5:  # if of shape "01:37"
            timestamp = "00:" + timestamp
        return timestamp
    return None


def find_speaker(text, all_speakers):
    for speaker in all_speakers:
        potential_speaker = re.findall(speaker, text.lower())
        if potential_speaker:
            return potential_speaker[0]
    return None


def get_speakers():
    transcript_paths = os.listdir("../transcripts")
    transcript_names = [p.split(".")[0] for p in transcript_paths]
    speakers = [" ".join(n.split("_")[0:2]) for n in transcript_names] + [
        " ".join(n.split("_")[2:4]) for n in transcript_names
    ]
    return list(set(speakers))


def load_one_transcript(fpath, chunked=True, chunk_size=100):

    if "spotify" in fpath:
        return load_spotify_transcript(fpath, chunked, chunk_size)
    with open(fpath, "r") as f:
        transcript = f.read().replace("...", "")
    transcript = transcript.split("\n")

    all_speakers = get_speakers()

    current_timestamp = ""
    current_speaker = ""

    entries = []
    for line in transcript:
        if line == "":
            continue

        if find_timestamp(line):
            current_timestamp = find_timestamp(line)

        if find_speaker(line, all_speakers):
            current_speaker = find_speaker(line, all_speakers)

        if not (find_speaker(line, all_speakers) or find_timestamp(line)):
            split_line = re.split("([.?;!])", line)
            utterance_texts = split_line[::2][:-1]
            utterance_signs = split_line[1::2]

            for i, _ in enumerate(utterance_texts):
                utterance = "".join([utterance_texts[i], utterance_signs[i]])
                utterance = " ".join(nltk.word_tokenize(utterance))
                entries.append((utterance, current_speaker, current_timestamp))

    if chunked:
        entries = split_into_chunks(entries, chunk_size)

    return entries


def load_spotify_transcript(fpath, chunked=True, chunk_size=100):
    with open(fpath, "r") as f:
        lines = f.readlines()

    times_and_speakers = lines[0::3]
    utterances = lines[1::3]

    speakers = [tns.split(":")[0] for tns in times_and_speakers]
    times = [find_timestamp(tns) for tns in times_and_speakers]
    utterances = [u.replace("\n", "") for u in utterances]
    utterances = [" ".join(nltk.word_tokenize(u)) for u in utterances]

    entries = list(zip(utterances, speakers, times))
    if chunked:
        entries = split_into_chunks(entries, chunk_size)
    return entries
