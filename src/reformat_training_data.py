import pickle
import numpy as np
import os

def load_tag2id():
    with open("../metadata/mrda_id_to_tag.pkl", "rb") as f:
        id2tag =  pickle.load(f)
    return {tag : id for id, tag in id2tag.items()}

def load_id2tag():
    with open("../metadata/mrda_id_to_tag.pkl", "rb") as f:
        return pickle.load(f)

def read_data(fname):
    tag2id = load_tag2id()

    with open(fname, "r") as f:
        lines = f.readlines()

    split_lines = [l.split("|") for l in lines]
    utterances = [l[1] for l in split_lines]
    tags = [l[-1][:-1] for l in split_lines]
    ids = [tag2id[tag] for tag in tags]
    return utterances, ids

def make_mrda_training_data():
    training_dir = "../data/mrda_corpus/train"
    filenames = os.listdir(training_dir)

    labels_list = []
    utterances_list = []

    for fname in filenames:
        fpath = training_dir + "/" + fname
        utterances, ids = read_data(fpath)

        id_string = fname[:-4] + "\t" + "\t".join([str(id) for id in ids])
        utterances_string = fname[:-4] + "\t" + "\t".join([u for u in utterances])

        labels_list.append(id_string)
        utterances_list.append(utterances_string)

        label_fpath = "../data/clean/mrda_labels.tsv"

        with open(label_fpath, "w") as f:
            f.write("\n".join(labels_list))

        training_utterance_fpath = "../data/clean/mrda_utterances.tsv"

        with open(training_utterance_fpath, "w") as f:
            f.write("\n".join(utterances_list))
    return



if __name__ == "__main__":
    make_mrda_training_data()
