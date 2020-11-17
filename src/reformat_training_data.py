import pickle
import numpy as np
import os
from mappings import get_id2tag

def load_tag2id():
    with open("../metadata/mrda_id_to_tag.pkl", "rb") as f:
        id2tag =  pickle.load(f)
    return {tag : id for id, tag in id2tag.items()}

def load_id2tag():
    with open("../metadata/mrda_id_to_tag.pkl", "rb") as f:
        return pickle.load(f)

def make_id2tag(fpath, labels):
    print("ALL", labels)
    unique_labels = list(set(labels))
    print("UNIQUE", unique_labels)
    id2tag = {id : tag for id, tag in enumerate(unique_labels)}

    with open(fpath, "wb") as f:
        pickle.dump(id2tag, f)

    return id2tag

#def get_id2tag(fpath, labels=None):
#    if os.path.exists(fpath):
#        with open(fpath, "rb") as f:
#            return pickle.load(f)
#    else:
#        return make_id2tag(fpath, labels)


def read_mrda_file(fname, detail_level = 2):

    if detail_level == 0:
        id2tag_path = "../helper_files/mrda_basic_id_to_tag.pkl"
    elif detail_level == 1:
        id2tag_path = "../helper_files/mrda_general_id_to_tag.pkl"
    elif detail_level == 2:
        id2tag_path = "../helper_files/mrda_full_id_to_tag.pkl"

    with open(fname, "r") as f:
        lines = f.readlines()

    split_lines = [l.split("|") for l in lines]
    utterances = [l[1] for l in split_lines]

    tags = [l[detail_level + 2].replace("\n", "") for l in split_lines]
    id2tag = get_id2tag("mrda", detail_level)
    tag2id = {t : id for id, t in id2tag.items()}
    ids = [tag2id[tag] for tag in tags]
    return utterances, ids

def read_mrda_training_data(detail_level = 0):
    training_dir = "../data/mrda_corpus/train"
    test_dir = "../data/mrda_corpus/test"
    val_dir = "../data/mrda_corpus/val"

    labels_list = []
    utterances_list = []

    for dir in [training_dir, test_dir, val_dir]:
        filenames = os.listdir(dir)

        for fname in filenames:
            fpath = dir + "/" + fname
            utterances, ids = read_mrda_file(fpath, detail_level)

            #id_string = fname[:-4] + "\t" + "\t".join([str(id) for id in ids])
            #utterances_string = fname[:-4] + "\t" + "\t".join([u for u in utterances])

            utterances_list.append(utterances)
            labels_list.append(ids)

            #label_fpath = "../data/clean/mrda_labels.tsv"

            #with open(label_fpath, "w") as f:
            #    f.write("\n".join(labels_list))

            #training_utterance_fpath = "../data/clean/mrda_utterances.tsv"

            #with open(training_utterance_fpath, "w") as f:
                #f.write("\n".join(utterances_list))
    return utterances_list, labels_list



if __name__ == "__main__":
    u, l = read_mrda_training_data(detail_level=2)
