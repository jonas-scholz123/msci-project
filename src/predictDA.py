import numpy as np
import pickle
import re
import os

import tensorflow as tf
from utils import get_embedding_matrix, get_tokenizer, \
    make_model_readable_X, load_all_transcripts, merge_offset_arrays, load_one_transcript
from bilstm_crf import get_bilstm_crf_model
from mappings import get_id2tag, get_tag2full_label
import config

max_nr_utterances = config.data["max_nr_utterances"]
max_nr_words = config.data["max_nr_words"]
corpus = config.corpus["corpus"]
detail_level = config.corpus["detail_level"]

def make_annotated_transcript(transcript):
    '''
    Completes the end-to-end process for any given transcript.
    =========================================================
    PARAMS:
        transcript: list of lists of strings (chunked transcript pieces)

    OUTPUTS:
        annotated_transcript: list of tuples (utterance, DA(utterance))
    '''
    #get id2tag map and inverse
    id2tag = get_id2tag(corpus, detail_level = detail_level)
    tag2id = {t : id for id, t in id2tag.items()}
    tag2full = get_tag2full_label(corpus, detail_level)
    n_tags = len(tag2id.keys())

    tokenizer = get_tokenizer(rebuild_from_all_texts=False) #TODO set to false for final model
    word2id = tokenizer.word_index

    X = make_model_readable_X(transcript, tokenizer, max_nr_utterances, max_nr_words)

    # we create an offset version of the array so that we don't have contextless boundaries from chunking!
    flattened_X = X.reshape((X.shape[0]*X.shape[1], X.shape[-1]))
    offset_flattened_X = flattened_X[max_nr_utterances//2:-max_nr_utterances//2]
    offset_X = offset_flattened_X.reshape((
                    offset_flattened_X.shape[0]//max_nr_utterances,
                    max_nr_utterances,
                    offset_flattened_X.shape[-1]))

    # import pretrained GloVe embeddings
    embedding_matrix = get_embedding_matrix("../data/embeddings/glove.840B.300d.txt",
        word2id, force_rebuild=False) #set force rebuild to False when not changing total vocabulary

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

    model = get_bilstm_crf_model(embedding_matrix, max_nr_utterances, max_nr_words, n_tags)

    data_name = corpus + "_detail_" + str(detail_level)
    checkpoint_path = "../trained_model/bilstm_crf/ckpt_" + data_name + ".hdf5"
    if os.path.exists(checkpoint_path):
        print("loading trained weights...")
        model.load_weights(checkpoint_path)
        print("Done!")

    print("Making annotations...")
    y_hat = model.predict(X, batch_size=1).flatten()
    y_hat_offset = model.predict(offset_X, batch_size=1).flatten()

    y_hat = merge_offset_arrays(y_hat, y_hat_offset, step = max_nr_utterances//2)

    y_hat = [tag2full[id2tag[id]] for id in y_hat]

    u_joined_y_hat = []
    for t, y_hat_batch in zip(transcript, y_hat):
        u_joined_y_hat.append(tuple(zip(t, y_hat_batch)))

    #return annotated transcript
    print("Done!")
    return list(zip(sum(transcript, []), y_hat))

if __name__ == '__main__':
    #transcripts = load_all_transcripts(chunked=True, chunk_size=max_nr_utterances)
    #annotated_transcripts = [make_annotated_transcript(t) for t in transcripts]

    transcript = load_one_transcript("../transcripts/joe_rogan_elon_musk.txt",
        chunked=True, chunk_size=max_nr_utterances)
    annotated_transcript = make_annotated_transcript(transcript)
    annotated_transcript
