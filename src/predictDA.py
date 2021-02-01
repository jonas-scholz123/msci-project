import pandas as pd
import os

from utils import get_embedding_matrix, get_tokenizer, \
    make_model_readable_X, load_all_transcripts, merge_offset_arrays, load_one_transcript
from bilstm_crf import get_bilstm_crf_model
from mappings import get_id2tag, get_tag2full_label
import config


class DA_classifier:
    def __init__(self, verbose=False):

        self.max_nr_utterances = config.data["max_nr_utterances"]
        self.max_nr_words = config.data["max_nr_words"]
        self.corpus = config.corpus["corpus"]
        self.detail_level = config.corpus["detail_level"]

        self.verbose = verbose

        self.id2tag = get_id2tag(self.corpus, detail_level=self.detail_level)
        self.tag2id = {t: id for id, t in self.id2tag.items()}
        self.tag2full = get_tag2full_label(self.corpus, self.detail_level)
        self.n_tags = len(self.tag2id.keys())

        self.tokenizer = get_tokenizer(rebuild_from_all_texts=False)
        word2id = self.tokenizer.word_index

        # WARNING: if you force rebuild, the embedding matrix
        # may change and you may need to retrain the Neural Network!

        # set force rebuild to False when not changing total vocabulary
        self.embedding_matrix = get_embedding_matrix(
            "../data/embeddings/glove.840B.300d.txt",
            word2id,
            force_rebuild=False)

        # use GPU
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        self.model = get_bilstm_crf_model(
            self.embedding_matrix,
            self.n_tags)

        data_name = self.corpus + "_detail_" + str(self.detail_level)
        checkpoint_path = ("../trained_model/bilstm_crf/ckpt_"
                           + data_name
                           + ".hdf5")
        if os.path.exists(checkpoint_path):
            if self.verbose:
                print("loading trained weights...")
            self.model.load_weights(checkpoint_path)
            if self.verbose:
                print("Done!")
        else:
            print("WARNING: no model found in path, using untrained model!")

    def get_annotated_transcript(self, fname, force_rebuild=False):
        '''
        Wrapper for make_annotated_transcript, checks if df already exists and
        if so, just loads instead of rebuilding.

        PARAMS:
            str fname: name, without file extension, of transcript file e.g.
            joe_rogan_elon_musk

            Optional:
            bool force_rebuild: rebuilds from scratch even if transcript_df
                                already exists (default = False)
        RETURNS:
            pd.DataFrame transcript_df: annotated transcript
        '''
        # load from config
        max_nr_utterances = config.data["max_nr_utterances"]

        transcript_dir = config.paths["transcripts"]
        df_dir = config.paths["transcript_dfs"]

        transcript_path = transcript_dir + fname + ".txt"
        df_path = df_dir + fname + ".csv"

        if not os.path.exists(df_path) or force_rebuild:
            transcript = load_one_transcript(
                transcript_path,
                chunked=True,
                chunk_size=max_nr_utterances)
            transcript_df = self.make_annotated_transcript(transcript)
            transcript_df.to_csv(df_path, index=False)
        else:
            transcript_df = pd.read_csv(df_path)

        return transcript_df

    def get_all_annotated_transcripts(self, force_rebuild=False):
        '''
        Wrapper for get_annotated_transcript, gets all transcripts at once
        '''

        transcript_dir = config.paths["transcripts"]

        transcript_dfs = []

        for transcript_name in os.listdir(transcript_dir):
            transcript_dfs.append(self.get_annotated_transcript(
                transcript_name.split(".")[0], force_rebuild=force_rebuild))
        return transcript_dfs

    def make_annotated_transcript(self, transcript, verbose=False, model=None):
        '''
        Completes the end-to-end process for any given transcript.
        =========================================================
        PARAMS:
            transcript: list of lists of strings (chunked transcript pieces)

        OUTPUTS:
            annotated_transcript: list of tuples (utterance, DA(utterance))
        '''

        transcript_text = [[e[0].lower() for e in chunk]
                           for chunk in transcript]

        total_nr_utterances = len(sum(transcript_text, []))

        X = make_model_readable_X(transcript_text, self.tokenizer,
                                  self.max_nr_utterances, self.max_nr_words)

        # we create an offset version of the array
        # so that we don't have contextless boundaries from chunking!
        flattened_X = X.reshape((X.shape[0]*X.shape[1], X.shape[-1]))
        offset_flattened_X = flattened_X[
            self.max_nr_utterances//2:-self.max_nr_utterances//2]
        offset_X = offset_flattened_X.reshape((
                        offset_flattened_X.shape[0]//self.max_nr_utterances,
                        self.max_nr_utterances,
                        offset_flattened_X.shape[-1]))

        y_hat = self.model.predict(X, batch_size=1).flatten()
        y_hat_offset = self.model.predict(offset_X, batch_size=1).flatten()

        y_hat = merge_offset_arrays(y_hat, y_hat_offset,
                                    step=self.max_nr_utterances//2)

        y_hat = [self.tag2full[self.id2tag[id]] for id in y_hat]

        y_hat = y_hat[:total_nr_utterances]  # remove trailing 0's from padding

        u_joined_y_hat = []
        for t, y_hat_batch in zip(transcript_text, y_hat):
            u_joined_y_hat.append(tuple(zip(t, y_hat_batch)))

        # return annotated transcript
        if verbose:
            print("Done!")
        transcript_df = pd.DataFrame.from_records(
            sum(transcript, []),
            columns=["utterance", "speaker", "timestamp"])

        transcript_df["da_label"] = y_hat

        return transcript_df


if __name__ == '__main__':
    DAC = DA_classifier(verbose=False)

    t_path = os.listdir("../transcripts")[0]
    transcript = load_one_transcript("../transcripts/" + t_path)
    annotated_transcripts = DAC.make_annotated_transcript(transcript)
