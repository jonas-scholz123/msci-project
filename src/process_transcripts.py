from datetime import datetime
from tqdm import tqdm
import os
import numpy as np
import config
from utils import load_one_transcript
from predictDA import DA_classifier
from topics import TopicExtractor
import multiprocessing as mp


def timestamp_to_datetime(timestamp):
    return datetime.strptime(timestamp, "%H:%M:%S")


def get_fractional_time(current_timestamp, biggest_time):
    try:
        current_seconds = (
            biggest_time - timestamp_to_datetime(current_timestamp)
        ).total_seconds()

        total_seconds = (
            biggest_time - timestamp_to_datetime("00:00:00")
        ).total_seconds()
        return 1 - current_seconds / total_seconds
    except BaseException:
        return current_timestamp


def enhance_tdf(df):
    # calc relative time
    biggest_time = timestamp_to_datetime(df.loc[df.shape[0] - 1, "timestamp"])
    df["relative_time"] = df["timestamp"].apply(
        get_fractional_time, args=(biggest_time,)
    )

    # explicitly find speaker change positions
    speaker_change = np.zeros(df.shape[0], dtype=np.int32)
    speakers = df["speaker"]
    for i, speaker in speakers[:-1].iteritems():
        if speaker != speakers[i + 1]:
            speaker_change[i + 1] = 1
    df["speaker_change"] = speaker_change

    # length of utterances in seconds
    timestamps = df["timestamp"]
    lengths = np.zeros(df.shape[0])
    for i, timestamp in timestamps[:-1].iteritems():
        next_timestamp = timestamps[i + 1]
        lengths[i] = (
            timestamp_to_datetime(next_timestamp) - timestamp_to_datetime(timestamp)
        ).total_seconds()
    df["utterance_length"] = lengths
    return df


def process(fname):
    try:
        fname = fname.split(".")[0]
        transcript = load_one_transcript(config.paths["transcripts"] + fname + ".txt")
        # only process unprocessed transcripts
        tdf = process.dac.make_annotated_transcript(transcript)
        tdf = enhance_tdf(tdf)
        tdf = process.te.process(tdf)
        # csvs are human readable
        tdf.to_csv("../processed_transcripts/" + fname + ".csv", index=False)
        # pickle preserves objects such as lists, sets
        tdf.to_pickle("../processed_transcripts/" + fname + ".pkl")
    except UnboundLocalError:
        return


def process_init():
    process.dac = DA_classifier()
    process.te = TopicExtractor()


def process_all_transcripts(force_rebuild=False, max_nr=30):
    # TODO: make a central meta file that checks what stage every transcript
    # is in

    # te = TopicExtractor()
    # dac = DA_classifier()

    root = config.paths["transcripts"]
    fnames = os.listdir(root)

    # counter = 0
    mp.set_start_method("spawn")  # allows CUDA multiprocessing

    print("total nr transcripts: ", len(fnames))
    already_processed = set(os.listdir(config.paths["tdfs"]))
    # pkl and csv files
    print("already processed: ", len(already_processed) // 2)

    if not force_rebuild:
        fnames = [
            fname
            for fname in fnames
            if fname.split(".")[0] + ".pkl" not in already_processed
        ]

    n_remaining = len(fnames)
    print("remaining: ", n_remaining)

    nr_processes = config.processing["nr_processes"]

    if nr_processes > 1:
        with tqdm(total=n_remaining) as pbar:
            pool = mp.Pool(nr_processes, process_init)
            for i, _ in enumerate(pool.imap(process, fnames)):
                pbar.update()
    else:
        process_init()
        for fname in tqdm(fnames):
            process(fname)
    return


if __name__ == "__main__":
    process_all_transcripts(force_rebuild=False, max_nr=None)
