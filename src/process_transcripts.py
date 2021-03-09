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
    total_number = len(fnames)

    # counter = 0
    mp.set_start_method("spawn")  # allows CUDA multiprocessing

    print("total nr transcripts: ", len(fnames))
    already_processed = set(os.listdir(config.paths["tdfs"]))
    # pkl and csv files
    nr_already = len(already_processed) // 2
    print("already processed: ", nr_already)

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
        with tqdm(total=total_number, initial=nr_already) as pbar:
            pool = mp.Pool(nr_processes, process_init)
            for i, _ in enumerate(pool.imap(process, fnames)):
                pbar.update()
    else:
        process_init()
        for fname in tqdm(fnames):
            process(fname)
    return


def split_into_subfolders(root, extensions, files_per_dir):
    root_content = os.listdir(root)
    directories = [c for c in root_content if os.path.isdir(root + c)]
    files_root = [f for f in root_content if os.path.isfile(root + f)]

    filenames = list(set([f.split(".")[0] for f in files_root]))

    if directories:
        dir_counter = int(max(directories))
        current_directory = root + max(directories) + "/"
        remaining_space = files_per_dir - len(os.listdir(current_directory))
    else:
        dir_counter = 1
        remaining_space = files_per_dir
        current_directory = root + str(dir_counter) + "/"
        os.mkdir(current_directory)

    for i, fname in tqdm(enumerate(filenames)):
        if not fname:
            continue

        if remaining_space <= 0:
            dir_counter += 1
            current_directory = root + str(dir_counter) + "/"

            if not os.path.exists(current_directory):
                os.mkdir(current_directory)
                remaining_space = files_per_dir
            else:
                remaining_space = (
                    files_per_dir - len(os.listdir(current_directory)) // 2
                )
        for extension in extensions:
            os.rename(root + fname + extension, current_directory + fname + extension)
            remaining_space -= 1


def merge_folders_into_root():
    root = config.paths["tdfs"]

    for dir in tqdm(os.listdir(root)):
        if not os.path.isdir(root + dir):
            continue
        dir = root + dir + "/"
        for file in os.listdir(dir):
            os.rename(dir + file, root + file)


if __name__ == "__main__":
    # process_all_transcripts(force_rebuild=False, max_nr=None)
    # split_into_subfolders(config.paths["tdfs"], [".csv", ".pkl"], 200)
    # split_into_subfolders(config.paths["transcripts"], [".txt"], 100)

    fname = "joe_rogan_jack_dorsey.txt"

    fname = fname.split(".")[0]
    transcript = load_one_transcript(config.paths["transcripts"] + fname + ".txt")
    # only process unprocessed transcripts
    dac = DA_classifier()
    te = TopicExtractor()
    tdf = dac.make_annotated_transcript(transcript)
    tdf = enhance_tdf(tdf)
    tdf = te.process(tdf)
    # csvs are human readable
    tdf.to_csv("../processed_transcripts/" + fname + ".csv", index=False)
    # pickle preserves objects such as lists, sets
    tdf.to_pickle("../processed_transcripts/" + fname + ".pkl")
