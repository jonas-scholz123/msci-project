from datetime import datetime
from tqdm import tqdm
import os
import numpy as np

import config
from utils import load_one_transcript, load_all_transcripts
from predictDA import make_annotated_transcript
from topics import TopicExtractor

def timestamp_to_datetime(timestamp):
    return datetime.strptime(timestamp, "%H:%M:%S")

def get_fractional_time(current_timestamp, biggest_time):
    try:
        current_seconds = (biggest_time - timestamp_to_datetime(current_timestamp)).total_seconds()
        total_seconds = (biggest_time - timestamp_to_datetime("00:00:00")).total_seconds()
        return 1 - current_seconds/total_seconds
    except:
        return current_timestamp


def enhance_transcript_df(df):
    # calc relative time
    biggest_time = timestamp_to_datetime(df.loc[df.shape[0] - 1, "timestamp"])
    df["relative_time"] = (
        df["timestamp"].apply(get_fractional_time, args=(biggest_time, )))

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
        lengths[i] = (timestamp_to_datetime(next_timestamp)
                      - timestamp_to_datetime(timestamp)).total_seconds()
    df["utterance_length"] = lengths
    return df


def process_all_transcripts(force_rebuild=False):
    te = TopicExtractor()
    transcripts, fnames = load_all_transcripts(return_fnames=True)

    for transcript, fname in tqdm(zip(transcripts, fnames)):
        if (os.path.exists("../processed_transcripts/" + fname + ".csv")
                and not force_rebuild):
            continue
        # only process unprocessed transcripts
        tdf = make_annotated_transcript(transcript)
        tdf = enhance_transcript_df(tdf)
        tdf = te.process(tdf)
        # csvs are human readable
        tdf.to_csv("../processed_transcripts/" + fname + ".csv",
                   index=False)
        # pickle preserves objects such as lists, sets
        tdf.to_pickle("../processed_transcripts/" + fname + ".pkl")
    return


if __name__ == "__main__":
    process_all_transcripts(force_rebuild=True)
    # import pandas as pd
    # tdf = pd.read_pickle("../processed_transcripts/sam_harris_nicholas_christakis.pkl")

    # pd.options.display.width = 0
    # pd.options.display.max_rows = None

    # tdf
    # import numpy as np
    # all_topics = set()
    # for topics in tdf["topics"]:
    #     if type(topics) != list:
    #         continue
    #     for t in topics:
    #         all_topics.add(t)
    # import numpy as np
    # df = pd.DataFrame([0, 1, 10, 9, 8])
    # df["topics"] = np.empty((len(df), 0)).tolist() #set topics to empty list
#
    # df["topics"] = [list() for _ in range(len(df))]
#
    # df.loc[:, "topics"] = df.loc[:, "topics"].apply(lambda x: x + ["lol"])
