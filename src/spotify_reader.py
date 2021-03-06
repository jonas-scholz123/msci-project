import json
import os
from datetime import timedelta
import config

def json_to_transcript(in_path, out_path):
    transcript_data = extract_json_transcript(in_path)
    if transcript_data:
        utterances, speakers_and_times = transcript_data
        write_transcript(out_path, utterances, speakers_and_times)
        return True
    else:
        return False


def extract_json_transcript(path):
    with open(path) as f:
        transcript_json = json.load(f)

    max_st = 0
    utterances = []
    speakers_and_times = []
    for entry in transcript_json["results"]:
        entry = entry['alternatives'][0]
        if "words" not in entry.keys():
            continue
        word_entries = entry["words"]
        entry
        words = []
        time = word_entries[0]["startTime"]
        for we in word_entries:
            if "speakerTag" in we.keys():
                speaker_tag = int(we["speakerTag"])
                max_st = max(max_st, speaker_tag)
                word = we["word"]
                words.append(word)
                if "." in word or "?" in word or "!" in word:
                    utterances.append(" ".join(words) + "\n")
                    # 35.00s -> cut off s i.e. [:-1], turn into float, then turn
                    # into desired format (hh:mm:ss)
                    seconds = int(float(time[:-1]))
                    timestring = '(0' + str(timedelta(seconds=seconds)) + ')'
                    speaker_timestring = ("Speaker " + str(speaker_tag) + ": "
                                          + timestring + "\n")
                    speakers_and_times.append(speaker_timestring)
                    time = we["startTime"]
                    words = []
    if max_st == 0:
        return
    return utterances, speakers_and_times


def write_transcript(path, utterances, speakers_and_times):
    whitespaces = ["\n"] * len(utterances)
    lines = list(zip(speakers_and_times, utterances, whitespaces))
    lines = [list(l) for l in lines]
    lines = sum(lines, [])
    with open(path, "w") as f:
        f.writelines(lines)
    return


def extract_spotify(max_nr=None):

    json_paths = []

    def extract_jsons(root):
        for path in os.listdir(root):
            full_path = root + path
            if os.path.isdir(full_path):
                extract_jsons(full_path + "/")
            else:
                json_paths.append(full_path)

    root = config.paths["spotify_root"]
    extract_jsons(root)

    counter = 0
    for jp in json_paths:
        hash = jp.split("/")[-1].split(".")[0]
        new_fn = "spotify_" + hash
        new_fp = config.paths["transcripts"] + new_fn + ".txt"

        if os.path.exists(new_fp):
            continue

        utterances, speakers_and_times = extract_json_transcript(jp)

        if len(utterances) <= config.data["max_nr_utterances"]:
            #too short is bad
            continue

        write_transcript(new_fp, utterances, speakers_and_times)
        counter += 1

        if max_nr is not None and counter >= max_nr:
            break

def remove_all_spotify():
    dir = config.paths["transcripts"]

    for fp in os.listdir(dir):
        if fp.startswith("spotify"):
            os.remove(dir + fp)


extract_spotify(max_nr=50)
