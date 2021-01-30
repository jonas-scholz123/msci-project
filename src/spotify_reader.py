import json
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

path = "../transcripts/spotify_transcripts/podcasts-transcripts-6to7/spotify-podcasts-2020/podcasts-transcripts/6/0/show_60aEckwTYs8xCEpsAasV0o/3NHTGeZoLLIfoHnlwtOu6w.json"

#multi speakers:
#path = "../transcripts/spotify_transcripts/podcasts-transcripts-6to7/spotify-podcasts-2020/podcasts-transcripts/6/M/show_6me2wZOYKSbBIl3x5VVIP7/2q8NzBl4l18H4Hrxh47bx2.json"


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

utterances, speakers_and_times = extract_json_transcript(path)

write_transcript("../transcripts/person1_person2.txt",
    utterances, speakers_and_times)
