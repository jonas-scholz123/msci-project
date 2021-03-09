from nltk.metrics.segmentation import windowdiff, pk
import numpy as np
import pandas as pd
from topics import TopicExtractor
from utils import load_one_transcript
from collections import defaultdict


def bound2seg(bounds, max_i):
    return ["0" if i not in bounds else "1" for i in range(max_i)]


def get_topic_ranges(tdf):
    trs = defaultdict(list)

    for i, topics in tdf["topics"].iteritems():
        for topic in topics:
            existing_range = trs.get(frozenset(topic))
            if (
                existing_range is not None
                and i >= existing_range[-1][0]
                and i <= existing_range[-1][1]
            ):
                continue
            for j, next_topics in tdf.loc[i + 1 :, "topics"].iteritems():
                if topic in next_topics:
                    continue
                trs[frozenset(topic)].append((i, j - 1))
                break
    return trs


def get_geek_bounds(topic_ranges, max_i):
    # sort topics by tr, so that bigger ones placed at the bottom

    tr_items = [(topic, tr) for topic, trs in topic_ranges.items() for tr in trs]

    tr_items = sorted(tr_items, key=lambda x: x[1][1] - x[1][0], reverse=True)
    min_topic_length = 5

    covered = np.zeros(max_i)
    geek_bounds = []
    for item in tr_items:
        tr = item[1]
        if tr[1] - tr[0] < min_topic_length or any(covered[tr[0] : tr[1]] == 1):
            continue
        covered[tr[0] : tr[1]] = 1
        geek_bounds.append(tr[0])
    return geek_bounds


if __name__ == "__main__":
    # bayes-seg result:

    bayes_boundaries = [1, 72, 102, 103, 104, 105, 130, 131, 144, 158, 234, 235, 248]
    perfect_boundaries = [4, 21, 30, 49, 72, 104, 127, 131, 146, 169, 220, 225, 237]
    # bayes_boundaries = [13, 14, 15, 16, 21, 69, 106, 170, 171, 172, 222, 233, 248]

    max_i = max(sum([bayes_boundaries, perfect_boundaries], []))

    bayes_seg = bound2seg(bayes_boundaries, max_i)
    perfect_seg = bound2seg(perfect_boundaries, max_i)

    k = int(max_i / (2 * (len(perfect_boundaries))))  # halved avg segment size

    print("wd bayes: ", windowdiff(bayes_seg, perfect_seg, k=k))

    tdf = pd.read_pickle("../processed_transcripts/joe_rogan_elon_musk.pkl")[0:max_i]

    te = TopicExtractor()

    topic_ranges = get_topic_ranges(tdf)

    geek_bounds = get_geek_bounds(topic_ranges, max_i)
    geek_seg = bound2seg(geek_bounds, max_i)

    print("wd GEEK: ", windowdiff(geek_seg, perfect_seg, k=k))
