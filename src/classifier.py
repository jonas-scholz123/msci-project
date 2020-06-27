'''
Class to parse bodies of text-based discussions into phrases that can then be
classified as words of agreement, disagreement, or topics.

This initial implementation is a very basic, algorithmic approach used only as
a proof of concept.

TODO: Implement agreement/disagreement classifier based on this paper:
https://www.aclweb.org/anthology/W14-2617.pdf

'''

import pandas as pd
import re
import numpy as np

class Classifier():

    def __init__(self):
        return

    def read_file(self, fpath, medium = "podcast"):
        self.medium = medium

        if self.medium == "podcast":
            #open file
            with open(fpath, "r") as f:
                podcast = f.readlines()

            #organise content
            title_rows = podcast[::3]
            speakers = [row.split(":")[0] for row in title_rows]
            times = [":".join(re.findall("[0-9]+", row)) for row in title_rows]
            text = podcast[1::3]
            text = [t[0:-1] for t in text]

            #store content in dataframe
            podcast_df = pd.DataFrame([speakers, times, text]).transpose()
            podcast_df.columns = ["speaker", "time", "text"]

            self.discussion = podcast_df
            return




fpath = "../data/podcasts/joe_rogan_elon_musk_may_2020.txt"


classifier = Classifier()

classifier.read_file(fpath)

classifier.discussion

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

emotions = []
for sentence in classifier.discussion["text"]:
    ss = sia.polarity_scores(sentence)
    if ss["compound"] > 0.05:
        emotions.append("pos")
    elif ss["compound"] < -0.05:
        emotions.append("neg")
    else:
        emotions.append("neutral")

classifier.discussion["emotion"] = emotions

classifier.discussion[classifier.discussion["emotion"] == "neg"]
