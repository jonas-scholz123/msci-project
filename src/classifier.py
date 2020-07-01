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
import spacy

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import spatial
import sister

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class Classifier():

    def __init__(self):

        #Models
        self.sia = SentimentIntensityAnalyzer()
        self.word_type_model = spacy.load("en_core_web_sm")
        self.embedder = sister.MeanEmbedding(lang="en")

        #State parameters
        self.emotion_classified = False

        #Minimum words in sentence to accurately assess topic
        self.minwords = 10

        return

    def classify(self):
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

    def classify_emotion(self, pos_boundary = 0.05, neg_boundary = -0.05):
        # TODO: ideal boundaries
        emotions = []
        for sentence in self.discussion["text"]:
            ss = self.sia.polarity_scores(sentence)
            if ss["compound"] > pos_boundary:
                emotions.append("pos")
            elif ss["compound"] < neg_boundary:
                emotions.append("neg")
            else:
                emotions.append("neutral")

        self.discussion["emotion"] = emotions
        self.emotion_classified = True

        return self.discussion

    def classify_disagreement(self):
        # TODO: Corpus based model?

        # If emotion not yet classified, do that first:
        if not self.emotion_classified:
            self.classify_emotion()
        #stores disagreements
        disagreements = []

        #initial emotion
        prev_negative = False

        for emotion in self.discussion["emotion"].values:
            has_changed = (emotion == "pos" and prev_negative or emotion == "neg" and not prev_negative)
            disagreements.append(has_changed)
            if has_changed:
                prev_negative = not prev_negative

        self.discussion["emo_disagreement"] = disagreements
        return self.discussion

    def extract_topic_words(self, s):
        doc = self.word_type_model(s)
        topic_words = [str(tok.text) for tok in doc if tok.pos_ in ["NOUN"]]
        return " ".join(topic_words)

    def classify_topical_similarity(self, nr_topics = 3):

        similarities = []

        topic_words = self.discussion["topic_words"]

        topic_aggregator = list(self.discussion["topic_words"][:nr_topics].values)
        prev_topic = self.discussion["topic_words"][1]

        for topic in topic_words.values:
            if topic == prev_topic:
                similarities.append(1)
                continue

            new_topic_vector = self.embedder(topic)
            old_topic_vectors = [self.embedder(topic) for topic in topic_aggregator]

            all_similarities = [1 - spatial.distance.cosine(old, new_topic_vector) for old in old_topic_vectors]

            similarity = max(all_similarities)
            similarities.append(similarity)

            topic_aggregator.pop(0)
            topic_aggregator.append(topic)

        self.discussion["similarity"] = similarities
        self.discussion["topic_change"] = (pd.Series(similarities) < 0.4)

    def iterate_topic_words(self):
        sentences = self.discussion["text"]

        prev_topic_words = ""

        topic_words = []

        for s in sentences:
            # filter short filler sentences
            if len(s.split()) < self.minwords:
                topic_words.append(prev_topic_words)

            else:
                topic = self.extract_topic_words(s)
                if topic:
                    prev_topic_words = topic
                topic_words.append(prev_topic_words)

        self.discussion["topic_words"] = topic_words





#%%

if __name__ == "__main__":

    fpath = "../data/podcasts/joe_rogan_elon_musk_may_2020.txt"

    classifier = Classifier()

    classifier.read_file(fpath)

    classifier.classify_emotion()

    classifier.classify_disagreement()

    classifier.iterate_topic_words()

    classifier.classify_topical_similarity()

    classifier.discussion[classifier.discussion["similarity"] < 0.3]
    #%%
    index_high_sim = classifier.discussion[classifier.discussion["similarity_to_next"] > 0.8].index
    index_next = index_high_sim + 1

    combined_high_indices = sorted(list(set(index_high_sim.union(index_next))))
    combined_high_indices = combined_high_indices[:-1] #last index too high

    classifier.discussion.iloc[combined_high_indices].head(50)

#%%
    index_low_sim = classifier.discussion[classifier.discussion["similarity"] < 0.4].index
    index_next_low = index_low_sim - 1
    index_next_low2 = index_low_sim - 2

    combined_low_indices = index_low_sim.union(index_next_low)
    combined_low_indices = sorted(list(set(combined_low_indices.union(index_next_low2))))
    combined_low_indices = combined_low_indices[:-1] #last index too high

    classifier.discussion.iloc[combined_low_indices].head(50)

    #%%
