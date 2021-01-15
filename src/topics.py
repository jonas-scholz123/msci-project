import tensorflow_hub as hub
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils import load_one_transcript, load_all_transcripts, load_pretrained_glove
from predictDA import get_all_annotated_transcripts
from tqdm import tqdm
from gensim.models import word2vec
from analyse_transcripts import enhance_transcript_df
import pandas as pd
import config

pd.options.display.width = 0
pd.options.display.max_rows = None

#%%
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))

def make_similarity_matrix(labels, features):
    return np.inner(features, features)

def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    #fontsize_pt = plt.rcParams['ytick.labelsize']
    #dpi = 72.27
    #matrix_height_pt = fontsize_pt * len(features)
    #matrix_height_in = matrix_height_pt / dpi
    #print("plotting matrix of size ", matrix_height_in)
    plt.figure(figsize = (20, 20))

    sns.set(font_scale=1.2)
    g = sns.clustermap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.ax_heatmap.set_title("Clustered Semantic Textual Similarity")
    plt.savefig(config.paths["figures"] + "similarity.pdf")
#%%
if __name__ == "__main__":

    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
    from nltk.corpus.reader import NOUN
    from nltk.stem.wordnet import WordNetLemmatizer
    from polyglot.text import Text
    #%%
    #load glove:
    glove = load_pretrained_glove("../embeddings/glove.840B.300d.txt")

    #%%
    transcript_dfs = get_all_annotated_transcripts(force_rebuild=False)
    transcript_dfs[2]
    for transcript_df in transcript_dfs:
        enhance_transcript_df(transcript_df)
        transcript_df["utterance"] = transcript_df["utterance"].str.replace(" ' ", "'")
        transcript_df["utterance"] = transcript_df["utterance"].str.replace(" ’ ", "’")
    tdf = transcript_dfs[2]
    filler_das = ['Appreciation', 'Agree/Accept', 'Acknowledge (Backchannel)',
        'Repeat-phrase', 'Yes answers', 'Response Acknowledgement',
        'Affirmative non-yes answers', 'Backchannel in question form',
        'Negative non-no answers', 'Uninterpretable', 'Signal-non-understanding',
        'Hold before answer/agreement', 'Action-directive', 'Thanking']

    text = " ".join(tdf["utterance"])

#    # extract POS without any context and store in dict
#    # we do this because context of spoken conversation leads to a lot of
#    # falsely identified nouns
#    tokens = nltk.word_tokenize(text)
#    unique_tokens = set(tokens)
#    tok2pos = [nltk.pos_tag([tok])[0] for tok in unique_tokens]
#    tok2pos = {tok:pos for tok, pos in tok2pos}

    def add_topic_words(tdf):
        #TODO: add documentation for how to install NER/POS libraries
        #TODO: clean this up, put everything in smaller functions etc
        '''
        Extracts and adds (in a new column) words that represent the topic of a
        given sentence. This will later be used for segmentation.

        Uses POS (part of speech) tagging, NER (named entity recognition) and
        wordNet to figure out topic words
        =======================================================================
        PARAMS:
            pd.DataFrame tdf: transcript dataframe
        RETURNS:
            pd.DataFrame tdf: annotated dataframe
        '''

        #singular person words that were wrongly classified as topic words, are filtered out manually
        manual_filter_words = ["get", "thing"]

        stop_words = set(stopwords.words('english'))
        Lem = WordNetLemmatizer()

        sentence_keywords = []
        for entry in tdf["utterance"]:
            printable = False
            keywords = []
            tokens = nltk.word_tokenize(entry)
            tokens = [t for t in tokens if t not in stop_words]
            #TODO: implement two word combinations
            #TODO: Analogies destroy topic, maybe can't do anything about that
            for word, pos in nltk.pos_tag(tokens):
                #sometimes 1 letter words are misclassified as nouns and added
                if printable and word == "name":
                    print("name is identified as ", pos)
                    print("pos in NN: ", pos in ["NN", "NNP", "NNS", "CD"])
                    print("len(word) > 1: ", len(word) > 1)
                    print("not in stopwords", word not in stop_words)
                if (pos in ["NN", "NNP", "NNS", "CD"]
                    and len(word) > 1
                    and word not in stop_words):
                    if printable:
                        print('getting here')
                    #checks if it can be a noun
                    if pos in ["NN", "NNS"]:
                        #wordnet does not have plurals, roughly remove plurals
                        word = Lem.lemmatize(word)
                        for synset in wn.synsets(word, NOUN):
                            if printable:
                                print(synset)
                            if synset.name().split('.')[0] == word.lower():
                                #print("adding word1: ", word)
                                if printable:
                                    print(word, "Case 1")
                                keywords.append(word)
                                break
                    else:
                        keywords.append(word)

            # for two word combinations, such as "neural net", check if its in synset
            for words in zip(tokens[::2], tokens[1::2]):
                combined = "_".join(words)
                combined = Lem.lemmatize(combined)
                if len(wn.synsets(combined, NOUN)) > 0:
                    if printable:
                        print(combined, "Case 2")
                    keywords.append(combined)

            # if recognised as a named entity, add to key words
            text = Text(" ".join([w for w in entry.split(" ") if w not in stop_words]))
            text.language = "en" #set language, otherwise sometimes gets confused
            for e in text.entities:
                word = "_".join(text.words[e.start:e.end])
                if not ("'" in word or "’" in word): #weirdly, words with apostrophes mess with NER
                    if printable:
                        print(word, "Case 3")
                    keywords.append(word)

            keywords = set(keywords) #unique words
            to_remove = []
            for w in keywords:
                if Lem.lemmatize(w) in manual_filter_words:
                    to_remove.append(w)
            for w in to_remove: keywords.remove(w) #remove manually filtered words
            sentence_keywords.append(list(keywords))

        tdf["key_words"] = sentence_keywords
        tdf.loc[~tdf["key_words"].astype(bool), "key_words"] = None
        tdf.loc[tdf["da_label"].isin(filler_das), "key_words"] = None

        # temporarily set all entries without keywords to have the same keywords as previous:
        current_keywords = ["NO KEYWORDS YET"]
        for i, keywords in enumerate(tdf["key_words"]):
            if keywords:
                current_keywords = keywords
            else:
                tdf.at[i, "key_words"] = current_keywords
        return tdf
