data = dict(
    max_nr_utterances = 100,
    max_nr_words =  107, #max out of swda, mrda corpora
)

model = dict(
    dropout_rate = 0.2, #fraction of neurons disabled during training
    nr_lstm_cells = 300, #drives complexity of lstm layers
    init_lr = 1, # initial learning rate
    decay_steps = 1000, #after decay_steps, lr is reduced by a factor of decay_rate
    decay_rate = 0.5,
    batch_size = 5, #number of samples processed in parallel
    validation_fraction = 0.1,
    test_fraction = 0.1,
)

corpus = dict(
    corpus = 'swda', # 'mrda' or 'swda'
    detail_level = 0, #for mrda, 0, 1, 2 from lowest detail to highest
)

paths = dict(
    transcripts = '../transcripts/',
    transcript_dfs = '../processed_transcripts/',
    figures = '../figures/',
)

topics = dict(

    #singular person words that were wrongly classified as topic words, are filtered out manually
    #Note some words, like "time" are unlikely to be the topic of conversation
    #and more used in sentences such as "last time you did X" or "I remember"
    #a time when..."
    # other words are just nouns that are filler words such as "guy, thing, man"
    # some words are mostly used in combination with "of" such as "the process of"
    # "kind of" "sort of " are filtered
    #Don't is classified as Don (name)
    manual_filter_words = set(["get", "thing", "man", "go", "okay", "“", "Don",
                               "nobody", "are", "wow", "woah", "whoa", "perfect",
                               "way", "guy", "stuff", "day", "iteration", "bit",
                               "inch", "meter", "millimeter", "centimeter", "yard",
                               "kilometer", "mile", "foot", "time",
                               "Does", "process", "lot", "kind", "sort", "sometimes",
                               "somewhere", "something"]),

    filler_das = set(['Appreciation', 'Agree/Accept', 'Acknowledge (Backchannel)',
        'Repeat-phrase', 'Yes answers', 'Response Acknowledgement',
        'Affirmative non-yes answers', 'Backchannel in question form',
        'Negative non-no answers', 'Uninterpretable', 'Signal-non-understanding',
        'Hold before answer/agreement', 'Action-directive', 'Thanking']),

    max_gap = 10, #max number of sentences between two topic_word matches for it to no longer be one topic
    min_sim = 0.65,
)

# disable huge number of information prints by tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
