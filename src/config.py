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
    figures = '../figures',
)

# disable huge number of information prints by tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
