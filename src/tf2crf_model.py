from keras import Sequential
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from keras.optimizers import Adam, schedules
from tf2crf import CRF

def get_tf2crf_model(embedding_matrix, max_nr_utterances, max_nr_words, n_tags):
    dropout_rate = 0.2
    nr_lstm_cells = 300
    init_lr = 1

    embedding_layer = Embedding(
                        embedding_matrix.shape[0],
                        embedding_matrix.shape[1],
                        weights=[embedding_matrix],
                        input_length=max_nr_words,
                        trainable=True)

    utterance_encoder = Sequential()
    utterance_encoder.add(embedding_layer)

    # average pooling
    #utterance_encoder.add(Bidirectional(LSTM(300, return_sequences=True)))
    #utterance_encoder.add(AveragePooling1D(max_nr_words))

    #last pooling
    utterance_encoder.add(Bidirectional(LSTM(nr_lstm_cells)))

    utterance_encoder.add(Dropout(dropout_rate))
    utterance_encoder.add(Flatten()) #TODO does this do anything
    utterance_encoder.summary()

    crf = CRF(dtype='float32')

    x_input = Input(shape = (max_nr_utterances, max_nr_words))
    h = TimeDistributed(utterance_encoder)(x_input)
    h = Bidirectional(LSTM(nr_lstm_cells, return_sequences=True))(h)
    h = Dropout(dropout_rate)(h)
    h = Dense(n_tags, activation=None)(h)
    crf_output = crf(h)
    model = Model(x_input, crf_output)
    #h = Dense(n_tags, activation="softmax")(h)
    #model = Model(x_input, h)

    model.summary()
    lr_schedule = schedules.ExponentialDecay(init_lr, decay_steps = 1000, decay_rate = 0.5)
    optimizer = Adam(learning_rate = lr_schedule)
    model.compile("adam", loss=crf.loss, metrics=[crf.accuracy])
    return model
