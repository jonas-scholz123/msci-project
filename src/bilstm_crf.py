from keras import Sequential
from keras.models import Model, Input
from keras.layers import (
    LSTM,
    Embedding,
    Dense,
    TimeDistributed,
    Dropout,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    Flatten,
    AveragePooling1D,
)
from keras.optimizers import Adam, schedules
from tf2crf import CRF

import config


def get_bilstm_crf_model(embedding_matrix, n_tags, verbose=False):
    print("loading model...")
    max_nr_utterances = config.data["max_nr_utterances"]
    max_nr_words = config.data["max_nr_words"]
    dropout_rate = config.model["dropout_rate"]
    nr_lstm_cells = config.model["nr_lstm_cells"]
    init_lr = config.model["init_lr"]
    decay_steps = config.model["decay_steps"]
    decay_rate = config.model["decay_rate"]

    embedding_layer = Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=max_nr_words,
        trainable=True,
    )

    utterance_encoder = Sequential()
    utterance_encoder.add(embedding_layer)

    # average pooling
    # utterance_encoder.add(Bidirectional(LSTM(300, return_sequences=True)))
    # utterance_encoder.add(AveragePooling1D(max_nr_words))

    # last pooling
    utterance_encoder.add(Bidirectional(LSTM(nr_lstm_cells)))
    utterance_encoder.add(Dropout(dropout_rate))
    # utterance_encoder.add(Flatten())
    if verbose:
        utterance_encoder.summary()

    crf = CRF(dtype="float32")

    x_input = Input(shape=(max_nr_utterances, max_nr_words))
    h = TimeDistributed(utterance_encoder)(x_input)
    h = Bidirectional(LSTM(nr_lstm_cells, return_sequences=True))(h)
    h = Dropout(dropout_rate)(h)
    h = Dense(n_tags, activation=None)(h)
    crf_output = crf(h)
    model = Model(x_input, crf_output)

    if verbose:
        model.summary()
    lr_schedule = schedules.ExponentialDecay(
        init_lr, decay_steps=decay_rate, decay_rate=decay_rate
    )

    optimizer = Adam(learning_rate=lr_schedule)  # TODO: check if used?
    model.compile("adam", loss=crf.loss, metrics=[crf.accuracy])
    print("Done!")
    return model
