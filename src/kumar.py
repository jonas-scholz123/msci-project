import numpy as np
import os
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import pickle
from keras.preprocessing.text import Tokenizer

from utils import get_embedding_matrix, pad_nested_sequences, split_into_chunks, chunk
from reformat_training_data import read_mrda_training_data
from mappings import get_id2tag

# helper methods
def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, labels, batch_size):
    data_size = len(data)
    start_index = 0

    num_batches_per_epoch = int((len(data) + batch_size - 1) / batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index: end_index], labels[start_index: end_index]


def select(parameters, length):
    """Select the last valid time step output as the sentence embedding
    :params parameters: [batch, seq_len, hidden_dims]
    :params length: [batch]
    :Returns : [batch, hidden_dims]
    """
    shape = tf.shape(parameters)
    idx = tf.range(shape[0])
    idx = tf.stack([idx, length - 1], axis = 1)
    return tf.gather_nd(parameters, idx)
#%%

max_nr_utterances = 100
max_nr_words = 200

corpus = 'mrda'
detail_level = 0

conversations, labels = read_mrda_training_data(detail_level)

all_utterances = sum(conversations, [])

#open tag2id mapping for labels and create inverse
id2tag = get_id2tag(corpus, detail_level)
tag2id = {t : id for id, t in id2tag.items()}
n_tags = len(tag2id.keys())

conversations = chunk(conversations, max_nr_utterances)
labels = chunk(labels, max_nr_utterances)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_utterances)
word2id = tokenizer.word_index

conversation_sequences = [tokenizer.texts_to_sequences(c) for c in conversations]

# Toy data from original source
#toy_data = [[[1,2,3,4],[1,2,3],[2,3,5]],[[1,0], [4]],[[1,2,8,4],[1,1,3],[2,3,9,1,3,1,9]], [[1,2,3,4,5,7,8,9],[9,1,2,4],[8,9,0,1,2]],[[1,2,4,3,2,3],[9,8,7,5,5,5,5,5,5,5,5]],[[1,2,3,4,5,6,9],[9,1,0,0,2,4,6,5,4]],[[1,2,3,4,5,6,7,8,9],[9,1,2,4],[8,9,0,1,2]],[[1]] , [[1,2,11,2,3,2,1,1,3,4,4], [6,5,3,2,1,1,4,5,6,7], [9,8,1], [1,6,4,3,5,7,8], [0,9,2,4,6,2,4,6], [5,2,2,5,6,7,3,7,2,2,1], [0,0,0,1,2,7,5,3,7,5,3,6], [1,3,6,6,3,3,3,5,6,7,2,4,2,1], [1,2,4,5,2,3,1,5,1,1,2], [9,0,1,0,0,1,3,3,5,3,2], [0,9,2,3,0,2,1,5,5,6], [9,0,0,1,4,2,4,10,13,11,12], [0,0,1,2,3,0,1,1,0,1,2], [0,0,1,3,1,12,13,3,12,3], [0,9,1,2,3,4,1,3,2]]]
#toy_labels = [[1,2,1],[0, 3],[1,2,1],[1,0,2], [2,1], [1,1], [2,1,2], [4], [0,1,2,0,2,4,2,1,0,1,0,2,1,2,0]]

# Split data
n_samples = len(conversation_sequences)
valid_frac = 0.1
test_frac = 0.1

valid_index = int(n_samples * (1 - valid_frac - test_frac))
test_index = int(n_samples * (1 - test_frac))


train_data = conversation_sequences[:valid_index]
train_labels = labels[:valid_index]

test_data = conversation_sequences[valid_index:test_index]
test_labels = labels[valid_index:test_index]

dev_data = conversation_sequences[valid_index:]
dev_labels = labels[valid_index:]

#%%

# Global variables
hidden_size_lstm_1 = 200
hidden_size_lstm_2 = 300
lr = 1
tags = n_tags
word_dim = 300
proj1 = 200
proj2 = 100
words = 20001
batchSize = 10
log_dir = "train"
model_dir = "DAModel"
model_name = "ckpt"


# Dialogue Act Recognition Model
# Architecture: dataset --> embedding --> utterance-level bi-LSTM --> conversation-level bi-LSTM --> CRF --> one label per utterance
class DAModel():
    def __init__(self):
        with tf.variable_scope("placeholder"):
            self.dialogue_lengths = tf.placeholder(tf.int32, shape=[None], name="dialogue_lengths")
            self.word_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="word_ids")
            self.utterance_lengths = tf.placeholder(tf.int32, shape=[None, None], name="utterance_lengths")
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
            self.clip = tf.placeholder(tf.float32, shape=[], name='clip')

        with tf.variable_scope("embeddings"):
            _word_embeddings = tf.get_variable(
                name = "_word_embeddings",
                dtype = tf.float32,
                shape = [words, word_dim],
                #initializer = tf.random_uniform_initializer()
                initializer = tf.constant_initializer(get_embedding_matrix("", None))
            )
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings, 0.8)

        with tf.variable_scope("utterance_encoder"):
            s = tf.shape(self.word_embeddings)
            batch_size = s[0] * s[1]
            time_step = s[-2]

            word_embeddings = tf.reshape(self.word_embeddings, [batch_size, time_step, word_dim])
            length = tf.reshape(self.utterance_lengths, [batch_size])

            fw = tf.nn.rnn_cell.LSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple=True)
            bw = tf.nn.rnn_cell.LSTMCell(hidden_size_lstm_1, forget_bias=0.8, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, word_embeddings,sequence_length=length, dtype=tf.float32)
            output = tf.concat(output, axis=-1) # [batch_size, time_step, dim]

            # Select the last valid time step output as the utterance embedding,
            # this method is more concise than TensorArray with while_loop
            output = select(output, length) # [batch_size, dim]
            output = tf.reshape(output, (s[0], s[1], 2 * hidden_size_lstm_1))
            output = tf.nn.dropout(output, 0.8)

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_lstm_2, forget_bias=0.8, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_lstm_2, forget_bias=0.8, state_is_tuple=True)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length=self.dialogue_lengths, dtype=tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            outputs = tf.nn.dropout(outputs, 0.8)

        with tf.variable_scope("proj1"):
            output = tf.reshape(outputs, [-1, 2 * hidden_size_lstm_2])
            W = tf.get_variable("W", dtype=tf.float32, shape=[2 * hidden_size_lstm_2, proj1], initializer=tf.keras.initializers.glorot_uniform())
            b = tf.get_variable("b", dtype=tf.float32, shape=[proj1], initializer=tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(output, W) + b)

        with tf.variable_scope("proj2"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[proj1, proj2], initializer=tf.keras.initializers.glorot_uniform())
            b = tf.get_variable("b", dtype=tf.float32, shape =[proj2], initializer=tf.zeros_initializer())
            output = tf.nn.relu(tf.matmul(output, W) + b)

        with tf.variable_scope("logits"):
            nstep = tf.shape(outputs)[1]
            W = tf.get_variable("W", dtype=tf.float32, shape=[proj2, tags], initializer=tf.random_uniform_initializer())
            b = tf.get_variable("b", dtype=tf.float32, shape =[tags], initializer=tf.zeros_initializer())
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nstep, tags])

        with tf.variable_scope("loss"):
            transition_params = tf.get_variable("transitions", dtype=tf.float32, shape=[tags, tags])
            sequence_scores = tfa.text.crf_sequence_score(self.logits, self.labels, self.dialogue_lengths, transition_params)
            log_norm = tfa.text.crf_log_norm(self.logits, self.dialogue_lengths, transition_params)
            log_likelihood = sequence_scores - log_norm
            self.trans_params = transition_params
            self.loss = tf.reduce_mean(-log_likelihood) + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

        with tf.variable_scope("viterbi_decode"):
            viterbi_sequence, _ = tfa.text.crf_decode(self.logits, self.trans_params,  self.dialogue_lengths)
            batch_size = tf.shape(self.dialogue_lengths)[0]
            output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            def body(time, output_ta_1):
                length = self.dialogue_lengths[time]
                vcode = viterbi_sequence[time][:length]
                true_labs = self.labels[time][:length]
                accurate = tf.reduce_sum(tf.cast(tf.equal(vcode, true_labs), tf.float32))

                output_ta_1 = output_ta_1.write(time, accurate)
                return time + 1, output_ta_1


            def condition(time, output_ta_1):
                return time < batch_size


            i = 0
            [time, output_ta] = tf.while_loop(condition, body, loop_vars=[i, output_ta])
            output_ta = output_ta.stack()
            accuracy = tf.reduce_sum(output_ta)
            self.accuracy = accuracy / tf.reduce_sum(tf.cast(self.dialogue_lengths, tf.float32))

        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdagradOptimizer(lr)
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))


def main():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=config) as sess:
        model = DAModel()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if os.path.exists("../trained_model/kumar/model_glove.index"):
            print("MODEL FOUND, LOADING CHECKPOINT")
            saver.restore(sess, "../trained_model/kumar/model_glove")
        writer = tf.summary.FileWriter("../logs/kumar", sess.graph)

        clip = 2
        counter = 0

        for epoch in range(100):
            print("BEGINNING EPOCH ", epoch + 1, "/", 100)
            for train_batch_dialogues, train_batch_labels in minibatches(train_data, train_labels, batchSize):

                _, train_batch_dialogue_lengths = pad_sequences(train_batch_dialogues, 0)
                train_batch_word_ids, train_batch_utterance_lengths = pad_sequences(train_batch_dialogues, 0, nlevels=2)
                true_labs = train_batch_labels
                train_batch_true_labels, _ = pad_sequences(true_labs, 0)
                counter += 1
                train_loss, train_accuracy, _ = sess.run(
                    [model.loss, model.accuracy,model.train_op],
                    feed_dict = {
                      model.word_ids: train_batch_word_ids,
                      model.utterance_lengths: train_batch_utterance_lengths,
                      model.dialogue_lengths: train_batch_dialogue_lengths,
                      model.labels: train_batch_true_labels,
                      model.clip: clip
                    }
                )
                print("step = {}, train_loss = {}, train_accuracy = {}".format(counter, train_loss, train_accuracy))

                train_precision_summ = tf.Summary()
                train_precision_summ.value.add(tag='train_accuracy', simple_value=train_accuracy)
                writer.add_summary(train_precision_summ, counter)

                train_loss_summ = tf.Summary()
                train_loss_summ.value.add(tag='train_loss', simple_value=train_loss)
                writer.add_summary(train_loss_summ, counter)

                if counter % 100 == 0:
                    dev_loss = []
                    dev_acc = []

                    for dev_batch_dialogues, dev_batch_labels in minibatches(dev_data, dev_labels, batchSize):
                        _, dev_batch_dialogue_lengths = pad_sequences(dev_batch_dialogues, 0)
                        dev_batch_word_ids, dev_batch_utterance_lengths = pad_sequences(dev_batch_dialogues, 0, nlevels=2)
                        true_labs = dev_batch_labels
                        dev_batch_true_labels, _ = pad_sequences(true_labs, 0)
                        dev_batch_loss, dev_batch_acc = sess.run(
                          [model.loss, model.accuracy],
                          feed_dict = {
                            model.word_ids: dev_batch_word_ids,
                            model.utterance_lengths: dev_batch_utterance_lengths,
                            model.dialogue_lengths: dev_batch_dialogue_lengths,
                            model.labels: dev_batch_true_labels,
                            model.clip: clip
                          }
                        )
                        dev_loss.append(dev_batch_loss)
                        dev_acc.append(dev_batch_acc)

                    valid_loss = sum(dev_loss) / len(dev_loss)
                    valid_accuracy = sum(dev_acc) / len(dev_acc)

                    dev_precision_summ = tf.Summary()
                    dev_precision_summ.value.add(tag='dev_accuracy', simple_value=valid_accuracy)
                    writer.add_summary(dev_precision_summ, counter)

                    dev_loss_summ = tf.Summary()
                    dev_loss_summ.value.add(tag='dev_loss', simple_value=valid_loss)
                    writer.add_summary(dev_loss_summ, counter)
                    print("counter = {}, dev_loss = {}, dev_accuacy = {}".format(counter, valid_loss, valid_accuracy))

                    saver.save(sess, "../trained_model/kumar/model_glove")

        test_losses = []
        test_accs = []
        for test_batch_dialogues, test_batch_labels in minibatches(test_data, test_labels, batchSize):
            _, test_batch_dialogue_lengths = pad_sequences(test_batch_dialogues, 0)
            test_batch_word_ids, test_batch_utterance_lengths = pad_sequences(test_batch_dialogues, 0, nlevels=2)
            true_labs = test_batch_labels
            test_batch_true_labels, _ = pad_sequences(true_labs, 0)
            test_batch_loss, test_batch_acc = sess.run(
                [model.loss, model.accuracy],
                feed_dict={
                    model.word_ids: test_batch_word_ids,
                    model.utterance_lengths: test_batch_utterance_lengths,
                    model.dialogue_lengths: test_batch_dialogue_lengths,
                    model.labels: test_batch_true_labels,
                    model.clip: clip
                }
            )
            test_losses.append(test_batch_loss)
            test_accs.append(test_batch_acc)

if __name__ == '__main__':
    main()
