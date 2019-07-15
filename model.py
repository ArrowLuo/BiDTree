import os
import tensorflow as tf
from data_utils import minibatches, pad_sequences, get_chunks
from general_utils import Progbar, print_sentence
import general_utils as logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

class DepsModel(object):
    def __init__(self, config, embeddings, logger=None, graph_suffix=None, ):
        """
        Args:
            config: class with hyper parameters
            embeddings: np array with embeddings
            logger: logger instance
        """
        self.config = config
        self.embeddings = embeddings
        self.ntags = config.ntags
        self.nwords = config.nwords
        self.nchars = config.nchars
        self.nrels = config.nrels
        self.max_sentence_size = config.max_sentence_size
        self.max_word_size = config.max_word_size
        self.max_btup_deps_len = config.max_btup_deps_len
        self.max_upbt_deps_len = config.max_upbt_deps_len

        if graph_suffix is None:
            graph_suffix = '0'
        self.graph_suffix = str(graph_suffix)

        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        self.logger = logger
        self.initer = tf.truncated_normal_initializer(stddev=0.01)

    def add_placeholders(self):
        """
        Adds placeholders to self
        """
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size],
                                       name="word_ids" + self.graph_suffix)
        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size, self.max_word_size],
                                       name="char_ids" + self.graph_suffix)
        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths" + self.graph_suffix)

        # shape = (batch size, max length of sentence in batch)
        self.btup_word_orders = tf.placeholder(tf.int32, shape=[None, None],
                                               name="btup_word_orders" + self.graph_suffix)
        self.upbt_word_orders = tf.placeholder(tf.int32, shape=[None, None],
                                               name="upbt_word_orders" + self.graph_suffix)
        self.btup_word_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size],
                                            name="btup_word_ids" + self.graph_suffix)
        self.upbt_word_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size],
                                            name="upbt_word_ids" + self.graph_suffix)
        self.btup_formidxs = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size],
                                            name="btup_formidxs" + self.graph_suffix)
        self.upbt_formidxs = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size],
                                            name="upbt_formidxs" + self.graph_suffix)

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths" + self.graph_suffix)

        # shape = (batch size, max length of sentence, max length of deps)
        self.btup_deps_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size, self.max_btup_deps_len],
                                            name="btup_deps_ids" + self.graph_suffix)
        self.upbt_deps_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size, self.max_upbt_deps_len],
                                            name="upbt_deps_ids" + self.graph_suffix)
        self.btup_rels_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size, self.max_btup_deps_len],
                                            name="btup_rels_ids" + self.graph_suffix)
        self.upbt_rels_ids = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size, self.max_upbt_deps_len],
                                            name="upbt_rels_ids" + self.graph_suffix)
        self.btup_deps_lens = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size],
                                             name="btup_deps_lens" + self.graph_suffix)
        self.upbt_deps_lens = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size],
                                             name="upbt_deps_lens" + self.graph_suffix)

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, self.max_sentence_size], name="labels" + self.graph_suffix)

        # hyper parameters
        self.tbatch_size = tf.placeholder(dtype=tf.int32, shape=[], name="tbatch_size" + self.graph_suffix)
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout" + self.graph_suffix)
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr" + self.graph_suffix)

    def init_embedding(self, task_name):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope("words" + task_name + self.graph_suffix):
            nil_word_slot = np.ndarray(shape=(1, self.embeddings.shape[-1]), dtype=np.float32,
                                       buffer=np.random.randn(1, self.embeddings.shape[-1]))
            _embeddings_ = np.concatenate((self.embeddings, nil_word_slot), axis=0)
            _word_embeddings_ = tf.Variable(_embeddings_, dtype=tf.float32, trainable=self.config.train_embeddings)
            self.btup_word_embeddings = tf.nn.embedding_lookup(_word_embeddings_, self.btup_word_ids)
            self.upbt_word_embeddings = tf.nn.embedding_lookup(_word_embeddings_, self.upbt_word_ids)
            self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings_, self.word_ids)
        with tf.variable_scope("rels" + task_name + self.graph_suffix):
            nil_rels_slot = np.ndarray(shape=(1, self.config.dim_rel), dtype=np.float32,
                                       buffer=np.random.randn(1, self.embeddings.shape[-1]))
            _embeddings_ = tf.concat(values=[self.initer([self.nrels, self.config.dim_rel]), nil_rels_slot], axis=0)
            _rels_embeddings_ = tf.Variable(_embeddings_, dtype=tf.float32, trainable=self.config.train_embeddings)
            self.btup_rels_embeddings = tf.nn.embedding_lookup(_rels_embeddings_, self.btup_rels_ids)
            self.upbt_rels_embeddings = tf.nn.embedding_lookup(_rels_embeddings_, self.upbt_rels_ids)

    def lstm_dep_init(self, channel, dep_input_size, hidden_size, max_num_childs):
        init_const = tf.zeros([1, hidden_size])
        with tf.variable_scope(channel):
            W_i = tf.get_variable("W_i", shape=[dep_input_size, hidden_size], initializer=self.initer)
            U_i = tf.get_variable("U_i", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)
            b_i = tf.get_variable("b_i", initializer=init_const)
            U_it = tf.get_variable("U_it", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)

            W_f = tf.get_variable("W_f", shape=[dep_input_size, hidden_size], initializer=self.initer)
            b_f = tf.get_variable("b_f", initializer=init_const)
            U_f = tf.get_variable("U_f", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)
            U_ft = tf.get_variable("U_ft", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)

            W_o = tf.get_variable("W_o", shape=[dep_input_size, hidden_size], initializer=self.initer)
            U_o = tf.get_variable("U_o", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)
            b_o = tf.get_variable("b_o", initializer=init_const)
            U_ot = tf.get_variable("U_ot", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)

            W_u = tf.get_variable("W_u", shape=[dep_input_size, hidden_size], initializer=self.initer)
            U_u = tf.get_variable("U_u", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)
            b_u = tf.get_variable("b_u", initializer=init_const)
            U_ut = tf.get_variable("U_ut", shape=[self.nrels, hidden_size, hidden_size], initializer=self.initer)

    def cond1(self, i, const, steps, *argks):
        return i < steps

    def cond2(self, i, steps, *argks):
        return i < steps

    def lstm_dep(self, bno, start, seq_len, word_embeddings, rels_embeddings, rels_ids,
                 input_childs, input_num_child, init_state_dep, word_orders, scope):

        def loop_over_seq(ind, const, steps,
                          word_embeddings, rels_embeddings, rels_ids, input_childs, input_num_child,
                          states_dep, states_series):
            word_inputs = tf.expand_dims(word_embeddings[bno][ind], 0)
            childs = input_childs[bno][ind]
            rels_inputs = tf.expand_dims(rels_embeddings[bno][ind], 1)
            num_child = input_num_child[bno][ind]
            rels_id = rels_ids[bno][ind]

            with tf.variable_scope(scope, reuse=True):
                W_i = tf.get_variable("W_i")
                U_i = tf.get_variable("U_i")
                b_i = tf.get_variable("b_i")
                U_it = tf.get_variable("U_it")

                W_f = tf.get_variable("W_f")
                b_f = tf.get_variable("b_f")
                U_f = tf.get_variable("U_f")
                U_ft = tf.get_variable("U_ft")

                W_o = tf.get_variable("W_o")
                U_o = tf.get_variable("U_o")
                b_o = tf.get_variable("b_o")
                U_ot = tf.get_variable("U_ot")

                W_u = tf.get_variable("W_u")
                U_u = tf.get_variable("U_u")
                b_u = tf.get_variable("b_u")
                U_ut = tf.get_variable("U_ut")

                it = tf.matmul(word_inputs, W_i) + b_i
                ot = tf.matmul(word_inputs, W_o) + b_o
                ut = tf.matmul(word_inputs, W_u) + b_u

                def matmul(k, steps, it, ot, ut):
                    it += tf.matmul(states_series[0][childs[k]], U_i[rels_id[k]])
                    ot += tf.matmul(states_series[0][childs[k]], U_o[rels_id[k]])
                    ut += tf.matmul(states_series[0][childs[k]], U_u[rels_id[k]])
                    it += tf.matmul(rels_inputs[k], U_it[rels_id[k]])
                    ot += tf.matmul(rels_inputs[k], U_ot[rels_id[k]])
                    ut += tf.matmul(rels_inputs[k], U_ut[rels_id[k]])
                    return k + 1, steps, it, ot, ut

                _, _, ht_i, ht_o, ht_u = tf.while_loop(self.cond2, matmul, [0, num_child, it, ot, ut])

                u_input = tf.tanh(ht_u)
                input_gate = tf.sigmoid(ht_i)
                output_gate = tf.sigmoid(ht_o)
                cell_state = input_gate * u_input

                def child_sum(k, steps, ft):
                    ft += tf.matmul(states_series[0][childs[k]], U_f[rels_id[k]])
                    ft += tf.matmul(rels_inputs[k], U_ft[rels_id[k]])
                    return k + 1, steps, ft

                ft = tf.matmul(word_inputs, W_f) + b_f

                def cell_state_sp(k, steps, cell_state):
                    _, _, f_sp = tf.while_loop(self.cond2, child_sum, [k, k + 1, ft])
                    cell_state += tf.sigmoid(f_sp) * states_series[1][childs[k]]
                    return k + 1, steps, cell_state

                _, _, cell_state = tf.while_loop(self.cond2, cell_state_sp, [0, num_child, cell_state])

                # [1, hidden_size]
                hds = tf.expand_dims(output_gate * tf.tanh(cell_state), 0)
                cds = tf.expand_dims(cell_state, 0)
                # [2, hidden_size]
                states_dep = tf.stack([hds, cds], axis=0)

                hds_ = tf.cond(tf.equal(ind, const), lambda: states_dep[0],
                               lambda: tf.concat([states_series[0], states_dep[0]], 0))
                cds_ = tf.cond(tf.equal(ind, const), lambda: states_dep[1],
                               lambda: tf.concat([states_series[1], states_dep[1]], 0))
                states_series = tf.stack([hds_, cds_], axis=0)

            return ind + 1, const, steps, \
                   word_embeddings, rels_embeddings, rels_ids, input_childs, input_num_child, \
                   states_dep, states_series

        x = tf.constant(0)
        _, _, _, _, _, _, _, _, _, \
        states_series_dep = tf.while_loop(
            self.cond1, loop_over_seq, [start, start, seq_len,
                                        word_embeddings, rels_embeddings, rels_ids, input_childs, input_num_child,
                                        init_state_dep, init_state_dep],
            shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(),
                              word_embeddings.get_shape(), rels_embeddings.get_shape(),
                              rels_ids.get_shape(),
                              input_childs.get_shape(), input_num_child.get_shape(),
                              tf.TensorShape([2, None, 1, self.config.hidden_size]),
                              tf.TensorShape([2, None, 1, self.config.hidden_size])])

        def loop_over_sort(ind, const, steps, states_series_dep, word_orders, states_series):
            hds = tf.expand_dims(states_series_dep[0][word_orders[bno][ind]], 0)
            cds = tf.expand_dims(states_series_dep[1][word_orders[bno][ind]], 0)
            hds_ = tf.cond(tf.equal(ind, const), lambda: hds,
                           lambda: tf.concat([states_series[0], hds], 0))
            cds_ = tf.cond(tf.equal(ind, const), lambda: cds,
                           lambda: tf.concat([states_series[1], cds], 0))
            states_series = tf.stack([hds_, cds_], axis=0)

            return ind + 1, const, steps, \
                   states_series_dep, word_orders, states_series

        _, _, _, _, _, \
        sorted_states_series = tf.while_loop(self.cond1, loop_over_sort,
                                             [start, start, seq_len, states_series_dep, word_orders,
                                              init_state_dep],
                                             shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(),
                                                               states_series_dep.get_shape(), word_orders.get_shape(),
                                                               tf.TensorShape([2, None, 1, self.config.hidden_size])])

        padding_state = tf.truncated_normal([2, 1, 1, self.config.hidden_size], -0.1, 0.1)

        def paddingSenetnceLength(sid, const, steps, ture_length, padding_state,
                                  hidden_states_series, states_series):
            hds = tf.cond(tf.less(sid, ture_length), lambda: hidden_states_series[0][sid],
                          lambda: padding_state[0][0])
            cds = tf.cond(tf.less(sid, ture_length), lambda: hidden_states_series[1][sid],
                          lambda: padding_state[1][0])
            hds = tf.expand_dims(hds, 0)
            cds = tf.expand_dims(cds, 0)

            hds_ = tf.cond(tf.equal(sid, const), lambda: hds,
                           lambda: tf.concat([states_series[0], hds], 0))
            cds_ = tf.cond(tf.equal(sid, const), lambda: cds,
                           lambda: tf.concat([states_series[1], cds], 0))
            states_series = tf.stack([hds_, cds_], axis=0)
            return sid + 1, const, steps, ture_length, padding_state, \
                   hidden_states_series, states_series

        _, _, _, _, _, _, \
        states_series_dep = tf.while_loop(self.cond1, paddingSenetnceLength,
                                          [start, start, self.max_sentence_size, seq_len, padding_state,
                                           sorted_states_series, init_state_dep],
                                          shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(), x.get_shape(),
                                                            padding_state.get_shape(), sorted_states_series.get_shape(),
                                                            tf.TensorShape([2, None, 1, self.config.hidden_size])])

        return states_series_dep

    def bilstm_dep(self, bid=0):
        init_state = tf.zeros([2, 1, 1, self.config.hidden_size])
        btup_states_series_dep = self.lstm_dep(bid, 0, self.sequence_lengths[bid],
                                               self.btup_word_embeddings, self.btup_rels_embeddings,
                                               self.btup_rels_ids,
                                               self.btup_deps_ids, self.btup_deps_lens, init_state,
                                               self.btup_word_orders, "lstm_btup")
        upbt_states_series_dep = self.lstm_dep(bid, 0, self.sequence_lengths[bid],
                                               self.upbt_word_embeddings, self.upbt_rels_embeddings,
                                               self.upbt_rels_ids,
                                               self.upbt_deps_ids, self.upbt_deps_lens, init_state,
                                               self.upbt_word_orders, "lstm_upbt")
        return btup_states_series_dep, upbt_states_series_dep

    def add_bilstm_op(self, task_name, combine_embeddings):
        """
        """
        with tf.variable_scope("bi-lstm" + task_name + self.graph_suffix):
            lstm_cell_f = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            lstm_cell_b = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_f, lstm_cell_b, combine_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            combine_embeddings_output = tf.nn.dropout(output, self.dropout)
        return combine_embeddings_output

    def add_logits_op(self, task_name, inputv):
        inputv_shape = inputv.get_shape().as_list()
        in_size = inputv_shape[-1]
        out_size = self.ntags
        with tf.variable_scope("proj" + task_name + self.graph_suffix):
            W = tf.Variable(np.random.randn(in_size, out_size), name="W", dtype=tf.float32) / np.sqrt(in_size / 2)
            b = tf.Variable(np.zeros([out_size]), name="b", dtype=tf.float32)
            output = tf.reshape(inputv, [-1, in_size])
            pred = tf.nn.relu(tf.matmul(output, W) + b)
            self.logits = tf.reshape(pred, [-1, inputv_shape[1], out_size])  # shape = (?,?,3)

    def add_loss_op(self, task_name):
        """
        Adds loss to self
        """
        num_tags = self.ntags
        transitions = tf.Variable(np.random.randn(num_tags, num_tags),
                                  name="transitions" + task_name + self.graph_suffix, dtype=tf.float32) / np.sqrt(
            num_tags / 2)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths, transitions
        )
        self.loss = tf.reduce_mean(-log_likelihood)

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += 0.0001 * sum(regularization_loss)

    def add_train_op(self, task_name):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step" + task_name + self.graph_suffix):
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def build(self):
        self.add_placeholders()
        self.init_embedding('_ae_')

        dep_input_size = self.config.dim
        self.lstm_dep_init("lstm_btup", dep_input_size, self.config.hidden_size, self.max_btup_deps_len)
        self.lstm_dep_init("lstm_upbt", dep_input_size, self.config.hidden_size, self.max_upbt_deps_len)

        def generateBatch(bid, const, steps, hidden_states_series):
            btup_states_series_dep, upbt_states_series_dep = self.bilstm_dep(bid)
            hidden_states = tf.concat([btup_states_series_dep[0], upbt_states_series_dep[0]], -1)
            hidden_states = tf.expand_dims(hidden_states[:, 0], 0)
            hidden_states = tf.nn.dropout(hidden_states, self.dropout)
            hidden_states_series = tf.cond(tf.equal(bid, const), lambda: hidden_states,
                                           lambda: tf.concat([hidden_states_series, hidden_states], 0))
            return bid + 1, const, steps, hidden_states_series

        hidden_states_size = 2 * self.config.hidden_size
        x = tf.constant(0)
        init_hidden_states = tf.zeros([1, 1, hidden_states_size])
        _, _, _, hidden_states_series = tf.while_loop(self.cond1, generateBatch,
                                                      [0, 0, self.tbatch_size, init_hidden_states],
                                                      shape_invariants=[x.get_shape(), x.get_shape(),
                                                                        self.tbatch_size.get_shape(),
                                                                        tf.TensorShape(
                                                                            [None, None, hidden_states_size])])

        hidden_states_series = tf.reshape(hidden_states_series, [-1, self.max_sentence_size, hidden_states_size])
        hidden_states_series = tf.nn.dropout(hidden_states_series, self.dropout)
        hidden_states_series = self.add_bilstm_op("_wo_m_", hidden_states_series)
        self.add_logits_op('_ae_', hidden_states_series)
        self.add_loss_op('_ae_')
        self.add_train_op('_ae_')
        self.add_init_op()

    def get_feed_dict(self, words, poss, chunks, labels=None,
                      btup_idx_list=None, btup_words_list=None, btup_depwords_list=None,
                      btup_deprels_list=None, btup_depwords_length_list=None,
                      upbt_idx_list=None, upbt_words_list=None, upbt_depwords_list=None,
                      upbt_deprels_list=None, upbt_depwords_length_list=None,
                      btup_formidx_list=None, upbt_formidx_list=None, lr=None, dropout=None):
        """
        Given some data, pad it and build a feed dictionary
        """
        # perform padding of the given data
        if self.config.chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, self.nwords, self.max_sentence_size,
                                                       self.max_word_size)
            char_ids, word_lengths = pad_sequences(char_ids, self.nchars, self.max_sentence_size, self.max_word_size,
                                                   nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, self.nwords, self.max_sentence_size, self.max_word_size)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }
        if self.config.chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        if labels is not None:
            labels, _ = pad_sequences(labels, 2, self.max_sentence_size, self.max_word_size)
            feed[self.labels] = labels
        if lr is not None:
            feed[self.lr] = lr
        if dropout is not None:
            feed[self.dropout] = dropout

        # Begin using deps tree
        feed[self.tbatch_size] = len(btup_idx_list)
        if btup_idx_list is not None:
            btup_idx_list, _ = pad_sequences(btup_idx_list, -1, self.max_sentence_size)
            feed[self.btup_word_orders] = btup_idx_list
        if btup_words_list is not None:
            btup_words_list, _ = pad_sequences(btup_words_list, self.nwords, self.max_sentence_size)
            feed[self.btup_word_ids] = btup_words_list
        if btup_depwords_list is not None:
            btup_depwords_list, _ = pad_sequences(btup_depwords_list, -1, self.max_sentence_size,
                                                  self.max_btup_deps_len, nlevels=2)
            feed[self.btup_deps_ids] = btup_depwords_list
        if btup_deprels_list is not None:
            btup_deprels_list, _ = pad_sequences(btup_deprels_list, self.nrels, self.max_sentence_size,
                                                 self.max_btup_deps_len, nlevels=2)
            feed[self.btup_rels_ids] = btup_deprels_list
        if btup_depwords_length_list is not None:
            btup_depwords_length_list, _ = pad_sequences(btup_depwords_length_list, 0, self.max_sentence_size)
            feed[self.btup_deps_lens] = btup_depwords_length_list

        if upbt_idx_list is not None:
            upbt_idx_list, _ = pad_sequences(upbt_idx_list, -1, self.max_sentence_size)
            feed[self.upbt_word_orders] = upbt_idx_list
        if upbt_words_list is not None:
            upbt_words_list, _ = pad_sequences(upbt_words_list, self.nwords, self.max_sentence_size)
            feed[self.upbt_word_ids] = upbt_words_list
        if upbt_depwords_list is not None:
            upbt_depwords_list, _ = pad_sequences(upbt_depwords_list, -1, self.max_sentence_size,
                                                  self.max_upbt_deps_len, nlevels=2)
            feed[self.upbt_deps_ids] = upbt_depwords_list
        if upbt_deprels_list is not None:
            upbt_deprels_list, _ = pad_sequences(upbt_deprels_list, self.nrels, self.max_sentence_size,
                                                 self.max_upbt_deps_len, nlevels=2)
            feed[self.upbt_rels_ids] = upbt_deprels_list
        if upbt_depwords_length_list is not None:
            upbt_depwords_length_list, _ = pad_sequences(upbt_depwords_length_list, 0, self.max_sentence_size)
            feed[self.upbt_deps_lens] = upbt_depwords_length_list

        if btup_formidx_list is not None:
            btup_formidx_list, _ = pad_sequences(btup_formidx_list, -1, self.max_sentence_size)
            feed[self.btup_formidxs] = btup_formidx_list
        if upbt_formidx_list is not None:
            upbt_formidx_list, _ = pad_sequences(upbt_formidx_list, -1, self.max_sentence_size)
            feed[self.upbt_formidxs] = upbt_formidx_list

        return feed, sequence_lengths

    def predict_batch(self, sess, words, poss, chunks,
                      btup_idx_list, btup_words_list, btup_depwords_list, btup_deprels_list, btup_depwords_length_list,
                      upbt_idx_list, upbt_words_list, upbt_depwords_list, upbt_deprels_list, upbt_depwords_length_list,
                      btup_formidx_list, upbt_formidx_list):
        fd, sequence_lengths = self.get_feed_dict(words, poss, chunks, None,
                                                  btup_idx_list, btup_words_list, btup_depwords_list,
                                                  btup_deprels_list, btup_depwords_length_list,
                                                  upbt_idx_list, upbt_words_list, upbt_depwords_list,
                                                  upbt_deprels_list, upbt_depwords_length_list,
                                                  btup_formidx_list, upbt_formidx_list, dropout=1.0)

        viterbi_sequences = []
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=fd)
        # iterate over the sentences
        for logit, sequence_length in zip(logits, sequence_lengths):
            # keep only the valid time steps
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences, sequence_lengths

    def run_epoch(self, sess, train, train_deps, dev, dev_deps, vocab_words, vocab_tags, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        """
        self.config.istrain = True  # set to train first, #batch normalization#
        nbatches = (len(train_deps) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (words, poss, chunks, labels,
                btup_idx_list, btup_words_list, btup_depwords_list, btup_deprels_list, btup_depwords_length_list,
                upbt_idx_list, upbt_words_list, upbt_depwords_list, upbt_deprels_list, upbt_depwords_length_list,
                btup_formidx_list, upbt_formidx_list) in enumerate(
            minibatches(train, train_deps, self.config.batch_size)):
            fd, sequence_lengths = self.get_feed_dict(words, poss, chunks, labels,
                                                      btup_idx_list, btup_words_list, btup_depwords_list,
                                                      btup_deprels_list, btup_depwords_length_list,
                                                      upbt_idx_list, upbt_words_list, upbt_depwords_list,
                                                      upbt_deprels_list, upbt_depwords_length_list,
                                                      btup_formidx_list, upbt_formidx_list, self.config.lr,
                                                      self.config.dropout)

            _, train_loss, logits = sess.run([self.train_op, self.loss, self.logits], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])

        acc, recall, f1, test_acc = self.run_evaluate(sess, dev, dev_deps, vocab_words, vocab_tags)
        self.logger.info(
            "- dev acc {:04.2f} - dev recall {:04.2f} - f1 {:04.2f} - test acc {:04.2f}".format(100 * acc, 100 * recall,
                                                                                                100 * f1,
                                                                                                100 * test_acc))
        return acc, recall, f1, train_loss

    def train(self, train, train_deps, dev, dev_deps, vocab_words, vocab_tags):
        """
        Performs training with early stopping and lr exponential decay
        """
        best_score = 0
        saver = tf.train.Saver()
        nepoch_no_imprv = 0

        gpuConfig = tf.ConfigProto()
        gpuConfig.gpu_options.allow_growth = True
        with tf.Session(config=gpuConfig) as sess:
            sess.run(self.init)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
                acc, recall, f1, train_loss = self.run_epoch(sess, train, train_deps, dev, dev_deps, vocab_words,
                                                             vocab_tags, epoch)

                # early stopping and saving best parameters
                if f1 > best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1
                    self.logger.info("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        break

    def get_aspect_polarity_pairs(self, asps):
        strs = []
        for a in asps:
            strs.append(str(a[1]) + "-" + str(a[2]) + "-" + a[0])
        return strs

    def run_evaluate(self, sess, test, test_deps, vocab_words, vocab_tags, print_test_results=False):
        """
        Evaluates performance on test set
        """
        idx_to_words = {}
        if print_test_results:
            idx_to_words = {idx: word for word, idx in vocab_words.iteritems()}

        test_accs = []
        self.config.istrain = False  # set to test first, #batch normalization#
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, poss, chunks, labels, \
            btup_idx_list, btup_words_list, btup_depwords_list, btup_deprels_list, btup_depwords_length_list, \
            upbt_idx_list, upbt_words_list, upbt_depwords_list, upbt_deprels_list, upbt_depwords_length_list, \
            btup_formidx_list, upbt_formidx_list in minibatches(test, test_deps, self.config.batch_size):

            labels_pred, sequence_lengths = self.predict_batch(sess, words, poss, chunks,
                                                               btup_idx_list, btup_words_list, btup_depwords_list,
                                                               btup_deprels_list, btup_depwords_length_list,
                                                               upbt_idx_list, upbt_words_list, upbt_depwords_list,
                                                               upbt_deprels_list, upbt_depwords_length_list,
                                                               btup_formidx_list, upbt_formidx_list)
            if print_test_results:
                char_ids, word_ids = zip(*words)

            index = 0
            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                test_accs += map(lambda a_b: a_b[0] == a_b[1], zip(lab, lab_pred))

                lab_chunks = set(get_chunks(lab, vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, vocab_tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

                if print_test_results:
                    self.logger.info(" ".join([idx_to_words[w] for w in word_ids[index][:length]]))
                    self.logger.info(" ".join(self.get_aspect_polarity_pairs(lab_chunks)))
                    self.logger.info(" ".join(self.get_aspect_polarity_pairs(lab_pred_chunks)))

                index += 1

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        test_acc = np.mean(test_accs)
        return p, r, f1, test_acc

    def evaluate(self, test, test_deps, vocab_words, vocab_tags):
        saver = tf.train.Saver()

        gpuConfig = tf.ConfigProto()
        gpuConfig.gpu_options.allow_growth = True
        with tf.Session(config=gpuConfig) as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.config.model_output)
            acc, recall, f1, _ = self.run_evaluate(sess, test, test_deps, vocab_words, vocab_tags,
                                                   print_test_results=self.config.show_test_results)
            self.logger.info(
                "- test acc {:04.2f} - test recall {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * recall, 100 * f1))
