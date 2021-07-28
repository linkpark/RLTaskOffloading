
import tensorflow as tf
import numpy as np

from RLWorkflow.seq2seq import model_helper

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from tensorflow.python.ops.distributions import categorical

class EpsilonGreedySampleHelper(tf.contrib.seq2seq.SampleEmbeddingHelper):
    def __init__(self, sequence_length, embedding, start_tokens, end_token, epsilon, epsilon_thresh_hold, softmax_temperature=None, seed=None):
        super(EpsilonGreedySampleHelper, self).__init__(
            embedding, start_tokens, end_token, softmax_temperature, seed
        )
        self._sequence_length = ops.convert_to_tensor(
            sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape())

        self.epsilon_threshold = epsilon_thresh_hold
        self.epsilon = epsilon

    # do the sampling process
    def sample(self, time, outputs, state, name=None):
        """sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))

        def true_fn():
            logits = math_ops.multiply(math_ops.divide(0.5, outputs), outputs)
            sample_id_sampler = categorical.Categorical(logits=logits)
            sample_ids = sample_id_sampler.sample(seed=self._seed)

            return sample_ids

        def false_fn():
            sample_ids = math_ops.argmax(outputs, axis=-1, output_type=dtypes.int32)
            return sample_ids

        sample_ids = tf.cond(
            self.epsilon < self.epsilon_threshold, true_fn=true_fn, false_fn=false_fn
        )

        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for Sampling Embedding Helper."""
        del outputs  # unused by next_inputs_fn

        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = math_ops.reduce_all(finished)

        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)

class Seq2seqQNet(object):
    def __init__(self, name,
                 hparams, reuse,
                 encoder_inputs,
                 encoder_lengths,
                 decoder_inputs,
                 decoder_full_length,
                 decoder_targets):
        self.encoder_hidden_unit = hparams.num_units
        self.decoder_hidden_unit = hparams.num_units
        self.is_bidencoder = hparams.is_bidencoder
        self.reuse = reuse

        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention
        self.learning_rate = hparams.learning_rate
        self.unit_type = hparams.unit_type
        self.inference_model = hparams.inference_model

        # default setting
        self.mode = tf.contrib.learn.ModeKeys.TRAIN

        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers

        self.single_cell_fn = None
        self.is_greedy = hparams.is_greedy
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token

        self.encoder_inputs = encoder_inputs
        self.encoder_lengths = encoder_lengths
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

        self.decoder_full_length = decoder_full_length
        self.epsilon_threshold = tf.placeholder(dtype=tf.float32, shape=())
        self.epsilon = tf.placeholder(dtype=tf.float32, shape=())

        with tf.variable_scope(name, reuse=self.reuse, initializer=tf.glorot_normal_initializer()):
            self.scope = tf.get_variable_scope().name
            self.embeddings = tf.Variable(tf.random_uniform(
                [self.n_features,
                 self.encoder_hidden_unit],
                -1.0, 1.0), dtype=tf.float32)

            # using a fully connected layer as embeddings
            self.encoder_embeddings = tf.contrib.layers.fully_connected(self.encoder_inputs,
                                                                        self.encoder_hidden_unit,
                                                                        activation_fn=None,
                                                                        scope="encoder_embeddings",
                                                                        reuse=tf.AUTO_REUSE)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings,
                                                             self.decoder_inputs)

            self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets,
                                                         self.n_features,
                                                         dtype=tf.float32)

            self.output_layer = tf.layers.Dense(self.n_features, use_bias=False, name="output_projection")

            if self.is_bidencoder:
                self.encoder_outputs, self.encoder_state = self.create_bidrect_encoder(hparams)
            else:
                self.encoder_outputs, self.encoder_state = self.create_encoder(hparams)

            # training decoder
            self.decoder_outputs, self.decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                           self.encoder_state, model="train")
            self.q_logits = self.decoder_outputs.rnn_output
            self.q = tf.reduce_sum(self.q_logits * self.decoder_targets_embeddings, axis=-1)

            # sample decoder
            self.sample_decoder_outputs, self.sample_decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                                         self.encoder_state,
                                                                                         model="sample")
            self.sample_q_logits = self.sample_decoder_outputs.rnn_output
            self.sample_action = self.sample_decoder_outputs.sample_id
            one_hot_sample_action = tf.one_hot(self.sample_action,
                                             self.n_features,
                                             dtype=tf.float32)

            self.sample_q = tf.reduce_sum(self.sample_q_logits * one_hot_sample_action, axis=-1)

            # greedy decoder
            self.greedy_decoder_outputs, self.greedy_decoder_state = self.create_decoder(hparams, self.encoder_outputs,
                                                                                         self.encoder_state,
                                                                                         model="greedy")
            self.greedy_q = self.greedy_decoder_outputs.rnn_output
            self.greedy_action = self.greedy_decoder_outputs.sample_id



    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def _build_decoder_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        """Build a multi-layer RNN cell that can be used by decoder"""
        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def create_encoder(self, hparams):
        # Build RNN cell
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
            encoder_cell = self._build_encoder_cell(hparams=hparams,
                                                    num_layers=self.num_layers,
                                                    num_residual_layers=self.num_residual_layers)

            # encoder_cell = tf.contrib.rnn.GRUCell(self.encoder_hidden_unit)
            # currently only consider the normal dynamic rnn
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                sequence_length = self.encoder_lengths,
                inputs=self.encoder_embeddings,
                dtype=tf.float32,
                time_major=self.time_major,
                swap_memory=True,
                scope=scope
            )

        return encoder_outputs, encoder_state

    def create_bidrect_encoder(self, hparams):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
            num_bi_layers = int(self.num_layers / 2)
            num_bi_residual_layers = int(self.num_residual_layers / 2)
            forward_cell = self._build_encoder_cell(hparams=hparams,
                                                    num_layers=num_bi_layers,
                                                    num_residual_layers=num_bi_residual_layers)
            backward_cell = self._build_encoder_cell(hparams=hparams,
                                                     num_layers=num_bi_layers,
                                                     num_residual_layers=num_bi_residual_layers)

            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell,
                backward_cell,
                inputs=self.encoder_embeddings,
                time_major=self.time_major,
                swap_memory=True,
                dtype=tf.float32)

            encoder_outputs = tf.concat(bi_outputs, -1)

            if num_bi_layers == 1:
                encoder_state = bi_state
            else:
                encoder_state = []
                for layer_id in range(num_bi_layers):
                    encoder_state.append(bi_state[0][layer_id])  # forward
                    encoder_state.append(bi_state[1][layer_id])  # backward

                encoder_state = tuple(encoder_state)

            return encoder_outputs, encoder_state

    def create_decoder(self, hparams, encoder_outputs, encoder_state, model):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:
            if model == "greedy":
                self.helper  = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embeddings,
                    # Batchsize * Start_token
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
                )

            elif model == "sample":
                self.helper  = EpsilonGreedySampleHelper(
                    sequence_length=self.decoder_full_length,
                    embedding=self.embeddings,
                    epsilon_thresh_hold=self.epsilon_threshold,
                    epsilon = self.epsilon,
                    start_tokens=tf.fill([tf.size(self.decoder_full_length)], self.start_token),
                    end_token=self.end_token
                )

            elif model == "train":
                self.helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)
            else:
                self.helper  = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)

            if self.is_attention:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)
                # decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_unit)
                if self.time_major:
                    # [batch_size, max_time, num_nunits]
                    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
                else:
                    attention_states = encoder_outputs

                print("decoder hidden unit: ",self.decoder_hidden_unit)
                print("attention states: ", attention_states.shape)
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.decoder_hidden_unit, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.decoder_hidden_unit)

                decoder_initial_state = (
                    decoder_cell.zero_state(tf.size(self.decoder_full_length),
                                            dtype=tf.float32).clone(
                        cell_state=encoder_state))
                print("decoder_initial_state is:", decoder_initial_state)

            else:
                decoder_cell = self._build_decoder_cell(hparams=hparams,
                                                        num_layers=self.num_layers,
                                                        num_residual_layers=self.num_residual_layers)

                decoder_initial_state = encoder_state

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=self.helper,
                initial_state=decoder_initial_state,
                output_layer=self.output_layer)

            outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                       output_time_major=self.time_major,
                                                                       maximum_iterations=self.decoder_full_length[0])
        return outputs, last_state

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def sample_from_current_policy(self, sess, encoder_input_batch, decoder_full_length):
        return sess.run([self.sample_q, self.sample_action],
                        feed_dict={
                            self.encoder_inputs: encoder_input_batch,
                            self.decoder_full_length: decoder_full_length
                        })

    def step(self, encoder_input_batch, decoder_full_length, epsilon_threshold=1.0):
        sess = tf.get_default_session()

        if self.time_major == True:
            encoder_input_batch = np.swapaxes(encoder_input_batch, 0, 1)

        # random generate epsilon
        epsilon = np.random.uniform(low=0.0, high=1.0)

        sample_action_sequence, sample_greedy_action_sequence = sess.run([self.sample_action, self.greedy_action],
                        feed_dict={
                            self.encoder_inputs: encoder_input_batch,
                            self.decoder_full_length: decoder_full_length,
                            self.epsilon_threshold: epsilon_threshold,
                            self.epsilon: epsilon
                        })
        if self.time_major == True:
            sample_action_sequence = np.array(sample_action_sequence).swapaxes(0,1)
            sample_greedy_action_sequence = np.array(sample_greedy_action_sequence).swapaxes(0,1)

        return np.array(sample_action_sequence), np.array(sample_greedy_action_sequence)

    def get_qvalues(self, encoder_input_batch, decoder_input, decoder_full_length, decoder_target):
        sess = tf.get_default_session()

        if self.time_major == True:
            encoder_input_batch = np.swapaxes(encoder_input_batch, 0, 1)

        q_values = sess.run(self.q,
                             feed_dict={
                                 self.encoder_inputs: encoder_input_batch,
                                 self.decoder_full_length: decoder_full_length,
                                 self.decoder_inputs: decoder_input,
                                 self.decoder_targets: decoder_target
                             })

        return q_values



if __name__ == "__main__":
    hparams = tf.contrib.training.HParams(
        unit_type="layer_norm_lstm",
        num_units=256,
        learning_rate=0.00005,
        supervised_learning_rate=0.00005,
        n_features=2,
        time_major=False,
        is_attention=True,
        forget_bias=1.0,
        dropout=0,
        num_gpus=1,
        num_layers=2,
        num_residual_layers=0,
        is_greedy=False,
        inference_model="sample",
        start_token=0,
        end_token=5,
        is_bidencoder=True
    )

    ob_numpy = np.random.random(size=(5, 5, 25))
    ob_length_numpy = np.array([5]*5)

    with tf.Session() as sess:
        decoder_input = tf.placeholder(tf.int32, [None, None])
        action = tf.placeholder(tf.int32, [None, None])
        action_length = tf.placeholder(tf.int32, [None])

        ob = tf.placeholder(dtype=tf.float32, shape=[None, None, 25])
        ob_length = tf.placeholder(dtype=tf.int32, shape=[None])

        train_model = Seq2seqQNet("q-value", hparams, reuse=True, encoder_inputs=ob,
                                    encoder_lengths=ob_length,
                                    decoder_inputs=decoder_input,
                                    decoder_full_length=action_length,
                                    decoder_targets=action)
        sess.run(tf.global_variables_initializer())

        sample_action, sample_greedy_action = train_model.step(ob_numpy, ob_length_numpy, epsilon_threshold=0.1)

        print("sample_action: ", sample_action)
        print("sample_greedy_action:", sample_greedy_action)

        sample_decoder_input = np.column_stack(
            (np.ones(sample_action.shape[0], dtype=np.int32) * 0, sample_action[:, 0:-1]))

        # print("decdoer input: ", sess.run(decoder_input, feed_dict={decoder_input:sample_decoder_input}))
        # print(sample_decoder_input)
        q_values = train_model.get_qvalues(encoder_input_batch=ob_numpy, decoder_input=sample_decoder_input,
                                       decoder_full_length=ob_length_numpy, decoder_target = sample_action)
        print("q_values: ", q_values)




