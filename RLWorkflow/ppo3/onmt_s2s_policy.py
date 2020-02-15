import tensorflow as tf
import numpy as np

import opennmt as onmt
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from opennmt.utils.beam_search import get_state_shape_invariants
from RLWorkflow.common import tf_util

hparams = tf.contrib.training.HParams(
    encoder_layers = 3,
    decoder_layers = 3,
    feature_size=5,
    encoder_units=512,
    decoder_units=512,
    encoder_num_heads=8,
    decoder_num_heads=8,
    ff_units = 2048,
    start_token = 0,
    end_token=8
)


# self-attention decoder based on open nmt
class SeqSelfAttentionDecoder(SelfAttentionDecoder):
    def sample_decode(self,
                        symbols_to_logits_fn,
                        initial_ids,
                        end_id,
                        vocab_size,
                        decode_length=None,
                        state=None,
                        return_state=False,
                        seed=None):
        batch_size = tf.shape(initial_ids)[0]
        finished = tf.tile([False], [batch_size])
        step = tf.constant(0)
        inputs = tf.expand_dims(initial_ids, 1)
        lengths = tf.zeros([batch_size], dtype=tf.int32)
        output_logits = tf.zeros([batch_size, 1, vocab_size])

        def _condition(unused_step, finished, unused_inputs,
                       unused_lengths, unused_state, unused_output_logits=None):
            return tf.logical_not(tf.reduce_all(finished))

        def _body(step, finished, inputs, lengths, state, output_logits=None, seed=None):
            inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))

            logits, state = symbols_to_logits_fn(inputs, step, state)
            output_logits = tf.concat((output_logits, logits), axis=1)

            #sample_ids = tf.argmax(probs, axis=-1)

            sample_id_sampler = tf.distributions.Categorical(logits)
            sample_ids = sample_id_sampler.sample(seed=seed)


            next_inputs = tf.concat([inputs, tf.cast(sample_ids, inputs.dtype)], -1)
            next_lengths = inputs_lengths
            next_finished = tf.logical_or(
                finished,
                tf.equal(tf.squeeze(sample_ids, axis=[1]), end_id))
            step = step + 1

            if decode_length is not None:
                next_finished = tf.logical_or(next_finished, step >= decode_length)

            return step, next_finished, next_inputs, next_lengths, state, output_logits

        _, _, outputs, lengths, state, output_logits = tf.while_loop(
            _condition,
            _body,
            loop_vars=(step, finished, inputs, lengths, state, output_logits),
            shape_invariants=(
                tf.TensorShape([]),
                finished.get_shape(),
                tf.TensorShape([None, None]),
                lengths.get_shape(),
                tf.contrib.framework.nest.map_structure(get_state_shape_invariants, state),
                tf.TensorShape([None, None, vocab_size])),
            parallel_iterations=1)

        if return_state:
            return outputs, lengths, output_logits, state
        return outputs, lengths, output_logits

    def greedy_decode(self,
                        symbols_to_logits_fn,
                        initial_ids,
                        end_id,
                        vocab_size,
                        decode_length=None,
                        state=None,
                        return_state=False,
                        seed=None):
        batch_size = tf.shape(initial_ids)[0]
        finished = tf.tile([False], [batch_size])
        step = tf.constant(0)
        inputs = tf.expand_dims(initial_ids, 1)
        lengths = tf.zeros([batch_size], dtype=tf.int32)
        output_logits = tf.zeros([batch_size, 1, vocab_size])

        def _condition(unused_step, finished, unused_inputs,
                       unused_lengths, unused_state, unused_output_logits=None):
            return tf.logical_not(tf.reduce_all(finished))

        def _body(step, finished, inputs, lengths, state, output_logits=None, seed=None):
            inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))

            logits, state = symbols_to_logits_fn(inputs, step, state)
            output_logits = tf.concat((output_logits, logits), axis=1)

            probs = tf.nn.log_softmax(tf.to_float(logits))
            # greedily get the sample id with max probability
            sample_ids = tf.argmax(probs, axis=-1)

            next_inputs = tf.concat([inputs, tf.cast(sample_ids, inputs.dtype)], -1)
            next_lengths = inputs_lengths
            next_finished = tf.logical_or(
                finished,
                tf.equal(tf.squeeze(sample_ids, axis=[1]), end_id))
            step = step + 1

            if decode_length is not None:
                next_finished = tf.logical_or(next_finished, step >= decode_length)

            return step, next_finished, next_inputs, next_lengths, state, output_logits

        _, _, outputs, lengths, state, output_logits = tf.while_loop(
            _condition,
            _body,
            loop_vars=(step, finished, inputs, lengths, state, output_logits),
            shape_invariants=(
                tf.TensorShape([]),
                finished.get_shape(),
                tf.TensorShape([None, None]),
                lengths.get_shape(),
                tf.contrib.framework.nest.map_structure(get_state_shape_invariants, state),
                tf.TensorShape([None, None, vocab_size])),
            parallel_iterations=1)

        if return_state:
            return outputs, lengths, output_logits, state
        return outputs, lengths, output_logits

    def greedy_decoding(self,
                        embedding,
                        start_tokens,
                        end_token,
                        vocab_size=None,
                        initial_state=None,
                        output_layer=None,
                        maximum_iterations=250,
                        mode=tf.estimator.ModeKeys.PREDICT,
                        memory=None,
                        memory_sequence_length=None,
                        dtype=None,
                        return_alignment_history=False):
        cache = self._init_cache(memory, memory_sequence_length=memory_sequence_length)
        symbols_to_logits_fn = self._symbols_to_logits_fn(
            embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype or memory.dtype)

        outputs, lengths, output_logits, cache = self.greedy_decode(
            symbols_to_logits_fn,
            start_tokens,
            end_token,
            vocab_size=vocab_size,
            decode_length=maximum_iterations,
            state=cache,
            return_state=True)
        outputs = tf.slice(outputs, [0, 1], [-1, -1])  # Ignore <s>.
        output_logits = tf.slice(output_logits, [0, 1, 0], [-1, -1, -1])

        # Make shape consistent with beam search.
        #outputs = tf.expand_dims(outputs, 1)
        #lengths = tf.expand_dims(lengths, 1)

        if return_alignment_history:
            attention = tf.expand_dims(cache["attn"], 1)
            return (outputs, None, lengths, output_logits, attention)
        return (outputs, None, lengths, output_logits)

    def sampling_decode(self,
                        embedding,
                        start_tokens,
                        end_token,
                        vocab_size=None,
                        initial_state=None,
                        output_layer=None,
                        maximum_iterations=250,
                        mode=tf.estimator.ModeKeys.PREDICT,
                        memory=None,
                        memory_sequence_length=None,
                        dtype=None,
                        return_alignment_history=False):
        cache = self._init_cache(memory, memory_sequence_length=memory_sequence_length)
        symbols_to_logits_fn = self._symbols_to_logits_fn(
            embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype or memory.dtype)

        outputs, lengths, output_logits, cache = self.sample_decode(
            symbols_to_logits_fn,
            start_tokens,
            end_token,
            vocab_size=vocab_size,
            decode_length=maximum_iterations,
            state=cache,
            return_state=True)
        outputs = tf.slice(outputs, [0, 1], [-1, -1])  # Ignore <s>.
        output_logits = tf.slice(output_logits, [0,1,0], [-1,-1,-1])

        # Make shape consistent with beam search.
        #outputs = tf.expand_dims(outputs, 1)
        #lengths = tf.expand_dims(lengths, 1)

        if return_alignment_history:
            attention = tf.expand_dims(cache["attn"], 1)
            return (outputs, None, lengths, output_logits, attention)
        return (outputs, None, lengths, output_logits)

class SelfAttentionSequencePolicy(object):
    def __init__(self, name,
                 hparams, reuse,
                 encoder_inputs,
                 encoder_lengths,
                 decoder_inputs,
                 decoder_full_length,
                 decoder_targets):
        self.hparams = hparams
        self.encoder_inputs = encoder_inputs
        self.encoder_lengths = encoder_lengths
        self.decoder_inputs = decoder_inputs
        self.decoder_full_length = decoder_full_length
        self.decoder_targets = decoder_targets
        self.time_major=False

        with tf.variable_scope(name, reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
            self.encoder = self._create_self_attention_encoder()
            self.decoder = self._create_self_attention_decoder()
            self.scope = tf.get_variable_scope().name
            self.embeddings = tf.Variable(tf.random_uniform(
                [self.hparams.feature_size,
                 self.hparams.decoder_units],
                -1.0, 1.0), dtype=tf.float32)

            # using a fully connected layer as embeddings
            self.encoder_embeddings = tf.contrib.layers.fully_connected(
                encoder_inputs,
                self.hparams.encoder_units,
                activation_fn=None,
                scope="encoder_embeddings",
                reuse=tf.AUTO_REUSE)

            self.decoder_embeddings = tf.nn.embedding_lookup(self.embeddings,
                                                             decoder_inputs)
            self.decoder_targets_embeddings = tf.one_hot(decoder_targets,
                                                         self.hparams.feature_size,
                                                         dtype=tf.float32)
            self.output_layer = tf.layers.Dense(self.hparams.feature_size, use_bias=False,
                                                name="output_projection")

            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
                self.memory, self.encoder_state, self.memory_sequence_length = \
                    self.encoder.encode(self.encoder_embeddings, sequence_length=self.encoder_lengths)

            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as scope:
                self.training_logits, _, _ = self.decoder.decode(inputs=self.decoder_embeddings,
                                                                sequence_length=self.decoder_full_length,
                                                                vocab_size=self.hparams.feature_size, memory=self.memory,
                                                                memory_sequence_length=self.memory_sequence_length)

                self.pi = tf.nn.softmax(self.training_logits)
                self.q = tf.layers.dense(self.training_logits, self.hparams.feature_size, activation=None, reuse=tf.AUTO_REUSE,
                                         name="qvalue_layer")
                self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)

                self.decoder_prediction = tf.argmax(self.pi, axis=-1)

                start_token_batch = tf.fill([tf.size(self.decoder_full_length)], self.hparams.start_token)

                # sampling from the current policy
                self.sample_decoder_prediction, decoder_state, sequence_length, self.sample_decoder_logits = self.decoder.sampling_decode(
                    self.embeddings,
                    start_tokens=start_token_batch,
                    end_token=self.hparams.end_token,
                    vocab_size=self.hparams.feature_size,
                    maximum_iterations=20,
                    memory=self.memory,
                    initial_state=self.encoder_state,
                    memory_sequence_length=self.memory_sequence_length)
                self.sample_pi = tf.nn.softmax(self.sample_decoder_logits)
                self.sample_q = tf.layers.dense(self.sample_decoder_logits, self.hparams.feature_size,
                                                activation=None, reuse=tf.AUTO_REUSE, name="qvalue_layer")

                self.sample_v = tf.reduce_sum(self.sample_pi * self.sample_q, axis=-1)
                self.sample_neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sample_decoder_prediction,
                                                                                     logits=self.sample_decoder_logits)

                # greedily sampling from current policy
                self.greedy_decoder_prediction, decoder_state, sequence_length, self.greedy_decoder_logits = self.decoder.greedy_decoding(
                    self.embeddings,
                    start_tokens=start_token_batch,
                    end_token=self.hparams.end_token,
                    vocab_size=self.hparams.feature_size,
                    maximum_iterations=20,
                    memory=self.memory,
                    initial_state=self.encoder_state,
                    memory_sequence_length=self.memory_sequence_length)

                self.greedy_pi = tf.nn.softmax(self.greedy_decoder_logits)
                self.greedy_q = tf.layers.dense(self.greedy_decoder_logits, self.hparams.feature_size,
                                                activation=None,
                                                reuse=tf.AUTO_REUSE,
                                                name="qvalue_layer")
                self.greedy_v = tf.reduce_sum(self.greedy_pi * self.greedy_q, axis=-1)

    def _create_self_attention_encoder(self):
        encoder = onmt.encoders.SelfAttentionEncoder(num_layers=self.hparams.encoder_layers,
                                                     num_units=self.hparams.encoder_units,
                                                     num_heads=self.hparams.encoder_num_heads,
                                                     dropout=0.0)

        # encoder = onmt.encoders.ConvEncoder(num_layers=self.hparams.encoder_layers,
        #                                     num_units=self.hparams.encoder_units)

        return encoder

    def _create_self_attention_decoder(self):
        decoder = SeqSelfAttentionDecoder(num_layers=self.hparams.decoder_layers,
                                          num_units=self.hparams.decoder_units,
                                          num_heads=self.hparams.decoder_num_heads,
                                          dropout=0.0)

        return decoder

    def kl(self, other):
        a0 = self.training_logits - tf.reduce_max(self.training_logits, axis=-1, keepdims=True)
        a1 = other.training_logits - tf.reduce_max(other.training_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.training_logits - tf.reduce_max(self.training_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def neglogp(self):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.training_logits,
            labels=self.decoder_targets_embeddings)


    def predict_training(self, sess, encoder_input_batch, encoder_lengths, decoder_input, decoder_full_length):
        return sess.run([self.decoder_prediction, self.pi],
                        feed_dict={
                            self.encoder_inputs: encoder_input_batch,
                            self.encoder_lengths: encoder_lengths,
                            self.decoder_inputs: decoder_input,
                            self.decoder_full_length: decoder_full_length
                        })

    def sample_from_current_policy(self, sess, encoder_input_batch, decoder_full_length):
        batch_size = encoder_input_batch.shape[0]
        sequence_length = encoder_input_batch.shape[1]
        encoder_lengths = np.empty(batch_size, dtype=np.int32)
        encoder_lengths.fill(sequence_length)

        sample_decoder_prediction, sample_v = sess.run([self.sample_decoder_prediction, self.sample_v],
                        feed_dict={
                            self.encoder_inputs: encoder_input_batch,
                            self.encoder_lengths: encoder_lengths,
                            self.decoder_full_length: decoder_full_length
                        })

        return sample_decoder_prediction, sample_v

    def step(self, encoder_input_batch, decoder_full_length, encoder_lengths):
        sess = tf.get_default_session()

        sample_decoder_prediction, sample_v, sample_neglogp = sess.run([self.sample_decoder_prediction, self.sample_v, self.sample_neglogp],
                                                       feed_dict={
                                                           self.encoder_inputs: encoder_input_batch,
                                                           self.encoder_lengths: encoder_lengths,
                                                           self.decoder_full_length: decoder_full_length
                                                       })

        return sample_decoder_prediction, sample_v, sample_neglogp


    def greedy_predict(self, encoder_input_batch, decoder_full_length, encoder_lengths):
        sess = tf.get_default_session()

        return sess.run(self.greedy_decoder_prediction, feed_dict={
                                        self.encoder_inputs: encoder_input_batch,
                                        self.encoder_lengths: encoder_lengths,
                                        self.decoder_full_length: decoder_full_length})

    def logp(self):
        return -self.neglogp()

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def save_policy(self, save_path):
        sess = tf.get_default_session()
        tf_util.save_variables(save_path=save_path, variables=None, sess=sess)

    def load_policy(self, load_path):
        sess = tf.get_default_session()
        tf_util.save_variables(load_path=load_path, variables=None, sess=sess)


if __name__ == "__main__":
    ob = tf.placeholder(shape=[None, None, 25],
                        dtype=tf.float32,
                        name='encoder_inputs')

    input_length_array = tf.placeholder(shape=[None],
                                       dtype=tf.int32,
                                       name='input_lengths')
    decoder_inputs= tf.placeholder(shape=[None, None],
                                   dtype=tf.int32,
                                   name='decoder_inputs')
    decoder_full_length = tf.placeholder(shape=[None, ],
                                         dtype=tf.int32,
                                         name='decoder_full_length')
    ac = tf.placeholder(shape=[None, None],
                        dtype=tf.int32,
                        name='decoder_targets')
    LR = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')

    pi = SelfAttentionSequencePolicy("pi", hparams=hparams, reuse=True, encoder_inputs=ob, encoder_lengths=input_length_array,
                       decoder_inputs=decoder_inputs, decoder_full_length=decoder_full_length, decoder_targets=ac)
    oldpi = SelfAttentionSequencePolicy("oldpi", hparams=hparams, reuse=False, encoder_inputs=ob, encoder_lengths=input_length_array,
                          decoder_inputs=decoder_inputs, decoder_full_length=decoder_full_length, decoder_targets=ac)


