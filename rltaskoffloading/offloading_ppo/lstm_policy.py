import tensorflow as tf
import numpy as np

from rltaskoffloading.seq2seq import model_helper

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from tensorflow.python.ops.distributions import categorical


# keep decoder only
class LSTMPolicy(object):
    def __init__(self,
                 name,
                 hparams,
                 reuse,
                 encoder_inputs,
                 ):

        self.encoder_hidden_unit = hparams.num_units

        self.encoder_inputs = encoder_inputs
        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers

        self.time_major = hparams.time_major

        self.n_features = hparams.n_features
        self.reuse = reuse
        self.mode = tf.contrib.learn.ModeKeys.TRAIN
        self.single_cell_fn = None

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

            # define the lstm encoder
            self.encoder_outputs, self.encoder_state = self.create_encoder(hparams)

            self.out_put_logits = tf.contrib.layers.fully_connected(self.encoder_outputs,
                                                                    self.n_features,
                                                                    activation_fn=None,
                                                                    scope="output_layer",
                                                                    reuse=tf.AUTO_REUSE)

            self.pi = tf.nn.softmax(self.out_put_logits)
            self.q = tf.layers.dense(self.out_put_logits, self.n_features, activation=None,
                                     reuse=tf.AUTO_REUSE, name="qvalue_layer")

            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)

            # define the sample action Categorical distribution.
            sample_id_sampler = categorical.Categorical(logits=self.out_put_logits)
            self.sample_actions = sample_id_sampler.sample()

            self.action_embeddings = tf.one_hot(self.sample_actions,
                                                self.n_features,
                                                dtype=tf.float32)

            self.sample_neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.action_embeddings,
                                                                             logits=self.out_put_logits)

            # define the greedy action
            self.greedy_actions = tf.argmax(self.out_put_logits, axis=-1, output_type=tf.int32)

    def step(self, obs):
        sess = tf.get_default_session()
        sample_actions, sample_vf, sample_neglogp = sess.run(
            [self.sample_actions, self.vf, self.sample_neglogp],
             feed_dict={
                self.encoder_inputs : obs
             }
        )

        sample_actions = np.array(sample_actions)
        sample_vf = np.array(sample_vf)
        sample_neglogp = np.array(sample_neglogp)

        return sample_actions, sample_vf, sample_neglogp

    def greedy_predict(self, obs):
        sess = tf.get_default_session()
        greedy_actions = sess.run(self.greedy_actions, feed_dict={
            self.encoder_inputs: obs
        })

        greedy_actions = np.array(greedy_actions)

        return greedy_actions

    def kl(self, other):
        a0 = self.out_put_logits - tf.reduce_max(self.out_put_logits, axis=-1, keepdims=True)
        a1 = other.out_put_logits - tf.reduce_max(other.out_put_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0

        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.out_put_logits - tf.reduce_max(self.out_put_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def neglogp(self):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.out_put_logits,
            labels=self.action_embeddings)

    def logp(self):
        return -self.neglogp()

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


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
                inputs=self.encoder_embeddings,
                dtype=tf.float32,
                time_major=self.time_major,
                swap_memory=True,
                scope=scope
            )

        return encoder_outputs, encoder_state



if __name__ == "__main__":
    action = tf.placeholder(tf.int32, [None, None])
    obs = tf.placeholder(dtype=tf.float32, shape=[None, None, 25])
    ob_numpy = np.random.random(size=(5, 5, 25))

    hparams = tf.contrib.training.HParams(
        unit_type="lstm",
        num_units=256,
        learning_rate=0.00005,
        supervised_learning_rate=0.00005,
        n_features=2,
        time_major=False,
        is_attention=False,
        forget_bias=1.0,
        dropout=0,
        num_gpus=1,
        num_layers=2,
        num_residual_layers=0,
        is_greedy=False,
        inference_model="sample",
        start_token=0,
        end_token=5,
        is_bidencoder=False
    )

    ann_policy = LSTMPolicy(name='pi',
                             hparams=hparams,
                             reuse=True,
                             encoder_inputs=obs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sample_actions, sample_vf, sample_neglogp = ann_policy.step(ob_numpy)

        print(sample_actions, sample_vf, sample_neglogp)


