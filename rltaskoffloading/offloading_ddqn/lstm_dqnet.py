import tensorflow as tf
import numpy as np

from rltaskoffloading.seq2seq import model_helper

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from tensorflow.python.ops.distributions import categorical


# keep decoder only
class LSTMDuelQnet(object):
    def __init__(self,
                 name,
                 hparams,
                 reuse,
                 encoder_inputs,
                 actions,
                 ):

        self.encoder_hidden_unit = hparams.num_units

        self.actions = actions

        self.encoder_inputs = encoder_inputs
        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers

        self.time_major = hparams.time_major

        self.n_features = hparams.n_features
        self.reuse = reuse
        self.mode = tf.contrib.learn.ModeKeys.TRAIN
        self.single_cell_fn = None

        self.epsilon_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)

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

            self.adv = tf.contrib.layers.fully_connected(self.out_put_logits, self.n_features,
                                                         activation_fn=None,
                                                         scope="advantages",
                                                         reuse=tf.AUTO_REUSE)
            self.vf = tf.contrib.layers.fully_connected(self.out_put_logits, 1,
                                                        activation_fn=None,
                                                        scope="state-values",
                                                        reuse=tf.AUTO_REUSE)

            self.actions_one_hot = tf.one_hot(self.actions,
                                             self.n_features,
                                             dtype=tf.float32)


            # create duel q-networks
            delta_adv = (1.0/self.n_features * tf.reduce_sum(self.adv, axis=-1, keepdims=True))

            self.q_logits = self.vf + (self.adv - delta_adv)

            self.greedy_actions = tf.argmax(self.q_logits, axis=-1, output_type=tf.int32)
            self.q = tf.reduce_sum(self.q_logits * self.actions_one_hot, axis=-1)

            # create random action
            logits = math_ops.multiply(math_ops.divide(0.5, (self.q_logits + 1e-5) ), (self.q_logits + 1e-5))
            sample_id_sampler = categorical.Categorical(logits=logits)
            self.sample_action = sample_id_sampler.sample()

            self.sample_action = self.greedy_actions * self.epsilon_mask + self.sample_action * (1-self.epsilon_mask)

    def step(self, obs, epsilon_threshold=1.0):
        sess = tf.get_default_session()

        epsilon = np.random.uniform(low=0.0, high=1.0, size=(obs.shape[0], obs.shape[1]))
        mask = np.array(epsilon < epsilon_threshold, np.int32)

        sample_actions, greedy_actions= sess.run(
            [self.sample_action, self.greedy_actions],
             feed_dict={
                self.encoder_inputs : obs,
                self.epsilon_mask: mask
             }
        )

        return sample_actions, greedy_actions

    def greedy_predict(self, obs):
        sess = tf.get_default_session()
        greedy_actions = sess.run(self.greedy_actions, feed_dict={
            self.encoder_inputs: obs
        })

        greedy_actions = np.array(greedy_actions)

        return greedy_actions

    def get_qvalues(self, encoder_input_batch, actions):
        sess = tf.get_default_session()

        q_values = sess.run(self.q,
                            feed_dict={
                                self.encoder_inputs: encoder_input_batch,
                                self.actions: actions
                            })

        return q_values

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
    ob_length_numpy = np.array([5] * 5)

    with tf.Session() as sess:
        decoder_input = tf.placeholder(tf.int32, [None, None])
        action = tf.placeholder(tf.int32, [None, None])
        action_length = tf.placeholder(tf.int32, [None])

        ob = tf.placeholder(dtype=tf.float32, shape=[None, None, 25])
        ob_length = tf.placeholder(dtype=tf.int32, shape=[None])

        train_model = LSTMDuelQnet("q-value", hparams, reuse=True, encoder_inputs=ob, actions=action)

        sess.run(tf.global_variables_initializer())

        sample_actions, greedy_actions = train_model.step(obs=ob_numpy, epsilon_threshold = 1.0)

        print("sample actions: ", sample_actions)
        print("greedy actions: ", greedy_actions)

        q_values = train_model.get_qvalues(encoder_input_batch=ob_numpy, actions=greedy_actions)
        print("q_values: ", q_values)