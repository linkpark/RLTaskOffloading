import tensorflow as tf
import numpy as np

from rltaskoffloading.seq2seq import model_helper

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from tensorflow.python.ops.distributions import categorical


# keep decoder only
class ANNPolicy(object):
    def __init__(self,
                 name,
                 obs,
                 action,
                 hidden_units,
                 reuse,
                 action_dim,
                 ):

        self.obs = obs
        self.action = action
        self.action_dim = action_dim
        self.reuse = reuse

        with tf.variable_scope(name, reuse=self.reuse, initializer=tf.glorot_normal_initializer()):
            self.scope = tf.get_variable_scope().name
            hidden_logits = tf.layers.dense(self.obs, hidden_units, activation=tf.nn.relu,
                                        reuse=tf.AUTO_REUSE, name='hidden_1')

            hidden_logits = tf.layers.dense(hidden_logits, hidden_units, activation=tf.nn.relu,
                                        reuse=tf.AUTO_REUSE, name='hidden_2')

            self.out_put_logits = tf.layers.dense(hidden_logits, self.action_dim, activation=None, reuse=tf.AUTO_REUSE)
            self.pi = tf.nn.softmax(self.out_put_logits)
            self.q = tf.layers.dense(self.out_put_logits, self.action_dim, activation=None,
                                     reuse=tf.AUTO_REUSE, name="qvalue_layer")

            self.vf = tf.reduce_sum(self.pi * self.q, axis=-1)

            # define the sample action Categorical distribution.
            sample_id_sampler = categorical.Categorical(logits=self.out_put_logits)
            self.sample_actions = sample_id_sampler.sample()

            self.action_embeddings = tf.one_hot(self.sample_actions,
                                                self.action_dim,
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
                self.obs : obs
             }
        )

        sample_actions = np.array(sample_actions)
        sample_vf = np.array(sample_vf)
        sample_neglogp = np.array(sample_neglogp)

        return sample_actions, sample_vf, sample_neglogp

    def greedy_predict(self, obs):
        sess = tf.get_default_session()
        greedy_actions = sess.run(self.greedy_actions, feed_dict={
            self.obs: obs
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


if __name__ == "__main__":
    action = tf.placeholder(tf.int32, [None, None])
    obs = tf.placeholder(dtype=tf.float32, shape=[None, None, 25])
    ob_numpy = np.random.random(size=(5, 5, 25))

    ann_policy = ANNPolicy("pi",
                            obs,
                            action,
                            hidden_units=256,
                            reuse=True,
                            action_dim=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sample_actions, sample_vf, sample_neglogp = ann_policy.step(ob_numpy)


