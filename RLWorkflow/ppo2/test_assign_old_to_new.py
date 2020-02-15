import numpy as np
import tensorflow as tf
import os.path as osp
import mpi4py as MPI
import RLWorkflow.common.tf_util as U
import RLWorkflow.seq2seq.seq2seq_hyparam as hparams
import time

from RLWorkflow import logger

from RLWorkflow.environment.workflow_env import WorkflowEnvironment
from RLWorkflow.ppo2.seq2seq_policy import Seq2seqPolicy
from mpi4py import MPI
from RLWorkflow.environment.workflow_env import WorkflowEnvironment
from RLWorkflow.common.mpi_adam import MpiAdam
from RLWorkflow.common.misc_util import zipsame
from RLWorkflow.common.console_util import fmt_row
from RLWorkflow.common.mpi_moment import mpi_moments
from RLWorkflow.common.dataset import Dataset

if __name__ == "__main__":
    env = WorkflowEnvironment(batch_size=100, graph_number=100, time_major=True)
    seq2seq_hparams = hparams.get_hparams()
    ob = tf.placeholder(shape=[None, None, 25],
                        dtype=tf.float32,
                        name='encoder_inputs')
    decoder_inputs = tf.placeholder(shape=[None, None],
                                    dtype=tf.int32,
                                    name='decoder_inputs')
    decoder_full_length = tf.placeholder(shape=[None, ],
                                         dtype=tf.int32,
                                         name='decoder_full_length')
    ac = tf.placeholder(shape=[None, None],
                        dtype=tf.int32,
                        name='decoder_targets')
    LR = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')

    pi = Seq2seqPolicy("pi", hparams=seq2seq_hparams, reuse=True, encoder_inputs=ob,
                       decoder_inputs=decoder_inputs, decoder_full_length=decoder_full_length, decoder_targets=ac)
    oldpi = Seq2seqPolicy("oldpi", hparams=seq2seq_hparams, reuse=False, encoder_inputs=ob,
                          decoder_inputs=decoder_inputs, decoder_full_length=decoder_full_length, decoder_targets=ac)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])

    atarg = tf.placeholder(dtype=tf.float32, shape=[None, None])  # target advantage function

    total_loss = pi.neglogp() * atarg
    var_list = pi.get_trainable_variables()
    grads = tf.gradients(total_loss, var_list)
    grads = list(zip(grads, var_list))
    trainer = tf.train.AdamOptimizer(learning_rate=LR)
    _train = trainer.apply_gradients(grads)

    compute_losses_upgrade = U.function([ob, decoder_inputs, decoder_full_length, ac, atarg, LR],
                                        [total_loss] +[_train])

    sess = U.single_threaded_session()
    sess.__enter__()

    U.initialize()

    test_encoder_input = env.encoder_batchs[0]
    test_decoder_input = env.decoder_batchs[0]
    test_decoder_full_length = env.decoder_full_lengths[0]
    test_task_graph_batch = env.task_graphs[0]
    test_reward_target = np.ones((20, 100),dtype=np.float32)

    sample_action = pi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
    oldpi_sample_action = oldpi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)

    print("current policy: {}".format(np.array(sample_action).swapaxes(0,1)[0]))
    print("old policy: {}".format(np.array(oldpi_sample_action).swapaxes(0, 1)[0]))
    print()

    print("after asign")
    assign_old_eq_new()
    sample_action = pi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
    oldpi_sample_action = oldpi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
    print("current policy: {}".format(np.array(sample_action).swapaxes(0, 1)[0]))
    print("old policy: {}".format(np.array(oldpi_sample_action).swapaxes(0, 1)[0]))
    print()

    print("afater update")
    compute_losses_upgrade(test_encoder_input, test_decoder_input, test_decoder_full_length, test_decoder_input, test_reward_target, 1e-5)

    sample_action = pi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
    oldpi_sample_action = oldpi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
    print("current policy: {}".format(np.array(sample_action).swapaxes(0, 1)[0]))
    print("old policy: {}".format(np.array(oldpi_sample_action).swapaxes(0, 1)[0]))
    print()

    print("after assign")
    assign_old_eq_new()
    sample_action = pi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
    oldpi_sample_action = oldpi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
    print("current policy: {}".format(np.array(sample_action).swapaxes(0, 1)[0]))
    print("old policy: {}".format(np.array(oldpi_sample_action).swapaxes(0, 1)[0]))

