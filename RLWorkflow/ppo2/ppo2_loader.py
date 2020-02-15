import numpy as np
import tensorflow as tf
import RLWorkflow.common.tf_util as U
import RLWorkflow.seq2seq.seq2seq_hyparam as hparams
import time

from RLWorkflow import logger

from RLWorkflow.ppo2.seq2seq_policy import Seq2seqPolicy
from RLWorkflow.environment.workflow_env import WorkflowEnvironment
from RLWorkflow.common.misc_util import zipsame
from RLWorkflow.common.console_util import fmt_row
from RLWorkflow.common.mpi_moment import mpi_moments
from RLWorkflow.common.dataset import Dataset


if __name__ == "__main__":

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

    env = WorkflowEnvironment(batch_size=100, graph_number=100,
                              graph_file_path="../data/CCR_DATA_20/CCR0.3/random.20.",
                              is_xml=False, time_major=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/workflow_model.ckpt')
        test_encoder_input = env.encoder_batchs[0]
        test_decoder_full_length = env.decoder_full_lengths[0]
        test_task_graph_batch = env.task_graphs[0]

        # greedy search
        costs = []
        for i in range(10):
            sample_action, _ = pi.sample_from_current_policy(sess, test_encoder_input, test_decoder_full_length)
            sample_action = np.array(sample_action).swapaxes(0, 1)
            cost = env.get_running_cost_batch(sample_action, test_task_graph_batch)
            costs.append(cost)
        min_cost = np.min(np.array(costs), axis=0)
        logger.record_tabular("Mean greedy search Makespan", np.mean(min_cost))

        # dump the tabular
        logger.dump_tabular()
