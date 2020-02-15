import tensorflow as tf

from RLWorkflow.ppo3.onmt_s2s_policy import SelfAttentionSequencePolicy
from RLWorkflow.ppo3.seq2seq_policy import Seq2seqPolicy

def build_policy(env, network, hparameters):
    # TODO: network type choice
    ob = tf.placeholder(dtype=tf.float32, shape=[None, None, env.input_dim])
    ob_length = tf.placeholder(dtype=tf.int32, shape=[None])

    def policy_fn(decoder_inputs,
                  decoder_full_length,
                  decoder_targets ):
        policy = Seq2seqPolicy("pi", hparameters, reuse=tf.AUTO_REUSE,
                                     encoder_inputs=ob,
                                     encoder_lengths=ob_length,
                                     decoder_inputs=decoder_inputs,
                                     decoder_full_length=decoder_full_length,
                                     decoder_targets=decoder_targets)

        # policy = SelfAttentionSequencePolicy("pi", hparameters, reuse=tf.AUTO_REUSE,
        #                        encoder_inputs=ob,
        #                        encoder_lengths=ob_length,
        #                        decoder_inputs=decoder_inputs,
        #                        decoder_full_length=decoder_full_length,
        #                        decoder_targets=decoder_targets)

        return policy
    return policy_fn


