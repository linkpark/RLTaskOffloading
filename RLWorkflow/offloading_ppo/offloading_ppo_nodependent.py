import tensorflow as tf
import numpy as np
import itertools
import functools
import time

from mpi4py import MPI

from RLWorkflow.common.mpi_adam_optimizer import MpiAdamOptimizer
from RLWorkflow.common.tf_util import get_session, save_variables, load_variables, initialize
import RLWorkflow.common.tf_util as U
from RLWorkflow.common.mpi_util import sync_from_root
from RLWorkflow.common.console_util import fmt_row
from RLWorkflow.offloading_ppo.ann_policy import ANNPolicy
from RLWorkflow import logger

from RLWorkflow.environment.offloading_env import OffloadingEnvironment
from RLWorkflow.environment.offloading_env import Resources

from RLWorkflow.common.dataset import Dataset
from RLWorkflow.common.misc_util import zipsame

class ANNPPOModel(object):
    def __init__(self, obs_dim, action_dim, hidden_units, ent_coef, vf_coef, max_grad_norm):
        sess = get_session()
        # sequential state

        # sequential action
        obs = tf.placeholder(tf.float32, [None, None, obs_dim])
        action = tf.placeholder(tf.int32, [None, None])
        # sequential adv
        adv = tf.placeholder(tf.float32, [None, None])
        # sequential return
        ret = tf.placeholder(tf.float32, [None, None])

        # keep track of old actor(sequential descision)
        oldneglogpac = tf.placeholder(tf.float32, [None, None])
        oldvpred = tf.placeholder(tf.float32, [None, None])
        lr = tf.placeholder(tf.float32, [])

        # Cliprange
        cliprange = tf.placeholder(tf.float32, [])

        train_model = ANNPolicy("pi", obs, action, hidden_units=hidden_units, reuse=True, action_dim=action_dim)
        act_model = ANNPolicy("oldpi", obs, action, hidden_units=hidden_units, reuse=False, action_dim=action_dim)

        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(act_model.get_variables(),
                                                                train_model.get_variables())])

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.entropy())

        vpred = train_model.vf
        vf_losses1 = tf.square(vpred - ret)
        vf_loss = tf.reduce_mean(vf_losses1)
        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(train_model.logp() - act_model.logp())

        # define the loss = -J is equivalent to max J
        pg_losses = -adv * ratio
        pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - cliprange, 1.0 + cliprange)

        # Final pg loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - oldneglogpac))
        kloldnew = act_model.kl(train_model)
        approxkl = tf.reduce_mean(kloldnew)
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), cliprange)))

        # total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # Update the parameters using loss
        # 1. get the model parameters
        params = tf.trainable_variables('pi')

        # 2. Build our trainer
        trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=lr, epsilon=1e-5)

        # 3. Calculate the gradients
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        _train = trainer.apply_gradients(grads_and_var)

        # decoder_input action_length is speciallized for the training model
        def train(learning_reate, clipingrange, obs,
                  returns, advs, actions, values, neglogpacs, states=None):
            # the advantage function is calculated as A(s,a) = R + yV(s') - V(s)
            # the return = R + yV(s')

            # Sequential Normalize the advantages
            advs = (advs - np.mean(advs, axis=0)) / (np.std(advs, axis=0) + 1e-8)

            td_map = {obs: obs, action: actions,  adv: advs, ret: returns, lr: learning_reate,
                      cliprange: clipingrange, oldneglogpac: neglogpacs, oldvpred: values}

            return sess.run([pg_loss, vf_loss, entropy, approxkl, clipfrac, _train], td_map)[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.greedy_predict = act_model.greedy_predict

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        self.assign_old_eq_new = assign_old_eq_new

        if MPI.COMM_WORLD.Get_rank() == 0:
            initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        sync_from_root(sess, global_variables)

class Runner():
    def __init__(self, env, model, nepisode, gamma, lam):
        self.lam = lam
        self.gamma = gamma
        self.model = model
        self.nepisode = nepisode
        self.env = env

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_tdlamret, mb_adv = [], []
        mb_decoder_input = []
        mb_decoder_length = []
        mb_encoder_batch = []
        mb_encoder_length = []
        mb_task_graph = []

        for task_graph_batch, encoder_batch, encoder_length, \
            decoder_lengths, max_running_time, min_running_time in zip(self.env.task_graphs,
                                                                       self.env.encoder_batchs,
                                                                       self.env.encoder_lengths,
                                                                       self.env.decoder_full_lengths,
                                                                       self.env.max_running_time_batchs,
                                                                       self.env.min_running_time_batchs):
            for _ in range(self.nepisode):
                actions, values, neglogpacs = self.model.step(encoder_input_batch=encoder_batch,
                                                              decoder_full_length=decoder_lengths,
                                                              encoder_lengths=encoder_length)

                mb_encoder_batch += encoder_batch.tolist()
                mb_encoder_length += encoder_length.tolist()
                actions = np.array(actions)
                values = np.array(values)
                neglogpacs = np.array(neglogpacs)

                decoder_input = np.column_stack(
                    (np.ones(actions.shape[0], dtype=int) * self.env.start_symbol, actions[:, 0:-1]))
                mb_decoder_input += decoder_input.tolist()
                mb_decoder_length += decoder_lengths.tolist()
                mb_actions += actions.tolist()
                mb_values += values.tolist()
                mb_neglogpacs += neglogpacs.tolist()

                rewards = self.env.step(task_graph_batch=task_graph_batch, action_sequence_batch=actions,
                                        max_running_time_batch=max_running_time,
                                        min_running_time_batch=min_running_time)

                mb_rewards += rewards.tolist()
                mb_task_graph += task_graph_batch

                time_length = values.shape[1]
                batch_size = values.shape[0]
                vpred_batch = np.column_stack((values, np.zeros(batch_size, dtype=float)))
                last_gae_lam = np.zeros(batch_size, dtype=float)
                tdlamret = []
                adv = []

                for t in reversed(range(time_length)):
                    delta = rewards[:, t] + self.gamma * vpred_batch[:, t + 1] - vpred_batch[:, t]
                    gaelam = last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
                    adv.append(gaelam)
                    tdlam = vpred_batch[:, t + 1] + gaelam
                    tdlamret.append(tdlam)

                # tdlamret.reverse()
                adv.reverse()
                #
                # tdlamret = np.array(tdlamret).swapaxes(0, 1)
                adv = np.array(adv).swapaxes(0, 1)
                #
                # mb_tdlamret += tdlamret.tolist()
                mb_adv += adv.tolist()

                import scipy
                import scipy.signal
                for reward in rewards:
                    mb_tdlamret.append(scipy.signal.lfilter([1], [1, float(-self.gamma)], reward[::-1], axis=0)[::-1])

        # return the trajectories
        return mb_encoder_batch, mb_encoder_length, mb_decoder_input, \
               mb_actions, mb_decoder_length, mb_values, mb_rewards, mb_neglogpacs, mb_tdlamret, mb_adv