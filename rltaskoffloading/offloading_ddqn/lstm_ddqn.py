import tensorflow as tf
import numpy as np
import itertools
import functools
import time

from mpi4py import MPI

from rltaskoffloading.common.mpi_adam_optimizer import MpiAdamOptimizer
from rltaskoffloading.common.tf_util import get_session, save_variables, load_variables, initialize
import rltaskoffloading.common.tf_util as U
from rltaskoffloading.common.mpi_util import sync_from_root
from rltaskoffloading.common.schedule import LinearSchedule
from rltaskoffloading.common.console_util import fmt_row
from rltaskoffloading.offloading_ddqn.lstm_dqnet import LSTMDuelQnet
from rltaskoffloading import logger

from rltaskoffloading.environment.offloading_env import OffloadingEnvironment
from rltaskoffloading.environment.offloading_env import Resources

from rltaskoffloading.common.misc_util import zipsame
from rltaskoffloading.offloading_ddqn.seq2seq_replay_buffer import SeqReplayBuffer

def calculate_qoe(latency_batch, energy_batch, env):
    all_local_time, all_local_energy = env.get_all_locally_execute_time_batch()
    all_local_time = np.squeeze(all_local_time)
    all_local_energy = np.squeeze(all_local_energy)
    latency_batch = np.squeeze(latency_batch)
    energy_batch = np.squeeze(energy_batch)
    qoe_batch = []

    for latency, energy, single_all_local_latency, single_all_local_energy in zip(latency_batch, energy_batch, all_local_time, all_local_energy):
        qoe = env.lambda_t * ((latency - single_all_local_latency) / single_all_local_latency) + \
              env.lambda_e * ((energy - single_all_local_energy) / single_all_local_energy)

        qoe = -qoe
        qoe_batch.append(qoe)

    return qoe_batch

# LSTM + DuelQnets + Double Deep Q-learning
class LSTMDDQN(object):
    def __init__(self, hparams, ob_dim, gamma, max_grad_norm):
        # placeholder for qnet
        self.ob_input = tf.placeholder(dtype=tf.float32, shape=[None, None, ob_dim])

        self.action = tf.placeholder(tf.int32, [None, None])
        # placeholder for target q net
        self.target_ob_input = tf.placeholder(dtype=tf.float32, shape=[None, None, ob_dim])
        self.target_next_q = tf.placeholder(tf.float32, [None, None])
        self.target_action = tf.placeholder(tf.int32, [None, None])

        # sequential reward
        self.r = tf.placeholder(tf.float32, [None, None])
        self.lr = tf.placeholder(tf.float32, [])

        self.q_net = LSTMDuelQnet("qnet", hparams, reuse=True,
                            encoder_inputs=self.ob_input, actions=self.action
                            )

        self.target_q_net = LSTMDuelQnet("target_qnet", hparams, reuse=False,
                            encoder_inputs=self.target_ob_input, actions=self.target_action)

        self.update_target_q = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(self.target_q_net.get_trainable_variables(),
                                                                self.q_net.get_trainable_variables())])
        # define loss function here:
        # target q-network evaluation, this q value is estimiated by double deep q learning

        # define the double the loss function of double dqn
        q_selected_target = self.r + gamma * tf.stop_gradient(self.target_next_q)
        td_error = self.q_net.q - q_selected_target
        self.q_loss = tf.reduce_mean(U.huber_loss(td_error))

        params = self.q_net.get_trainable_variables()
        trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=self.lr, epsilon=1e-5)

        grads_and_var = trainer.compute_gradients(self.q_loss, params)
        grads, var = zip(*grads_and_var)

        # clip the gradients
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        grads_and_var = list(zip(grads, var))
        self._train = trainer.apply_gradients(grads_and_var)

        self.loss_names = ["q_loss"]
        self.step = self.q_net.step

    def train(self, sess, learning_rate, obs, rewards, actions,
              target_next_q_values):

        td_map = {self.lr: learning_rate,
                  self.ob_input: obs,
                  self.r: rewards,
                  self.action: actions,
                  self.target_next_q: target_next_q_values}

        return sess.run([self.q_loss, self._train], td_map)

    def save(self, model_path):
        sess = tf.get_default_session()
        functools.partial(save_variables, sess=sess)(model_path)

    def load(self, model_path):
        sess = tf.get_default_session()
        functools.partial(load_variables, sess=sess)(model_path)

class Runner(object):
    def __init__(self, env, model, nepisode, replay_buffers):
        self.env = env
        self.model = model
        self.nepisode = nepisode
        self.replay_buffers = replay_buffers

    def run(self, epsilon_threshold=0.1):
        assert self.replay_buffers is not None

        for task_graph_batch, encoder_batch, encoder_length, \
            decoder_lengths, max_running_time, min_running_time, replay_buffer in zip(self.env.task_graphs,
                                                                       self.env.encoder_batchs,
                                                                       self.env.encoder_lengths,
                                                                       self.env.decoder_full_lengths,
                                                                       self.env.max_running_time_batchs,
                                                                       self.env.min_running_time_batchs,
                                                                       self.replay_buffers):
            for _ in range(self.nepisode):
                sample_actions, greedy_actions = self.model.step(obs=encoder_batch,
                                                                 epsilon_threshold=epsilon_threshold)

                sample_decoder_input = np.column_stack(
                    (np.ones(sample_actions.shape[0], dtype=np.int32) * self.env.start_symbol, sample_actions[:, 0:-1]))

                greedy_decoder_input =  np.column_stack(
                    (np.ones(greedy_actions.shape[0], dtype=np.int32) * self.env.start_symbol, sample_actions[:, 0:-1]))

                target_next_q = self.model.target_q_net.get_qvalues(encoder_input_batch=encoder_batch,
                                                                    actions=greedy_actions)

                # get Q(S_{t+1}, argmax_{a}Q(S_{t+1}, a; \theta_t), \theta_t^-).
                target_next_q = np.column_stack(
                    (target_next_q[:, 1:], np.zeros(greedy_actions.shape[0], dtype=np.int32)))

                rewards = self.env.step(task_graph_batch=task_graph_batch, action_sequence_batch=sample_actions,
                                        max_running_time_batch=max_running_time,
                                        min_running_time_batch=min_running_time)

                replay_buffer.add_batch(batch_ob_seq=encoder_batch,
                                        batch_act_seq=sample_actions,
                                        batch_dec_seq= sample_decoder_input,
                                        batch_dec_length = decoder_lengths,
                                        batch_greedy_act_seq=greedy_actions,
                                        batch_greedy_dec_seq=greedy_decoder_input,
                                        batch_rew_seq=rewards,
                                        batch_target_next_q=target_next_q)

    def evaluate(self):
        running_cost = []
        energy_consumption = []
        running_qoe = []

        for encoder_batch, encoder_length, decoder_lengths, task_graph_batch \
                in zip(self.env.encoder_batchs, self.env.encoder_lengths,
                       self.env.decoder_full_lengths, self.env.task_graphs):
            _, actions = self.model.step(obs=encoder_batch)
            actions = np.array(actions)
            env_running_cost, env_energy_consumption = self.env.get_running_cost(action_sequence_batch=actions,
                                                                                 task_graph_batch=task_graph_batch)

            qoe = calculate_qoe(env_running_cost, env_energy_consumption, self.env)

            running_cost += env_running_cost
            energy_consumption += env_energy_consumption
            running_qoe += qoe

        return running_cost, energy_consumption, running_qoe

def ddqn_learning(env,
                  ddqn_model,
                  reply_buffer_num=9,
                  reply_buffer_size=10000,
                  eval_envs=None,
                  final_epsilon=0.02,
                  train_freq = 1,
                  target_freq = 10,
                  eval_freq = 10,
                  nupdates=1000,
                  nsample_episode=10,
                  warmup_episode=10,
                  lr=1e-4,
                  batch_size=500,
                  log_interval=1,
                  update_numbers=10,
                  load_path=None):
    sess = tf.get_default_session()

    replay_buffers = []
    for _ in range(reply_buffer_num):
        replay_buffers.append(SeqReplayBuffer(size=reply_buffer_size))

    runner = Runner(model=ddqn_model, env=env, nepisode=nsample_episode, replay_buffers=replay_buffers)

    eval_runners = []
    if eval_envs is not None:
        for eval_env in eval_envs:
            eval_runners.append(Runner(model=ddqn_model, env=eval_env, nepisode=1, replay_buffers=None))

    exploration = LinearSchedule(schedule_timesteps=nupdates//2, final_p=final_epsilon, initial_p=1.0)
    # warm up period
    for _ in range(warmup_episode):
        runner.run(epsilon_threshold=1.0)

    for update in range(1, nupdates+1):
        tstart = time.time()

        # sample trajectories from the envrionment
        runner.run(exploration.value(update))
        sample_time_cost = time.time()
        print("sample time cost: ", (sample_time_cost - tstart))

        if update % train_freq == 0:
            logger.log(fmt_row(13, ddqn_model.loss_names))
            for _ in range(update_numbers):
                batch_losses = []
                for replay_buffer in replay_buffers:
                    ob_seq, ac_seq, dec_input_seq, decoder_full_length, greedy_ac_seq, \
                    greedy_dec_input_seq, rew_seq, target_next_q = replay_buffer.random_sample(batch_size)

                    q_loss, _ = ddqn_model.train(sess, learning_rate=lr, obs=ob_seq,
                                  rewards=rew_seq,
                                  target_next_q_values=target_next_q,
                                  actions=ac_seq)

                    batch_losses.append(q_loss)

                logger.log(fmt_row(13, [np.mean(batch_losses, axis=0)]))
        update_time_cost = time.time()
        print("Update time cost: ", (update_time_cost - sample_time_cost))

        # sychronous the parameters between target q net and q net.
        if update % target_freq == 0:
            print("Update target q network")
            ddqn_model.update_target_q()

        if update % eval_freq == 0:
            # add the
            ddqn_model.save("./checkpoint/ddqn_offloading_model.ckpt")
            running_cost = []
            energy_consumption = []
            running_qoe = []
            #
            for eval_runner in eval_runners:
                Tc, Ec, qoe = eval_runner.evaluate()

                running_cost.append(np.mean(Tc))
                energy_consumption.append(np.mean(Ec))
                running_qoe.append(np.mean(qoe))

            j = 0
            for eval_env, run_time, energy, running_mean_qoe in zip(
                    eval_envs, running_cost,
                    energy_consumption,
                    running_qoe):
                logger.logkv(str(j) + 'th run time cost ', run_time)
                logger.logkv(str(j) + 'th energy cost ', energy)
                logger.logkv(str(j) + 'th qoe ', running_mean_qoe)

                logger.logkv(str(j) + 'th HEFT run time cost', eval_env.heft_avg_run_time)
                logger.logkv(str(j) + 'th HEFT energy cost', eval_env.heft_avg_energy)
                logger.logkv(str(j) + 'th HEFT qoe', eval_env.heft_avg_qoe)
                j += 1

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dumpkvs()

def DDQNTO_number(lambda_t = 1.0, lambda_e = 0.0, logpath="./log/DDQNTO-all-graph-LO",
           unit_type="layer_norm_lstm", num_units=256, learning_rate=0.00005, supervised_learning_rate=0.00005,
           n_features=2, time_major=False, is_attention=True, forget_bias=1.0, dropout=0, num_gpus=1,
           num_layers=2, num_residual_layers=0, is_greedy=False, encode_dependencies=False,
           inference_model="sample", start_token=0,
           end_token=5, is_bidencoder=True,
           train_graph_file_paths=["../offloading_data/offload_random10/random.10."],
           test_graph_file_paths=["../offloading_data/offload_random10_test/random.10."],
           batch_size=500, graph_number=500):

    logger.configure(logpath, ['stdout', 'json', 'csv'])

    hparams = tf.contrib.training.HParams(
        unit_type=unit_type,
        num_units=num_units,
        learning_rate=learning_rate,
        supervised_learning_rate=supervised_learning_rate,
        n_features=n_features,
        time_major=time_major,
        is_attention=is_attention,
        forget_bias=forget_bias,
        dropout=dropout,
        num_gpus=num_gpus,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        is_greedy=is_greedy,
        inference_model=inference_model,
        start_token=start_token,
        end_token=end_token,
        is_bidencoder=is_bidencoder
    )

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster, batch_size=batch_size, graph_number=graph_number,
                                graph_file_paths=train_graph_file_paths,
                                time_major=False,
                                lambda_t=lambda_t, lambda_e=lambda_e,
                                encode_dependencies=encode_dependencies)

    eval_envs = []
    for path in test_graph_file_paths:
        eval_env = OffloadingEnvironment(resource_cluster=resource_cluster, batch_size=100, graph_number=100,
                                           graph_file_paths=[path],
                                           time_major=False,
                                           lambda_t=lambda_t, lambda_e=lambda_e,
                                           encode_dependencies=encode_dependencies
                                           )

        eval_env.calculate_heft_cost()
        eval_envs.append(eval_env)
    print("Finishing initialization of environment")

    model = LSTMDDQN(hparams=hparams, ob_dim=env.input_dim, gamma=0.99, max_grad_norm=1.0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ddqn_learning(env=env,
                      eval_envs=eval_envs,
                      ddqn_model=model,
                      reply_buffer_num=1,
                      reply_buffer_size=100000,
                      final_epsilon=0.01,
                      train_freq=1,
                      target_freq=5,
                      nupdates=1000,
                      eval_freq=1,
                      warmup_episode=10,
                      nsample_episode=10,
                      lr=5e-4,
                      batch_size=500,
                      update_numbers=10,
                      load_path=None
                      )

def DDQNTO_trans(lambda_t = 1.0, lambda_e = 0.0, logpath="./log/all-graph-LO",
           unit_type="layer_norm_lstm", num_units=256, learning_rate=0.00005, supervised_learning_rate=0.00005,
           n_features=2, time_major=False, is_attention=True, forget_bias=1.0, dropout=0, num_gpus=1,
           num_layers=2, num_residual_layers=0, is_greedy=False, encode_dependencies=False,
           inference_model="sample", start_token=0,
           end_token=5, is_bidencoder=True,
           train_graph_file_paths=["../offloading_data/offload_random10/random.10."],
           test_graph_file_paths=["../offloading_data/offload_random10_test/random.10."],
           batch_size=500, graph_number=500,
           bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0]):

    logger.configure(logpath, ['stdout', 'json', 'csv'])

    hparams = tf.contrib.training.HParams(
        unit_type=unit_type,
        num_units=num_units,
        learning_rate=learning_rate,
        supervised_learning_rate=supervised_learning_rate,
        n_features=n_features,
        time_major=time_major,
        is_attention=is_attention,
        forget_bias=forget_bias,
        dropout=dropout,
        num_gpus=num_gpus,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        is_greedy=is_greedy,
        inference_model=inference_model,
        start_token=start_token,
        end_token=end_token,
        is_bidencoder=is_bidencoder
    )

    def test_case(bandwidth=5.0, log_path='./log', lambda_t=1.0, lambda_e=0.0, nupdates=2000):
        logger.configure(log_path, ['stdout', 'json', 'csv'])
        resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                     mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=bandwidth,
                                     bandwith_dl=bandwidth)

        env = OffloadingEnvironment(resource_cluster=resource_cluster, batch_size=batch_size, graph_number=graph_number,
                                    graph_file_paths=train_graph_file_paths,
                                    time_major=False,
                                    lambda_t=lambda_t, lambda_e=lambda_e,
                                    encode_dependencies=encode_dependencies # In DDQNTO, we don't consider the task dependency
                                    )

        # define the evaluation environment
        eval_envs = []
        eval_env_1 = OffloadingEnvironment(resource_cluster=resource_cluster, batch_size=100, graph_number=100,
                                           graph_file_paths=test_graph_file_paths,
                                           time_major=False,
                                           lambda_t=lambda_t, lambda_e=lambda_e,
                                           encode_dependencies=encode_dependencies
                                           )
        eval_env_1.calculate_heft_cost()

        eval_envs.append(eval_env_1)
        print("Finishing initialization of environment")

        # limit the number of cpu cores
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)

        with tf.Session(config=session_conf) as sess:
            model = LSTMDDQN(hparams=hparams, ob_dim=env.input_dim, gamma=0.99, max_grad_norm=1.0)
            sess.run(tf.global_variables_initializer())
            ddqn_learning(env=env,
                          eval_envs=eval_envs,
                          ddqn_model=model,
                          reply_buffer_num=1,
                          reply_buffer_size=100000,
                          final_epsilon=0.02,
                          train_freq=1,
                          target_freq=5,
                          nupdates=nupdates,
                          eval_freq=1,
                          warmup_episode=10,
                          nsample_episode=10,
                          lr=5e-4,
                          batch_size=500,
                          update_numbers=10,
                          load_path=None
                          )
            sess.close()

        tf.reset_default_graph()

    for bandwidth in bandwidths:
        test_case(bandwidth=bandwidth, lambda_t=lambda_t, lambda_e=lambda_e, log_path=logpath + '-' + str(bandwidth) +'Mbps')


if __name__ == "__main__":
    DDQNTO_number()
