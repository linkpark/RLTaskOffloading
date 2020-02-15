import numpy as np
import tensorflow as tf
import RLWorkflow.ppo2.tf_util as U
import RLWorkflow.seq2seq.seq2seq_hyparam as hparams
import time

from RLWorkflow import logger

from RLWorkflow.ppo2.seq2seq_policy import Seq2seqPolicy
from RLWorkflow.environment.workflow_env import WorkflowEnvironment
from RLWorkflow.common.misc_util import zipsame
from RLWorkflow.common.console_util import fmt_row
from RLWorkflow.common.mpi_moment import mpi_moments
from RLWorkflow.common.dataset import Dataset

logger.configure('./log', ['stdout', 'json', 'csv'])

def traj_segment_generator(pi, env, horizon):
    sess = tf.get_default_session()
    t = 0

    encoder_inputs = []
    decoder_inputs = []
    sample_preds = []
    reward_sets = []
    value_preds = []
    decoder_full_lengths = []
    task_graph_batchs = []
    ep_rets = []

    while True:
        for encoder_input, decoder_full_length, task_graph_batch, max_running_time_batch, min_running_time_batch \
                            in zip(env.encoder_batchs, env.decoder_full_lengths, env.task_graphs, env.max_running_time_batchs,
                                   env.min_running_time_batchs):
            sample_actions, vpreds = pi.sample_from_current_policy(sess, encoder_input, decoder_full_length)
            sample_pred = np.array(sample_actions).swapaxes(0, 1)
            decoder_input = np.column_stack(
                (np.ones(sample_pred.shape[0], dtype=int) * env.start_symbol, sample_pred[:, 0:-1]))

            vpreds = np.array(vpreds).swapaxes(0, 1)
            reward_set = env.get_reward_batch_step_by_step(sample_pred, task_graph_batch,
                                                           max_running_time_batch=max_running_time_batch, min_running_time_batch=min_running_time_batch)
            cur_ep_ret = np.sum(reward_set, axis=-1)

            # timemajor change to batch major
            if env.time_major == True:
                encoder_input = np.array(encoder_input).swapaxes(0, 1)

            encoder_inputs += encoder_input.tolist()
            decoder_inputs += decoder_input.tolist()
            sample_preds += sample_pred.tolist()
            reward_sets += reward_set.tolist()
            value_preds += vpreds.tolist()
            decoder_full_lengths += decoder_full_length.tolist()
            task_graph_batchs += task_graph_batch
            ep_rets += cur_ep_ret.tolist()

            t += encoder_input.shape[0]

        if t % horizon == 0 and t >0:
            t = 0
            tr = {"encoder_input": encoder_inputs,
                  "decoder_input": decoder_inputs,
                  "decoder_target": sample_preds,
                  "decoder_reward": reward_sets,
                  "value_predict": value_preds,
                  "decoder_sequence_length": decoder_full_lengths,
                  "task_graph": task_graph_batchs,
                  "ep_rets": ep_rets}


            encoder_inputs = []
            decoder_inputs = []
            sample_preds = []
            reward_sets = []
            value_preds = []
            decoder_full_lengths = []
            task_graph_batchs = []
            ep_rets = []

            yield tr


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    vpred_batch = np.array(seg["value_predict"])
    rew_batch = np.array(seg["decoder_reward"])
    time_length = vpred_batch.shape[1]
    batch_size = vpred_batch.shape[0]

    vpred_batch = np.column_stack((vpred_batch, np.zeros(batch_size, dtype=float)))
    last_gae_lam = np.zeros(batch_size, dtype=float)
    tdlamret = []
    adv = []

    for t in reversed(range(time_length)):
        delta = rew_batch[:, t] + gamma * vpred_batch[:, t+1] - vpred_batch[:, t]
        gaelam = last_gae_lam = delta + gamma * lam * last_gae_lam
        adv.append(gaelam)
        tdlam = vpred_batch[:, t+1] + gaelam
        tdlamret.append(tdlam)

    tdlamret.reverse()
    adv.reverse()

    tdlamret = np.array(tdlamret).swapaxes(0,1)
    adv = np.array(adv).swapaxes(0, 1)

    seg["tdlamret"] = tdlamret
    seg["adv"] = adv
    #return tdlamret, adv


# ppo learning for GPU version (donot use mpiadm)
def ppo_learning(env, clip_param, entcoeff, gamma, optim_stepsize,
                 lam, adam_epsilon=1e-5, max_grad_norm=1.0 ,
                 max_episodes=0, max_timesteps=0, optim_epochs=0, optim_batchsize=0, schedule='constant'):
    seq2seq_hparams = hparams.get_hparams()

    ob = tf.placeholder(shape=[None, None, 25],
                        dtype=tf.float32,
                        name='encoder_inputs')
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

    pi = Seq2seqPolicy("pi", hparams=seq2seq_hparams, reuse=True, encoder_inputs=ob,
                       decoder_inputs=decoder_inputs, decoder_full_length=decoder_full_length, decoder_targets=ac)
    oldpi = Seq2seqPolicy("oldpi", hparams=seq2seq_hparams, reuse=False, encoder_inputs=ob,
                          decoder_inputs=decoder_inputs, decoder_full_length=decoder_full_length, decoder_targets=ac)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])

    atarg = tf.placeholder(dtype=tf.float32, shape=[None, None]) # target advantage function
    ret = tf.placeholder(dtype=tf.float32, shape=[None, None])  # Empirical return

    kloldnew = oldpi.kl(pi)
    ent = pi.entropy()

    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.logp() - oldpi.logp()) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    vpredclipped = oldpi.v + tf.clip_by_value(pi.v - oldpi.v, -clip_param, clip_param)
    vf_losses1 = tf.square(pi.v - ret)

    vf_losses2 = tf.square(vpredclipped - ret)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    grads = tf.gradients(total_loss, var_list)

    if max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

    grads = list(zip(grads, var_list))
    trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=adam_epsilon)
    _train = trainer.apply_gradients(grads)

    compute_losses_upgrade = U.function([ob, decoder_inputs, decoder_full_length, ac, atarg, ret, LR], losses + [_train])
    compute_losses = U.function([ob, decoder_inputs, decoder_full_length, ac, atarg, ret], losses)

    U.initialize()
    saver = tf.train.Saver()
    tr_gen = traj_segment_generator(pi, env, horizon=2000)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    goal_track = []
    reward_track = []

    while True:
        if max_episodes and episodes_so_far >= max_episodes:
            break
        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        cur_lr = cur_lrmult * optim_stepsize
        start_time = time.time()
        logger.log("********** Iteration %i ************" % iters_so_far)
        # generate training data
        print("Generate training data!")
        tr = tr_gen.__next__()
        print("Finish training data!")
        add_vtarg_and_adv(tr, gamma, lam)
        atarg = tr["adv"]
        atarg = (atarg - np.mean(atarg, axis=0)) / np.std(atarg, axis=0)

        d = Dataset(dict(encoder_input=np.array(tr["encoder_input"]),
                         decoder_input=np.array(tr["decoder_input"]),
                         decoder_target=np.array(tr["decoder_target"]),
                         decoder_full_length=np.array(tr["decoder_sequence_length"]),
                         task_graph_batch=np.array(tr["task_graph"]),
                         atarg=atarg,
                         vtarg=np.array(tr["tdlamret"])), deterministic=False, shuffle=False)

        assign_old_eq_new()
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        for _ in range(optim_epochs):
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                decoder_full_length = batch["decoder_full_length"]
                if env.time_major:
                    encoder_input = np.array(batch["encoder_input"]).swapaxes(0, 1)
                    decoder_input = np.array(batch["decoder_input"]).swapaxes(0, 1)
                    decoder_target = np.array(batch["decoder_target"]).swapaxes(0, 1)
                    atarg = np.array(batch["atarg"]).swapaxes(0, 1)
                    tdlamret = np.array(batch["vtarg"]).swapaxes(0, 1)


                #print("the value target is: ", batch["atarg"][0])

                *newlosses, _ = compute_losses_upgrade(encoder_input, decoder_input, decoder_full_length, decoder_target, atarg,
                                            tdlamret, cur_lr)


                losses.append(newlosses)

            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            decoder_full_length = batch["decoder_full_length"]
            if env.time_major:
                encoder_input = np.array(batch["encoder_input"]).swapaxes(0, 1)
                decoder_input = np.array(batch["decoder_input"]).swapaxes(0, 1)
                decoder_target = np.array(batch["decoder_target"]).swapaxes(0, 1)
                atarg = np.array(batch["atarg"]).swapaxes(0, 1)
                tdlamret = np.array(batch["vtarg"]).swapaxes(0, 1)
            newlosses= compute_losses(encoder_input, decoder_input, decoder_full_length, decoder_target, atarg,
                                           tdlamret)
            losses.append(newlosses)

        meanlosses, _, _ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))

        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)

        sess = tf.get_default_session()

        # time major
        test_encoder_input = env.encoder_batchs[0]
        test_decoder_full_length = env.decoder_full_lengths[0]
        test_task_graph_batch = env.task_graphs[0]

        sample_action, _ = pi.sample_from_current_policy(sess, test_encoder_input, test_decoder_full_length)
        sample_action = np.array(sample_action).swapaxes(0, 1)

        cost = env.get_running_cost_batch(sample_action, test_task_graph_batch)
        logger.record_tabular("Mean makespan", np.mean(cost))

        greedy_action= pi.greedy_predict(sess, test_encoder_input, test_decoder_full_length)
        greedy_action = np.array(greedy_action).swapaxes(0, 1)
        cost = env.get_running_cost_batch(greedy_action, test_task_graph_batch)
        logger.record_tabular("Mean greedy makespan", np.mean(cost))

        heft_cost = env.get_heft_cost_by_graph_batch(test_task_graph_batch)
        logger.record_tabular("Mean HEFT Makespan", np.mean(heft_cost))

        #greedy search
        costs = []
        for i in range(10):
            sample_action, _ = pi.sample_from_current_policy(sess, test_encoder_input, test_decoder_full_length)
            sample_action = np.array(sample_action).swapaxes(0, 1)
            cost = env.get_running_cost_batch(sample_action, test_task_graph_batch)
            costs.append(cost)
        min_cost = np.min(np.array(costs), axis=0)
        logger.record_tabular("Mean greedy search Makespan", np.mean(min_cost))

        #print("trajectories episode returns is {}".format(np.array(tr["ep_rets"]).shape))
        sample_episode_rets = np.array(tr["ep_rets"])
        logger.record_tabular("Mean Reward", np.mean(sample_episode_rets))

        # dump the tabular
        logger.dump_tabular()

        goal_track.append(-np.mean(cost))
        reward_track.append(np.mean(sample_episode_rets))

        end_time = time.time()
        logger.info("Iteration time cost is {}s".format(end_time - start_time))

        episodes_so_far += 1
        iters_so_far += 1

        if episodes_so_far % 100 == 0:
            save_path = saver.save(sess, "./model/workflow_model.ckpt")
            print("model saved in path: %s" % save_path)


    return goal_track, reward_track


if __name__ == "__main__":
    sess = U.single_threaded_session()
    sess.__enter__()


    env = WorkflowEnvironment(batch_size=100, graph_number=100, graph_file_path="../data/random20/random.20.",
                              is_xml=False, time_major=True)


    goal_track, reward_track = ppo_learning(env=env, clip_param=0.2, entcoeff=0.01, gamma=0.99, optim_stepsize=5e-4, lam=0.95,
                                           max_episodes=6000, max_timesteps=100, optim_epochs=4, optim_batchsize=500,
                                           adam_epsilon=1e-5, schedule='constant')

    x = np.arange(0, len(goal_track), 1)

    print("Maxmium goal is {}".format(np.max(goal_track)))


    import matplotlib.pyplot as plt
    plt.plot(x, goal_track)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.show()

    x = np.arange(0, len(reward_track), 1)

    print("Maxmium episode reward is {}".format(np.max(reward_track)))

    plt.plot(x, reward_track)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
