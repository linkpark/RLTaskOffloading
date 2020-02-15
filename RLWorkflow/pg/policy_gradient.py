import tensorflow as tf
import numpy as np
import RLWorkflow.logger as logger
import RLWorkflow.pg.tf_util as U

from RLWorkflow.pg.seq2seq_policy import Seq2seqPolicy
from RLWorkflow.environment.old_workflow_env import WorkflowEnvironment
import RLWorkflow.seq2seq.seq2seq_hyparam as hparams
from RLWorkflow.common.dataset import Dataset

logger.configure("../log", ["stdout", "json", "csv"])

def traj_segment_generator(pi, env, horizon):
    sess = tf.get_default_session()
    t = 0

    encoder_inputs = []
    decoder_inputs = []
    sample_preds = []
    goal_sets = []
    decoder_full_lengths = []
    task_graph_batchs = []

    while True:
        for encoder_input, decoder_full_length, task_graph_batch in zip(env.encoder_batchs, env.decoder_full_lengths,
                                                                        env.task_graphs):
            sample_actions, vpreds = pi.sample_from_current_policy(sess, encoder_input, decoder_full_length)
            sample_pred = np.array(sample_actions).swapaxes(0, 1)
            decoder_input = np.column_stack(
                (np.ones(sample_pred.shape[0], dtype=int) * env.start_symbol, sample_pred[:, 0:-1]))

            goal_set = env.get_goal_batch_step_by_step(sample_pred, task_graph_batch)

            # timemajor change to batch major
            if env.time_major == True:
                encoder_input = np.array(encoder_input).swapaxes(0, 1)

            encoder_inputs += encoder_input.tolist()
            decoder_inputs += decoder_input.tolist()
            sample_preds += sample_pred.tolist()
            goal_sets += goal_set
            decoder_full_lengths += decoder_full_length.tolist()
            task_graph_batchs += task_graph_batch

            t += encoder_input.shape[0]

        if t % horizon == 0 and t >0:
            t = 0
            tr = {"encoder_input": encoder_inputs,
                  "decoder_input": decoder_inputs,
                  "decoder_target": sample_preds,
                  "decoder_reward": goal_sets,
                  "decoder_sequence_length": decoder_full_lengths,
                  "task_graph": task_graph_batchs}

            encoder_inputs = []
            decoder_inputs = []
            sample_preds = []
            goal_sets = []
            decoder_full_lengths = []
            task_graph_batchs = []

            yield tr

def policy_gradient_avgbaseline(gamma, optim_stepsize,
                 adam_epsilon=1e-5, max_episodes=0, max_timesteps=0, max_grad_norm=1.0,
                 optim_epochs=0, optim_batchsize=0, schedule='constant'):
    env = WorkflowEnvironment(batch_size=100, graph_number=100, graph_file_path="../data/random30/random.30.", is_xml=False, time_major=True)
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

    pi = Seq2seqPolicy(name="pg_policy", hparams=seq2seq_hparams, reuse=True, encoder_inputs=ob,
                       decoder_inputs=decoder_inputs, decoder_full_length=decoder_full_length, decoder_targets=ac)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None, None])  # target advantage function
    lr = tf.placeholder(name='learning_rate', dtype=tf.float32, shape=[])

    loss = tf.reduce_mean( pi.neglogp() * atarg )
    var_list = pi.get_trainable_variables()
    grads = tf.gradients(loss, var_list)

    if max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

    grads = list(zip(grads, var_list))
    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    _train = trainer.apply_gradients(grads)

    compute_losses = U.function([ob, decoder_inputs, decoder_full_length, ac, atarg, lr], [loss, _train])

    U.initialize()

    tr_gen = traj_segment_generator(pi, env, horizon=1000)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    goal_track = []

    while True:
        if max_episodes and episodes_so_far >= max_episodes:
            break

        if schedule == 'constant':
            cur_lrmult = 1e-5
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)
        # generate training data
        print("Generate training data!")
        tr = tr_gen.__next__()
        print("Finish training data!")

        atarg = np.array(tr["decoder_reward"]) - np.mean(np.array(tr["decoder_reward"]), axis=0)

        d = Dataset(dict(encoder_input=np.array(tr["encoder_input"]),
                         decoder_input=np.array(tr["decoder_input"]),
                         decoder_target=np.array(tr["decoder_target"]),
                         decoder_full_length=np.array(tr["decoder_sequence_length"]),
                         task_graph_batch=np.array(tr["task_graph"]),
                         atarg=atarg), deterministic=False, shuffle=False)

        logger.log("Optimizing...")

        losses = []
        for batch in d.iterate_once(optim_batchsize):
            decoder_full_length = batch["decoder_full_length"]
            if env.time_major:
                encoder_input = np.array(batch["encoder_input"]).swapaxes(0, 1)
                decoder_input = np.array(batch["decoder_input"]).swapaxes(0, 1)
                decoder_target = np.array(batch["decoder_target"]).swapaxes(0, 1)
                atarg = np.array(batch["atarg"]).swapaxes(0, 1)

            *newlosses, _ = compute_losses(encoder_input, decoder_input, decoder_full_length, decoder_target, atarg,
                                            cur_lrmult)

            losses.append(newlosses)
            logger.info("avg loss is: {}".format(newlosses))

        logger.record_tabular("Mean loss", np.mean(losses))

        sess = tf.get_default_session()

        # time major
        test_encoder_input = env.encoder_batchs[0]
        test_decoder_full_length = env.decoder_full_lengths[0]
        test_task_graph_batch = env.task_graphs[0]

        sample_action, _ = pi.sample_from_current_policy(sess, test_encoder_input, test_decoder_full_length)
        sample_action = np.array(sample_action).swapaxes(0, 1)
        print("sample action set: {}".format(sample_action[0]))

        cost = env.get_running_cost_batch(sample_action, test_task_graph_batch)
        logger.record_tabular("Mean makespan", np.mean(cost))
        heft_cost = env.get_heft_cost_by_graph_batch(test_task_graph_batch)
        logger.record_tabular("Mean HEFT Makespan", np.mean(heft_cost))

        logger.dump_tabular()

        goal_track.append(-np.mean(cost))

        episodes_so_far += 1
        iters_so_far += 1

    return goal_track

def main():
    sess = U.single_threaded_session()
    sess.__enter__()

    goal_track = policy_gradient_avgbaseline(gamma=0.99, optim_stepsize=5e-3,
                                            max_episodes=6000, max_timesteps=100, optim_epochs=4, optim_batchsize=1000,
                                            adam_epsilon=1e-6, schedule='constant')

    x = np.arange(0, len(goal_track), 1)

    print("Maxmium goal is {}".format(np.max(goal_track)))

    import matplotlib.pyplot as plt
    plt.plot(x, goal_track)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.show()

if __name__ == "__main__":
    main()