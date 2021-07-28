import numpy as np
import random


class SeqReplayBuffer(object):
    def __init__(self, size):
        """ Create the replay buffer to store the sequence data

        """
        self._observation_sequence = []
        self._action_sequence = []
        self._decoder_input_sequence = []
        self._decoder_full_length = []
        self._greedy_action_sequence = []
        self._greedy_decoder_input_sequence = []
        self._reward_sequence = []
        self._target_next_q = []

        self._maxsize = size
        self._next_idx = 0

    def add_batch(self, batch_ob_seq, batch_act_seq, batch_dec_seq, batch_dec_length,
                  batch_greedy_act_seq, batch_greedy_dec_seq, batch_rew_seq, batch_target_next_q
                  ):
        for ob_seq, act_seq, dec_seq, dec_length, greedy_act_seq, greedy_dec_seq, rew_seq, target_next_q in zip(batch_ob_seq, batch_act_seq, batch_dec_seq, batch_dec_length,
                  batch_greedy_act_seq, batch_greedy_dec_seq, batch_rew_seq, batch_target_next_q):
            self.add(ob_seq, act_seq, dec_seq, dec_length,  greedy_act_seq, greedy_dec_seq, rew_seq, target_next_q)

    def add(self, ob_seq, act_seq, dec_seq, dec_length, greedy_act_seq, greedy_dec_seq, rew_seq, target_next_q):
        if self._next_idx >= len(self._observation_sequence):
            self._observation_sequence.append(ob_seq)
            self._action_sequence.append(act_seq)
            self._decoder_input_sequence.append(dec_seq)
            self._decoder_full_length.append(dec_length)
            self._greedy_action_sequence.append(greedy_act_seq)
            self._greedy_decoder_input_sequence.append(greedy_dec_seq)
            self._reward_sequence.append(rew_seq)
            self._target_next_q.append(target_next_q)
        else:
            self._observation_sequence[self._next_idx] = ob_seq
            self._action_sequence[self._next_idx] = act_seq
            self._decoder_input_sequence[self._next_idx] = dec_seq
            self._decoder_full_length[self._next_idx] = dec_length
            self._greedy_action_sequence[self._next_idx] = greedy_act_seq
            self._greedy_decoder_input_sequence[self._next_idx] = greedy_dec_seq
            self._reward_sequence[self._next_idx] = rew_seq
            self._target_next_q[self._next_idx] = target_next_q

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def size(self):
        return len(self._observation_sequence)

    def random_sample(self, batch_size):
        idxes = [random.randint(0, len(self._observation_sequence) - 1) for _ in range(batch_size)]

        return np.array(self._observation_sequence)[idxes], \
               np.array(self._action_sequence)[idxes], \
               np.array(self._decoder_input_sequence)[idxes], \
               np.array(self._decoder_full_length)[idxes], \
               np.array(self._greedy_action_sequence)[idxes], \
               np.array(self._greedy_decoder_input_sequence)[idxes], \
               np.array(self._reward_sequence)[idxes], \
               np.array(self._target_next_q)[idxes]


if __name__ == "__main__":
    ob_seq = np.random.random(size=(10,20,25))
    ac_seq = np.random.random(size=(10,20))
    dec_input_seq = np.random.random(size=(10,20))
    dec_full_length = np.array([20]*10)
    greedy_ac_seq = np.random.random(size=(10,20))
    greedy_dec_input_seq = np.random.random(size=(10,20))
    reward_seq = np.random.random(size=(10,20))
    target_q_seq = np.random.random(size=(10, 20))

    replay_buffer = SeqReplayBuffer(size=10)

    for _ in range(100):
        replay_buffer.add_batch(ob_seq, ac_seq, dec_input_seq,dec_full_length, greedy_ac_seq, greedy_dec_input_seq, reward_seq, target_q_seq)


    print(replay_buffer.size())
