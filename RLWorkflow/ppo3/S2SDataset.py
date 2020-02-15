import tensorflow as tf

class S2SDataSet(object):
    def __init__(self):
        pass

    def set_training_set(self, state, encoder_length, sample_action, decoder_length, reward, adv, oldvpred):
        self.state = state
        self.encoder_length = encoder_length
        self.sample_action = sample_action
        self.decoder_length = decoder_length
        self.reward = reward
        self.adv = adv
        self.oldvpred = oldvpred

    def get_adv_iterator(self):
        adv_dataset = tf.data.Dataset.from_tensor_slices(self.adv)
        it_adv = adv_dataset.make_initializable_iterator()

        return it_adv

    def get_state_iterator(self):
        state_dataset = tf.data.Dataset.from_tensor_slices(self.state)
        it_state = state_dataset.make_initializable_iterator()

        return it_state

    def get_encoder_length_iterator(self):
        encoder_length_dataset = tf.data.Dataset.from_tensor_slices(self.encoder_length)
        it_encoder_length = encoder_length_dataset.make_initializable_iterator()

        return it_encoder_length

    def get_action_iterator(self):
        action_dataset = tf.data.Dataset.from_tensor_slices(self.sample_action)
        it_action = action_dataset.make_initializable_iterator()

        return it_action

    def get_decoder_length_iterator(self):
        decoder_length_dataset = tf.data.Dataset.from_tensor_slices(self.decoder_length)
        it_decoder_length = decoder_length_dataset.make_initializable_iterator()

        return it_decoder_length

    def get_reward_iterator(self):
        reward_dataset = tf.data.Dataset.from_tensor_slices(self.reward)
        it_reward = reward_dataset.make_initializable_iterator()

        return it_reward

    def get_oldvpred_iterator(self):
        oldvpred_dataset = tf.data.Dataset.from_tensor_slices(self.oldvpred)
        it_oldvpred = oldvpred_dataset.make_initializable_iterator()

        return it_oldvpred
