from cnn.threaded_batch_iter import threaded_batch_loop
from cnn.network_history import NetworkHistory
from cnn.stopping_criteria import StoppingCriteria
from cnn.network_input import NetworkInputSplit, NetworkInput
from cnn.network import Network
from constants import Constants


class NetworkTrainer:
    network = Network
    input = NetworkInputSplit

    def __init__(self, network: Network, input: NetworkInputSplit):
        self.network = network
        self.input = input

    def train(self, test: NetworkInput):

        train_data_len = self.input.train.length
        val_data_len = self.input.val.length
        batch_size = Constants.batch_size

        train_clinsig = sum(self.input.train.ys[..., 1])
        train_not_clinsig = train_data_len - train_clinsig
        val_clinsig = sum(self.input.val.ys[..., 1])
        val_not_clinsig = val_data_len - val_clinsig

        print('TRAIN SIZE: %d, BATCH SIZE: %d' % (train_data_len, batch_size))
        print('Train ClinSig (%06d) \t 0: %06d \t 1: %06d \t 1/0 : %.2f %% \t ' % (
            train_data_len, train_not_clinsig, train_clinsig, 100 * train_clinsig / train_data_len))
        print('Val ClinSig   (%06d) \t 0: %06d \t 1: %06d \t 1/0 : %.2f %% \t' % (
            val_data_len, val_not_clinsig, val_clinsig, 100 * val_clinsig / val_data_len))

        history = NetworkHistory(self.network, Constants.max_epochs)
        stopping_criteria = StoppingCriteria(history)
        history.epochs = stopping_criteria.max_epochs

        _iter = threaded_batch_loop(iterations=stopping_criteria.max_epochs, batch_size=Constants.batch_size)
        print(history.header())
        batch_num = 0

        # Decrese learning rate using learning rate decay factor every X iterations
        learning_rate = 0.05#0.1
        learning_rate_decay = 1.0#0.99
        decay_step = 100

        if Constants.do_train:
            for batch in _iter(self.input.train, augment=True):
                self.network.session.run(self.network.optimizer,
                                         feed_dict=batch.prepare_feed_dict(self.network, do_train=True, learning_rate=learning_rate))
                if _iter.new_epoch and _iter.epoch % 100 == 0:
                    history.update(self.input, _iter.epoch, test)
                    print(history.row(_iter.epoch))
                if _iter.new_epoch and _iter.epoch % decay_step == 0:
                    learning_rate = learning_rate_decay * learning_rate
                batch_num += 1
            return history
        else:  # Restore model instead of training phase
            print("LOADING MODEL: " + Constants.model_to_load)
            self.network.saver.restore(self.network.session, Constants.models_dir + Constants.model_to_load)
