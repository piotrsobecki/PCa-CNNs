from cnn.network_history import NetworkHistory
from cnn.network_input import NetworkInputSplit
from cnn.network import Network
from constants import Variables, Constants
from sklearn import metrics
from prostatex.batch import get_batch_xy
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

class NetworkStatistics:
    network = Network
    input = NetworkInputSplit
    history = NetworkHistory

    def __init__(self, network: Network, input: NetworkInputSplit, history: NetworkHistory):
        self.network = network
        self.input = input
        self.history = history

    def plot_roc(self):
        train_probabilities = self._get_probability_output(self.input.train_x, self.input.train_y)
        train_labels = self._get_labels_output(self.input.train_x, self.input.train_y)
        test_probabilities = self._get_probability_output(self.input.test_x, self.input.test_y)
        test_labels = self._get_labels_output(self.input.test_x, self.input.test_y)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        sets = ['train', 'test']
        fpr[0], tpr[0], _ = metrics.roc_curve(train_labels, train_probabilities[:,1])
        roc_auc[0] = metrics.auc(fpr[0], tpr[0])
        fpr[1], tpr[1], _ = metrics.roc_curve(test_labels, test_probabilities[:,1])
        roc_auc[1] = metrics.auc(fpr[1], tpr[1])
        self._save_roc_to_file(sets, roc_auc, fpr, tpr)

        plt.figure()
        for i in range(2):
            plt.plot(fpr[i], tpr[i], lw=2, label='ROC %s (area = %0.4f)' % (sets[i], roc_auc[i]))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC AUC')
            plt.legend(loc="lower right")
        plt.savefig(Constants.models_dir + 'roc.pdf', bbox_inches='tight')
        plt.show()

    def plot_train_history(self):
        plt.figure()
        plt.xlim([1.0, self.history.epochs])
        plt.ylim([0.6, 1.0])
        epochs_nums = np.arange(1, self.history.epochs + 1)
        plt.plot(epochs_nums, self.history.train_acc, lw=2, label='Training accuracy')
        plt.plot(epochs_nums, self.history.test_acc, lw=2, label='Test accuracy')
        self._save_history_to_file(epochs_nums, self.history.train_acc, self.history.test_acc, self.history.cost)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Network training history')
        plt.legend(loc="lower right")
        plt.savefig(Constants.models_dir + 'train_history.pdf', bbox_inches='tight')
        plt.show()

    def _get_probability_output(self, input_x, input_y):
        batch_size = int(len(input_x) / Variables.train_data_to_batch_size_ratio)
        batch_count = ceil(len(input_x) / batch_size)
        tensor_data = np.zeros((0,2))
        for i in range(batch_count):
            batch_xs, batch_ys = get_batch_xy(input_x, input_y, batch_size, i)
            batch_data = self.network.session.run(self.network.probabilities, feed_dict={self.network.x: batch_xs, self.network.y: batch_ys, self.network.keepratio: 1.})
            tensor_data = np.concatenate((batch_data, tensor_data), axis=0)
        return tensor_data

    def _get_labels_output(self, input_x, input_y):
        batch_size = int(len(input_x) / Variables.train_data_to_batch_size_ratio)
        batch_count = ceil(len(input_x) / batch_size)
        tensor_data = np.zeros(0)
        for i in range(batch_count):
            batch_xs, batch_ys = get_batch_xy(input_x, input_y, batch_size, i)
            batch_data = self.network.session.run(self.network.labels, feed_dict={self.network.x: batch_xs, self.network.y: batch_ys, self.network.keepratio: 1.})
            tensor_data = np.concatenate((batch_data, tensor_data), axis=0)
        return tensor_data

    def _save_roc_to_file(self, sets_names, roc_auc, fpr, tpr):
        with open(Constants.models_dir + 'roc.csv', 'w') as csv:
            # train set
            csv.write(sets_names[0] + ';' + str(roc_auc[0]) + '\n')
            for i in range(len(fpr[0])):
                csv.write(str(fpr[0][i]) + ';' + str(tpr[0][i]) + '\n')
            # test set
            csv.write(str(sets_names[1]) + ';' + str(roc_auc[1]) + '\n')
            for i in range(len(fpr[1])):
                csv.write(str(fpr[1][i]) + ';' + str(tpr[1][i]) + ';' + '\n')

    def _save_history_to_file(self, epochs_nums, train_acc, test_acc, train_cost):
        with open(Constants.models_dir + 'history.csv', 'w') as csv:
            csv.write('epoch;train_acc;test_acc;train_cost\n')
            for i in range(len(epochs_nums)):
                csv.write(str(epochs_nums[i]) + ';' + str(train_acc[i]) + ';' + str(test_acc[i]) + ';' + str(train_cost[i]) + '\n')