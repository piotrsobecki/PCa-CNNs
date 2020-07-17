from math import ceil

from numpy import ndarray, random
import numpy as np
import os
from cnn.network import Network
from cnn.network_input import NetworkInput, NetworkInputSplit
from cnn.threaded_batch_iter import _gen_batches
from constants import Constants
from operator import add
from scipy.stats.stats import pearsonr
import time


class NetworkHistory:
    train_auc = ndarray
    train_auc_all = dict
    train_last_dense = ndarray
    val_cost = ndarray
    train_cost = ndarray
    val_auc = ndarray
    val_auc_all = dict
    running_train_val_auc_cor = ndarray
    running_train_val_acc_cor = ndarray
    test_probabilities = ndarray
    train_probabilities = ndarray
    epochs = int
    best_epoch = int
    predictions = ndarray
    max_epoch = int
    all_epochs = list
    elapsed = ndarray
    time_start = int
    def __init__(self, network: Network, epochs: int):
        self.network = network
        self.epochs = epochs
        self.best_epoch = -1
        self.max_epoch = -1
        self.all_epochs = []
        self.train_auc_all = dict()
        self.train_losses = dict()
        self.elapsed = np.zeros(epochs + 100)
        self.train_auc = np.zeros(epochs + 100)
        self.val_cost = np.zeros(epochs + 100)
        self.train_cost = np.zeros(epochs + 100)
        self.train_cost_all = dict()
        self.train_last_dense = dict()
        self.train_predictions = dict()
        self.test_probabilities = dict()
        self.train_probabilities = dict()
        self.val_probabilities = dict()
        self.val_auc = np.zeros(epochs + 100)
        self.val_auc_all = dict()
        self.val_cost_all = dict()
        self.val_last_dense = dict()
        net_keys = ['DCE', 'DCE_PZ', 'DCE_TZ', 'DWI_ADC', 'DWI_ADC_TZ', 'DWI_ADC_PZ', 'T2', 'T2_PZ', 'T2_TZ', 'PZ',
                    'TZ', 'NET']
        self.auc_key_order = [*net_keys]
        self.cost_key_order = [*net_keys]
        self.auc_key_order.sort()
        self.cost_key_order.sort()
        self.time_start = time.time()

    def header(self):
        sn = self.get_subnets()
        train_prob_keys = list('TR:' + name for name in self.auc_key_order if name in sn)
        val_prob_keys = list('VL:' + name for name in self.auc_key_order if name in sn)
        t_p_k = '\t'.join(train_prob_keys)
        t_p_v = '\t'.join(val_prob_keys)
        # train_cost_keys = list('TR:'+name for name in self.cost_key_order)
        # val_cost_keys = list('VL:'+name for name in self.cost_key_order)
        return "ITER\tBEST_SCORE\tTR\tVL\t" + t_p_k + '\t' + t_p_v + '\t' + "ELAPSED"  # +'\t' + '\t'.join(train_cost_keys)+ '\t' + '\t'.join(val_cost_keys)

    def update(self, input: NetworkInputSplit, epoch, test: NetworkInput):

        self.elapsed[epoch] = time.time() - self.time_start

        self.train_auc_all[epoch] = self.get_aucs(input.train_test)
        self.train_auc[epoch] = self.train_auc_all[epoch]['NET']
        self.all_epochs.append(epoch)
        if epoch > self.max_epoch:
            self.max_epoch = epoch
        if not input.no_val:
            val_aucs = self.get_aucs(input.val)
            self.val_auc_all[epoch] = val_aucs
        else:
            self.val_auc_all[epoch] = self.train_auc_all[epoch]
        self.val_auc[epoch] = self.val_auc_all[epoch]['NET']


        if self.best_epoch < 0 or (input.no_val and epoch == self.epochs - 1) or (
                not input.no_val and self.val_auc[self.best_epoch] <= self.val_auc[epoch]):
            self.best_epoch = epoch
            self.test_probabilities[epoch] = self._feed_probabilities(test)['NET']
            self.train_probabilities[epoch] = self._feed_probabilities(input.train_test)['NET']
            self.val_probabilities[epoch] = self._feed_probabilities(input.val)['NET']
        else:
            for checkpoint_epoch in Constants.checkpoint_epochs:
                if checkpoint_epoch == epoch:
                    self.test_probabilities[epoch] = self._feed_probabilities(test)['NET']
                    self.train_probabilities[epoch] = self._feed_probabilities(input.train_test)['NET']
                    self.val_probabilities[epoch] = self._feed_probabilities(input.val)['NET']


    def get_subnets(self):
        fprobs = self.network.probabilities
        if 'DWI_ADC' in fprobs and 'TZ' in fprobs:
            fprobs['DWI_ADC_PZ'] = fprobs['DWI_ADC']
            fprobs['DWI_ADC_TZ'] = fprobs['DWI_ADC']
        if 'T2' in fprobs and 'TZ' in fprobs:
            fprobs['T2_PZ'] = fprobs['T2']
            fprobs['T2_TZ'] = fprobs['T2']
        if 'DCE' in fprobs and 'TZ' in fprobs:
            fprobs['DCE_PZ'] = fprobs['DCE']
            fprobs['DCE_TZ'] = fprobs['DCE']
        return [
            *fprobs.keys()
        ]

    def get_aucs(self, ni: NetworkInput):
        fprobs = self._feed_probabilities(ni)
        if 'DWI_ADC' in fprobs and 'TZ' in fprobs:
            fprobs['DWI_ADC_PZ'] = fprobs['DWI_ADC']
            fprobs['DWI_ADC_TZ'] = fprobs['DWI_ADC']
        if 'T2' in fprobs and 'TZ' in fprobs:
            fprobs['T2_PZ'] = fprobs['T2']
            fprobs['T2_TZ'] = fprobs['T2']
        if 'DCE' in fprobs and 'TZ' in fprobs:
            fprobs['DCE_PZ'] = fprobs['DCE']
            fprobs['DCE_TZ'] = fprobs['DCE']
        aucs = {name: ni.auc(probs, name=name) for name, probs in fprobs.items()}
        return aucs

    def _pearsonr(self, list_a, list_b):
        small_noise_a = [random.uniform(0, 0.00001) for i in range(0, len(list_a))]
        small_noise_b = [random.uniform(0, 0.00001) for i in range(0, len(list_b))]
        list_a_n = [*map(add, list_a, small_noise_a)]
        list_b_n = [*map(add, list_b, small_noise_b)]
        return pearsonr(list_a_n, list_b_n)

    def _get_loss(self, loss_tnsr, ni: NetworkInput):
        batch_size = Constants.batch_size
        batch_count = int(ceil(ni.length / batch_size))
        loss = 0.0
        for batch in _gen_batches(ni, Constants.batch_size, augment=False):
            loss += self.network.session.run(loss_tnsr, feed_dict=batch.prepare_feed_dict(self.network)) / batch_count
        return loss

    def _get_losses(self, losses, ni: NetworkInput):
        batch_size = Constants.batch_size
        batch_count = int(ceil(ni.length / batch_size))
        losses_out = {name: 0.0 for name, tnsr in losses.items()}
        loss = 0.0
        for batch in _gen_batches(ni, Constants.batch_size, augment=False):
            losses_batch = self.network.session.run(losses, feed_dict=batch.prepare_feed_dict(self.network))
            for name, loss in losses_batch.items():
                losses_out[name] += loss / batch_count
        return losses_out

    def dump(self, fname):
        fname = os.path.join(Constants.log_dir, Constants.test_name, fname)
        fname = os.path.abspath(fname)
        dname = os.path.dirname(fname)
        if not os.path.isdir(dname) and not os.path.isfile(dname):
            os.makedirs(dname)
        f = open(fname, "a+")
        f.write(self.header() + '\n')
        for epoch in self.all_epochs:
            f.write(self.row(epoch) + '\n')
        f.close()

    def _feed_probabilities(self, ni: NetworkInput):
        probs_out = None
        for batch in _gen_batches(ni, Constants.batch_size, augment=False):
            probs_batch = self.network.session.run(self.network.probabilities,
                                                   feed_dict=batch.prepare_feed_dict(self.network))
            if probs_out is None:
                probs_out = probs_batch
            else:
                probs_out = {name: np.append(val, probs_batch[name], axis=0) for name, val in probs_out.items()}
        return probs_out

    def row(self, epoch):
        train_aucs = '\t'.join(["%.3f" % self.train_auc_all[epoch][name] for name in self.auc_key_order if
                                name in self.train_auc_all[epoch]])
        val_aucs = '\t'.join(
            ["%.3f" % self.val_auc_all[epoch][name] for name in self.auc_key_order if name in self.val_auc_all[epoch]])
        bs = self.val_auc[self.best_epoch]
        elapsed = self.elapsed[epoch]
        # train_costs = '\t'.join(["%.3f" % self.train_cost_all[epoch][name] for name in self.cost_key_order])
        # val_costs = '\t'.join(["%.3f" % self.val_cost_all[epoch][name] for name in self.cost_key_order])
        return "%06d\t%.3f\t%.3f\t%.3f\t%s\t%s\t%d" % (
        epoch, bs, self.train_auc[epoch], self.val_auc[epoch], train_aucs, val_aucs,
        elapsed)  # , train_costs, val_costs)
