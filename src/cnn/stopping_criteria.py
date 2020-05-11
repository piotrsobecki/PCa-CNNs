from cnn.network_history import NetworkHistory
import numpy as np

from constants import Constants


class StoppingCriteria:

    epsilon  = 0.01

    #GL
    GLa = 35

    #PQ
    Pk = 15

    PQa = 1

    #UP
    UPk = 15

    #Other
    min_epochs = Constants.min_epochs

    max_epochs = Constants.max_epochs

    max_training_err = 0.1

    def __init__(self, network_history:NetworkHistory):
        self.network_history = network_history
        self.train_err = np.zeros(network_history.epochs)
        self.val_err = np.zeros(network_history.epochs)
        self.opt_err = np.zeros(network_history.epochs)

        self.gl = np.zeros(network_history.epochs)
        self.pk = np.zeros(network_history.epochs)

        self.ups = np.zeros(network_history.epochs)
        self.up = np.zeros(network_history.epochs)

        self.epoch_condition_h = np.zeros(network_history.epochs)
        self.min_training_h = np.zeros(network_history.epochs)
        self.gl_condition_h = np.zeros(network_history.epochs)
        self.pq_condition_h = np.zeros(network_history.epochs)
        self.up_condition_h = np.zeros(network_history.epochs)

        self.epoch_condition_s = False
        self.min_training_s = False
        self.gl_condition_s = False
        self.pq_condition_s = False
        self.up_condition_s = False

    def update(self,epoch):
        self.setup_values(epoch)
        self.setup_conditions(epoch)
        self.update_satisfied(epoch)


    def header(self):
        return "EPOCH\tGL\tPQ"

    def row(self, epoch):
        return "%04d\t%.2f\t%.2f" % (epoch, self.gl[epoch], self.pk[epoch])


    def setup_values(self,epoch):
#        self.train_err[epoch] = self.network_history.cost[epoch]

 #       self.val_err[epoch] = self.network_history.val_cost[epoch]

        self.opt_err[epoch] = np.min(self.val_err)

        self.gl[epoch] = 100 * ( self.val_err[epoch] / (self.opt_err[epoch] + self.epsilon) -1 )

        if epoch > self.Pk:
            self.pk[epoch] = 1000 * np.sum(self.train_err[epoch - self.Pk:]) / (self.Pk * np.min(self.train_err[epoch - self.Pk:]) + self.epsilon)

        if epoch > self.UPk:
            self.ups[epoch] = self.val_err[epoch] > self.val_err[epoch-self.UPk]
            self.up[epoch] = self.ups[epoch] and self.ups[epoch-1]


    def setup_conditions(self,epoch):
        self.epoch_condition_h[epoch] = self.epoch_condition(epoch)
        self.min_training_h[epoch] = self.training_err_condition(epoch)
        self.gl_condition_h[epoch] = self.gl_condition(epoch)
        self.pq_condition_h[epoch] = self.pq_condition(epoch)
        self.up_condition_h[epoch] = self.up_condition(epoch)

    def update_satisfied(self, epoch):
        self.epoch_condition_s = self.epoch_condition_s or self.epoch_condition_h[epoch]
        self.min_training_s = self.min_training_s or self.min_training_h[epoch] and self.epoch_condition_s
        self.gl_condition_s = self.gl_condition_s or self.gl_condition_h[epoch] and self.epoch_condition_s and self.min_training_s
        self.pq_condition_s = self.pq_condition_s or self.pq_condition_h[epoch] and self.epoch_condition_s and self.min_training_s
        self.up_condition_s = self.up_condition_s or self.up_condition_h[epoch] and self.epoch_condition_s and self.min_training_s
        self.stopping_condition = epoch >= self.max_epochs or (self.epoch_condition_s  and self.up_condition_s) #(self.epoch_condition_s and self.min_training_s and self.gl_condition_s and self.pq_condition_s and self.up_condition_s)

    def satisfied(self):
        return self.stopping_condition

    def training_err_condition(self, epoch):
        return self.train_err[epoch] < self.max_training_err

    def epoch_condition(self, epoch):
        return epoch>self.min_epochs

    def gl_condition(self,epoch):
        return self.gl[epoch] > self.GLa

    def pq_condition(self,epoch):
        return self.pk[epoch] > self.PQa

    def up_condition(self,epoch):
        return bool(self.up[epoch])