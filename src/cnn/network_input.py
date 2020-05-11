from numpy import ndarray, max, concatenate, append, array, argwhere, floor, argmax
from random import shuffle, sample
from sklearn.metrics import roc_auc_score
from cnn.network import Network

class NetworkInput:

    x_additional_decoder = ['TZ', 'PZ', 'AS', 'SV', 'CG']

    name = str
    ids = ndarray
    xs = dict
    ys = ndarray
    input_shape = ndarray
    dropped_modalities_idx = dict

    def __init__(self, name, ids, xs, ys:None):
        self.ids = ids
        self.name = name
        self.xs = xs
        self.ys = ys
        self.dropped_modalities_idx = {}
        if xs is not None:
            self.setup_shapes()

    def setup_shapes(self):
        self.input_shape = {name:val[0].shape for name, val in self.xs.items()}
        self.length = max([ len(val) for name, val in self.xs.items()])

    def copy(self):
        return NetworkInput( self.name, self.ids.copy(), self.copy_dict(self.xs), self.ys.copy())

    def zones(self):
        decoder = self.x_additional_decoder
        location_args = argmax(self.xs['location'], axis=1)
        return array([*map(lambda i: decoder[i], location_args)]).reshape([len(self), 1])

    def from_ids(self, idx):
        xs = {name: val[idx] for name, val in self.xs.items()}
        return NetworkInput(self.name, self.ids[idx], xs, self.ys[idx])


    def copy_dict(self, dict):
        return {name: val.copy() for name, val in dict.items()}

    @classmethod
    def new_ni(cls):
        return cls(None,None,None,None)

    def append(self, ni):
        self.ids = concatenate([self.ids, ni.ids], axis=0)
        self.ys = concatenate([self.ys, ni.ys], axis=0)
        current_size = len(self)
        for mod, idxs in ni.dropped_modalities_idx.items():
            if mod not in self.dropped_modalities_idx:
                self.dropped_modalities_idx[mod] = []
            self.dropped_modalities_idx[mod].extend([current_size + idx for idx in idxs])

        for name, val in ni.xs.items():
            self.xs[name] = concatenate([self.xs[name], val], axis=0)

    def shuffle(self):
        idx = [*range(len(self.ids))]
        shuffle(idx)
        self.ids = self.ids[idx]
        self.ys = self.ys[idx]
        for name,val in self.xs.items():
            self.xs[name] = val[idx]
        return self

    def balanced_sample_ratio(self, ratio=1.0):
        ys= self.ys[:, 1]
        idx_0 = [i for i, y in enumerate(ys) if not bool(int(y))]
        idx_1 = [i for i, y in enumerate(ys) if bool(int(y))]
        how_many = int( (min(len(idx_1), len(idx_0)) * ratio)/2)
        idx_1_s = sample(idx_1, how_many)
        idx_0_s = sample(idx_0, how_many)
        indexes = append(idx_1_s, idx_0_s)
        shuffle(indexes)
        return self.from_ids(indexes)

    def balanced_sample(self, how_many):
        ys= self.ys[:, 1]
        how_many = int(how_many/2)
        idx_0 = [i for i, y in enumerate(ys) if not bool(int(y))]
        idx_1 = [i for i, y in enumerate(ys) if bool(int(y))]
        idx_1_s = sample(idx_1, how_many)
        idx_0_s = sample(idx_0, how_many)
        indexes = append(idx_1_s, idx_0_s)
        shuffle(indexes)
        return self.from_ids(indexes)

    def setup_auc(self):
        def subindx(base, location):
            return  [id for id in base if id in location]
        locations = self.xs['location']
        locations_pz = locations[:, NetworkInput.x_additional_decoder.index('PZ')]
        locations_tz = 1 - locations_pz
        pz_idx = argwhere(locations_pz >= 1)
        tz_idx = argwhere(locations_tz >= 1)
        dwi_adc_idx = [id for id in range(0, len(self)) if 'DWI' not in self.dropped_modalities_idx or id not in self.dropped_modalities_idx['DWI']]
        t2_idx =  [id for id in range(0, len(self)) if 'T2' not in self.dropped_modalities_idx or id not in self.dropped_modalities_idx['T2']]
        dce_idx = [id for id in range(0, len(self)) if 'DCE' not in self.dropped_modalities_idx or id not in self.dropped_modalities_idx['DCE']]
        self.name_idx = {
            'T2': t2_idx,
            'T2_TZ': subindx(t2_idx, tz_idx),
            'T2_PZ': subindx(t2_idx, pz_idx),
            'DWI_ADC': dwi_adc_idx,
            'DWI_ADC_PZ': subindx(dwi_adc_idx, pz_idx),
            'DWI_ADC_TZ': subindx(dwi_adc_idx, tz_idx),
            'DCE': dce_idx,
            'DCE_PZ': subindx(dce_idx, pz_idx),
            'DCE_TZ': subindx(dce_idx, tz_idx),
            'PZ': pz_idx,
            'TZ': tz_idx,
            'NET': [*range(0, len(self))]
        }


    def _auc(self, preds, y):
        try:
            return roc_auc_score(y_true=y, y_score=preds)
        except ValueError as e:
            print(e)
            return 0.5

    def auc(self, probs, name="NET",  all=False):
        if not hasattr(self,'name_idx'):
            self.setup_auc()
        if all:
            idx = self.name_idx['NET']
        else:
            idx = self.name_idx[name]
        return self.auc_idx(probs, idx)

    def auc_idx(self, preds, idx):
        ys = self.ys[idx, 1]
        preds = preds[idx, 1]
        return self._auc(preds, ys)

    def prepare_feed_dict(self, network:Network, do_train=False, learning_rate=None):
        feed_dict = {tensor: self.xs[name] for name, tensor in network.x.items()}
        feed_dict[network.y] = self.ys
        feed_dict[network.training] = do_train
        feed_dict[network.learning_rate] = learning_rate
        return feed_dict

    def __len__(self):
        return len(self.ids)

class NetworkInputSplit:
    no_val = False
    train = NetworkInput
    train_test = NetworkInput
    val = NetworkInput

    input_shape = ndarray

    def __init__(self, train: NetworkInput, train_test:NetworkInput = None, val: NetworkInput = None):
        self.train = train
        self.train_test = train_test
        if self.train_test is None:
            self.train_test = train
        if val is None:
            self.no_val = True
            self.val = train
        else:
            self.val = val
        self.input_shape = train.input_shape

