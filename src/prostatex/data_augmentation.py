from scipy.ndimage.interpolation import rotate
import random
from constants import Constants
import numpy as np
import pickle
from cnn.network_input import NetworkInput

'''
Augment data by flipping, translating and rotating arrays
'''

sigma = 0.1
funcs_len = 1

not_augment_other = [
    'location'
]

# Returns boolean with probability of ratio
def randbool(ratio=0.5):
    return random.uniform(0.0, 1.0) < ratio


def get_funcs():
    '''
    mv_3 = random.randint(-2, 2)
    mv_2 = random.randint(-2, 2)
    mv_1 = random.randint(-2, 2)
    '''
    rot = random.randint(-45, 45)
    order = random.randint(0, 5)
    funcs = []

    if rot != 0:
        funcs.append(lambda name, xs: rotate(xs, rot, axes=(0, 1), reshape=False, output=None, order=order))
    '''
    if mv_1 != 0:
        funcs.append(lambda name, xs: np.roll(xs, mv_1 if not name.startswith("T2") else 4*mv_1, axis=0))
    if mv_2 != 0:
        funcs.append(lambda name, xs: np.roll(xs, mv_2 if not name.startswith("T2") else 4*mv_2, axis=1))
    if mv_3 != 0:
        funcs.append(lambda name, xs: np.roll(xs, mv_3 if not name.startswith("T2") else 4*mv_3, axis=2))
    if randbool(ratio=0.25):
        funcs.append(lambda name, xs: np.flip(xs, axis=0))
    if randbool(ratio=0.25):
        funcs.append(lambda name, xs: np.flip(xs, axis=1))
    if randbool(ratio=0.25):
        funcs.append(lambda name, xs: np.flip(xs, axis=2))
    '''
    return funcs  # do_shuffle(funcs)


def get_next_seed():
    return random.randint(0, 2 ** funcs_len)


def get_config(seed: int):
    return [int(x) for x in list(('{0:0' + str(funcs_len) + 'b}').format(seed))]


def get_funcs_config(config):
    return get_funcs()  # [t[0] for t in zip(get_funcs(), config) if bool(t[1])]


def apply_all_funcs(name, xs, fs_to_apply):
    out = xs
    for func in fs_to_apply:
        out = func(name, out)
    return out


def array_map(x, func):
    return np.array(list(map(func, x)))


def augment(_input: NetworkInput):
    def apply_augment_each_row(xs: NetworkInput):
        out = xs.copy()
        for i in range(out.length):
            config = get_config(get_next_seed())
            funcs = get_funcs_config(config)

            all_modalities = ["DWI-ADC", "T2", "DCE"]
            not_augment = []
            not_augment.extend(not_augment_other)
            '''
            DROPPING MODALITIES
            
            
            connected_modalities = {
                "DWI-ADC": ["DWI", "ADC"],
                "T2": ["T2-TRA", "T2-COR", "T2-SAG"]
            }
            modalities_count = len(all_modalities)
            dropped_modalities = 0  # To assure that not all modalities are dropped
            for m in all_modalities:
                if dropped_modalities < modalities_count and randbool(0.01):
                    print("%d - Dropped %s" % (i, m))
                    if m in connected_modalities:
                        modalities_to_drop = connected_modalities[m]
                    else:
                        modalities_to_drop = [m]
                    not_augment.extend(modalities_to_drop)
                    for mod_dropped in modalities_to_drop:
                        if mod_dropped not in _input.dropped_modalities_idx:
                            _input.dropped_modalities_idx[mod_dropped] = []
                        _input.dropped_modalities_idx[mod_dropped].append(i)
            '''

            for name, val in out.xs.items():
                if name not in not_augment:
                    out.xs[name][i, ..., 0] = apply_all_funcs(name, val[i, ..., 0], funcs)
        return out

    return apply_augment_each_row(_input)


def file_name( ni_name: str, batch_num:int):
    return Constants.net_inputs_dir + "aug_tmp_" + ni_name + "_" + str(batch_num) + ".bin"

def __deserialize_data(fname) -> NetworkInput:
    with open(fname, mode='rb') as binary_file:
        ni = pickle.load(binary_file)
    return ni

def __serialize_aug_data(ni: NetworkInput, batch_num:int):
    fname = file_name(ni.name,batch_num=batch_num)
    with open(fname, mode='wb') as binary_file:
        pickle.dump(ni, binary_file, protocol=4)
    return fname

def combine_fnames(input:NetworkInput, name,  fnames):
    out = input.copy()
    out.name = name
    for fname in fnames:
        ai = __deserialize_data(fname)
        out.append(ai)
    return out

def augment_data(name, input: NetworkInput, width_crop, depth_crop,
                 how_many=Constants.data_augment_approx) -> NetworkInput:
    fnames = []
    size = input.length
    how_many_repeats = int(how_many / size)
    random.seed(1)
    for repeat in range(0, how_many_repeats):
        print("Augmenting: (%d/%d)" % (repeat + 1, how_many_repeats))
        aug_i = augment(input)
        fname = __serialize_aug_data(aug_i, repeat)
        fnames.append(fname)
    out = combine_fnames(input, name, fnames)
    return out


def crop_data(ni: NetworkInput, width_crop, depth_crop):
    for name, arr in ni.xs.items():
        if name not in not_augment_other:
            ni.xs[name] = arr[:, width_crop:-width_crop, width_crop:-width_crop, depth_crop:-depth_crop]
    ni.setup_shapes()
    return ni
