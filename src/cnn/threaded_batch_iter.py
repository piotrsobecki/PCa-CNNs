import multiprocessing as mp
import numpy as np
import random
from threading import Thread
from queue import Queue
from constants import Variables
from prostatex.batch import get_batch_idx
from multiprocessing.managers import BaseManager

from cnn.network_input import NetworkInput
from scipy.ndimage.interpolation import rotate

from prostatex.data_augmentation import crop_data


def adj_contrast(x, contrast_factor=1.0):
    mean = np.mean(x)
    x_adj =  (x - mean) * contrast_factor + mean
    return x_adj

def adj_brightness(x, factor=1.0):
    x_adj = x + factor
    return x_adj

def randbool(ratio=0.5):
    return random.uniform(0.0, 1.0) < ratio

def apply_all_funcs( name, xs, fs_to_apply):
    out = xs
    for func in fs_to_apply:
        out = func(name, out)
    return out


def augment_batch(batch):
    for i in range(len(batch)):
        funcs = []
        #Rotate
        '''
        rot = random.randint(-45, 45)
        order = random.randint(0, 5)
        if rot != 0 and randbool(0.95):
            funcs.append(lambda name, xs: rotate(xs, rot, axes=(0, 1), reshape=False, output=None, order=order))
        '''

        #Shift contrast
        contrast_factor = max(np.random.normal(1.0, 0.75), 0.01)
        funcs.append(lambda name, xs: adj_contrast(xs, contrast_factor))

        #Shift brightness
        brightness_factor = np.random.normal(0.0, 0.10)
        funcs.append(lambda name, xs: adj_brightness(xs, brightness_factor))

        #Move
        mv_1 = random.randint(-4, 4)
        mv_2 = random.randint(-4, 4)
        mv_3 = random.randint(-2, 2)
        if mv_1 != 0:
            funcs.append(lambda name, xs: np.roll(xs, mv_1 if not name.startswith("T2") else 3 * mv_1, axis=0))
        if mv_2 != 0:
            funcs.append(lambda name, xs: np.roll(xs, mv_2 if not name.startswith("T2") else 3 * mv_2, axis=1))
        if mv_3 != 0:
            funcs.append(lambda name, xs: np.roll(xs, mv_3, axis=2))

        '''
        #Apply noise
        elif randbool(ratio=0.10):
            funcs.append(lambda name, xs: np.add(xs, np.random.normal(loc=0.0, scale=0.01, size=xs.shape)))
        elif randbool(ratio=0.25):
            funcs.append(lambda name, xs: np.add(xs, np.random.normal(loc=0.0, scale=0.001, size=xs.shape)))
        elif randbool(ratio=0.50):
            funcs.append(lambda name, xs: np.add(xs, np.random.normal(loc=0.0, scale=0.0001, size=xs.shape)))
        '''

        #Flip
        if randbool(ratio=0.50):
            funcs.append(lambda name, xs: np.flip(xs, axis=0))
        if randbool(ratio=0.50):
            funcs.append(lambda name, xs: np.flip(xs, axis=1))
        if randbool(ratio=0.50):
            funcs.append(lambda name, xs: np.flip(xs, axis=2))

        for name, val in batch.xs.items():
            if name not in ['location']:
                val3d = val[i, ..., 0]
                if randbool(0.125) and name.startswith("T2"):
                    val3d = np.zeros(val3d.shape)
                else:
                    val3d = apply_all_funcs(name, val3d, funcs)
                '''
                elif randbool(0.15):
                    if randbool(ratio=0.05):
                        funcs.append(lambda name, xs: np.add(xs, np.random.normal(loc=0.0, scale=0.1, size=xs.shape)))
                    if randbool(ratio=0.10):
                        funcs.append(lambda name, xs: np.add(xs, np.random.normal(loc=0.0, scale=0.01, size=xs.shape)))
                    if randbool(ratio=0.25):
                        funcs.append(lambda name, xs: np.add(xs, np.random.normal(loc=0.0, scale=0.001, size=xs.shape)))
                    funcs.append(lambda name, xs: np.add(xs, np.random.normal(loc=0.0, scale=0.0001, size=xs.shape)))
                '''
                batch.xs[name][i, ..., 0] = val3d
    return batch


def _gen_batches(dataset: NetworkInput, batch_size,  augment:bool = False):
    num_samples = len(dataset)
    batch_count = range(int(np.ceil(num_samples / batch_size)))
    for i in batch_count:
        batch = dataset.from_ids(get_batch_idx(num_samples, batch_size, i))
        if augment:
            batch = augment_batch(batch)
        batch = crop_data(batch, Variables.width_crop, Variables.depth_crop)
        yield batch


def _produce_samples(dataset: NetworkInput, batch_size,  augment:bool = False):
    num_samples = len(dataset)
    batch_count = range(int(np.ceil(num_samples / batch_size)))
    for i in batch_count:
        #batch_idx = random.sample(range(num_samples), batch_size)
        #batch = dataset.from_ids(batch_idx)

        #batch = dataset.from_ids(get_batch_idx(num_samples, batch_size, i))
        #batch_idx = get_batch_idx(num_samples, batch_size, i)
        batch = dataset.balanced_sample(batch_size)
        if augment:
            batch = augment_batch(batch)
        batch = crop_data(batch, Variables.width_crop, Variables.depth_crop)
        yield batch

def _producer(q: Queue, dataset: NetworkInput, batch_size:int, augment:bool = False):
    # load the batch generator as a python generator
    batch_gen = _gen_batches(dataset, batch_size, augment)
    # loop over generator and put each batch into the queue
    for data in batch_gen:
        q.put(data, block=True)
    # once the generator gets through all data issue the terminating command and close it
    q.put(None)

def _producer_whole(q: Queue, dataset: NetworkInput, iterations:int, batch_size:int, augment:bool = False):
    # load the batch generator as a python generator
    batch_i = 0
    num_samples = len(dataset)
    for iter in range(iterations):
        batch_idx = random.sample(range(num_samples), batch_size)
        batch = dataset.from_ids(batch_idx)
        if augment:
            batch = augment_batch(batch)
        batch = crop_data(batch, Variables.width_crop, Variables.depth_crop)
        q.put(batch, block=True)
        batch_i+=1
    # once the generator gets through all data issue the terminating command and close it
    q.put(None)



def _gen_ts(dataset: NetworkInput, batch_size,  augment:bool = False):
    num_samples = len(dataset)
    batch_count = range(int(np.ceil(num_samples / batch_size)))
    for i in batch_count:
        batch = dataset.from_ids(get_batch_idx(num_samples, batch_size, i))
        if augment:
            batch = augment_batch(batch)
        yield batch


class threaded_batch_loop(object):
    '''
    Batch iterator to make transformations on the data.
    Uses multiprocessing so that batches can be created on CPU while GPU runs previous batch
    '''

    def __init__(self, iterations:int,  batch_size:int):
        self.iterations = iterations
        self.batch_size = batch_size
        self.epoch = 0
        self.batch_num = 0
        self.new_epoch = True

    def __call__(self, dataset:NetworkInput, augment:bool = False):
        self.augment = augment
        self.dataset = dataset
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        runs the _gen_batches function in a seperate process so that it can be run while the GPU is running previous batch
        '''
        #q = mp.Queue(maxsize=20)
        q = Queue(maxsize=25)
        num_threads = 5
        '''
        for i in range(num_threads):
            # start the producer in a seperate process and set the process as a daemon so it can quit easily if you ctrl-c
            thread = mp.Process(target=_producer_whole, args=[q, self.dataset, int(self.iterations/num_threads), self.batch_size, self.augment])
            thread.daemon = True
            thread.start()
        '''
        workers = []
        for i in range(num_threads):
            args = [q, self.dataset, int(self.iterations/num_threads), self.batch_size, self.augment]
            worker = Thread(target=_producer_whole, args=args)
            workers.append(worker)

        # start all workers
        for worker in workers:
            worker.start()

        #self.batch_count = int(np.ceil(len(self.dataset) / self.batch_size))
        self.batch_num = 0
        # grab each successive list containing X_batch and y_batch which were added to the queue by the generator
        for data in iter(q.get, None):
            #self.epoch = int(np.floor(batch_num / self.batch_count))
            self.epoch = self.batch_num
            self.new_epoch = True #batch_num % self.batch_count == 0
            self.batch_num += 1
            yield data

        # wait for all workers to finish
        for worker in workers:
            worker.join()

