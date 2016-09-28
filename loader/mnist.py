# Based on https://gist.github.com/akesling/5358964

import os
import struct
import numpy as np

import random
import math


class MNIST(object):
    def __init__(self, root_dir="", slice_percent=None, seed=2016):
        self.root_dir = root_dir
        self.lbl = None
        self.img = None
        assert slice_percent==None or len(slice_percent)==2
        self.slice_percent = slice_percent
        self.seed_start = seed
        self.seed_current = seed

    def load_standard(self, name):
        if name == 'training':
            return self.load('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
        elif name == 'testing':
            return self.load('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
        raise RuntimeError('Unknown standard dataset "%s".' % name)

    def load(self, image_file, label_file):
        # set random seed for every new dataset
        self.seed_current = self.seed_start
        with open(os.path.join(self.root_dir, label_file), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        if len(lbl) == 0:
            raise RuntimeError("Dataset is empty.")

        with open(os.path.join(self.root_dir, image_file), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        begin = int(math.floor(self.slice_percent[0] / 100.0 * len(lbl)))
        end = int(math.floor(self.slice_percent[1] / 100.0 * len(lbl)) + 1)

        self.lbl = lbl[begin:end]
        self.img = img[begin:end]

        return self.lbl, self.img

    def random(self, max_iter=float("inf")):
        if self.lbl is None or self.img is None:
            raise RuntimeError("Dataset must be loaded before calling random().")

        while max_iter > 0:
            # set seed before each random range call
            random.seed(self.seed_current)
            self.seed_current += 1

            i = random.randrange(len(self.lbl))
            yield (self.lbl[i], self.img[i])
            max_iter -= 1
