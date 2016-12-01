import random
import itertools
from . import loader
from . import generator
import os
import numpy as np
import pdb


class SegMNIST(object):
    def __init__(self, mnist, gridH=2, gridW=2,
                 prob_mask_bg=0,
                 min_cells_with_digits=1,
                 max_cells_with_digits=None):
        assert mnist is not None
        self._mnist_iter = mnist.iter()

        self._prob_mask_bg = prob_mask_bg

        self._min_cells_with_digits = min_cells_with_digits
        self._max_cells_with_digits = max_cells_with_digits
        if self._max_cells_with_digits is None:
            self._max_cells_with_digits = gridH * gridW

        assert self._max_cells_with_digits <= gridH * gridW
        assert self._max_cells_with_digits >= self._min_cells_with_digits
        assert self._min_cells_with_digits >= 1

        self._gridH = gridH
        self._gridW = gridW

    @staticmethod
    def load_standard_MNIST(name, shuffle, path=None):
        if path is None:
            if "MNIST_PATH" in os.environ:
                path = os.environ["MNIST_PATH"]
            else:
                raise RuntimeError('Environment variable MNIST_PATH or ' +
                                   'function parameter path must be defined')

        if shuffle:
            D = loader.ShuffledMNIST
        else:
            D = loader.MNIST

        if name == 'training' or name == 'mnist-training':
            mnist = D(path, dataset_slice=(0, 5000))
            mnist.load_standard('training')
            return mnist
        elif name == 'validation' or name == 'mnist-validation':
            mnist = D(path, dataset_slice=(5000, 6000))
            mnist.load_standard('training')
            return mnist
        else:
            raise RuntimeError('Unknown standard MNIST-type dataset: %s' %
                               name)

    def create_batch(self, batch_size):
        img_data = np.zeros((batch_size, 1,
                             28 * self._gridH,
                             28 * self._gridW), dtype=np.uint8)
        seg_label = np.zeros((batch_size, 1,
                              28 * self._gridH,
                              28 * self._gridW), dtype=np.uint8)
        cls_label = np.zeros((batch_size, 10, 1, 1), dtype=np.uint8)

        for n in range(batch_size):
            ndigits = random.randint(self._min_cells_with_digits,
                                     self._max_cells_with_digits)
            grid = np.random.permutation(
                [True] * ndigits +
                [False] * (self._gridH * self._gridW - ndigits))
            grid = grid.reshape(self._gridH, self._gridW)

            (new_data, new_segm, labels) = generator.generate_textured_grid(
                self._mnist_iter,
                grid,
                bgmask=random.random() < self._prob_mask_bg)

            img_data[n, 0] = new_data
            seg_label[n] = new_segm

            # Randomly pick a label
            lbl = random.sample(labels, 1)[0]
            cls_label[n, lbl, 0, 0] = 1

            # from PIL import Image
            # img = Image.fromarray(new_data, 'L')
            # img.show()
            # pdb.set_trace()
        return (img_data, cls_label, seg_label)
