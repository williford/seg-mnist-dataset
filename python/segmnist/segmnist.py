import random
import itertools
from . import loader
from . import texture_generator
import os
import numpy as np
import pdb


class SegMNIST(object):
    def __init__(self, mnist, gridH=2, gridW=2,
                 prob_mask_bg=None,
                 min_cells_with_digits=1,
                 max_cells_with_digits=None,
                 position='grid',
                 nchannels=1):
        assert mnist is not None
        self._mnist_iter = mnist.iter()

        assert prob_mask_bg <= 1.0, "prob_mask_bg should be set from 0 to 1."
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

        self.set_generate_method(position)
        self._nchannels = nchannels

        self._scale_range = (1.0, 1.0)

    def set_min_digits(self, min_digits):
        self._min_cells_with_digits = min_digits

    def set_max_digits(self, max_digits):
        self._max_cells_with_digits = max_digits

    def set_nchannels(self, nchannels):
        self._nchannels = nchannels

    def set_scale_range(self, scale_range):
        self._scale_range = scale_range

    def set_generate_method(self, position):
        if position == "random":
            self._generate = texture_generator.generate_textured_image
        else:
            assert position == "grid"
            self._generate = texture_generator.generate_textured_grid

    def set_prob_mask_bg(self, prob_mask_bg):
        self._prob_mask_bg = prob_mask_bg

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

    """ Return single example with image, class labels, and segmentation labels.
    """
    def create_example(self):
        (img_data, cls_label, seg_label) = self.create_batch(1)
        img_data = img_data.reshape(img_data.shape[1:])
        seg_label = seg_label.reshape(seg_label.shape[2:])
        cls_label = cls_label.reshape(cls_label.shape[:2])
        return (img_data, cls_label, seg_label)

    """ Return batch with images, class labels, and segmentation labels.
        cls_label is a sparse vector with 1 set for every digit that
        appears in image.
    """
    def create_batch(self, batch_size):
        img_data = np.zeros((batch_size, self._nchannels,
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

            assert self._prob_mask_bg is not None, "Probability mask background (prob_mask_bg) not set!"
            # texture_generator.generate_textured_grid(
            (new_data, new_segm, labels) = self._generate(
                self._mnist_iter,
                grid,
                bgmask=random.random() < self._prob_mask_bg,
                nchannels=self._nchannels,
                scale_range=self._scale_range)

            if new_data.ndim <= 2:
                img_data[n, 0] = new_data
            else:
                img_data[n] = new_data

            seg_label[n] = new_segm

            for lbl in labels:
                cls_label[n, lbl, 0, 0] = 1

            # Randomly pick a label
            # lbl = random.sample(labels, 1)[0]
            # cls_label[n, lbl, 0, 0] = 1

            # from PIL import Image
            # img = Image.fromarray(new_data, 'L')
            # img.show()
            # pdb.set_trace()
        return (img_data, cls_label, seg_label)
