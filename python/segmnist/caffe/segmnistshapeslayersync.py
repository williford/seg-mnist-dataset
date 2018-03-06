# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe
import ast

import numpy as np
import os.path as osp

from xml.dom import minidom
import random
from random import shuffle
from threading import Thread
from PIL import Image

from segmnist import SegMNISTShapes
from segmnist.loader.mnist import load_standard_MNIST
from segmnist.segmnistshapes import SquareGenerator
from segmnist.segmnistshapes import RectangleGenerator

from segmnist.textures import TextureDispatcher
from segmnist.textures import WhiteNoiseTexture
from segmnist.textures import SinusoidalGratings
from segmnist.textures import IntermixTexture


class SegMNISTShapesLayerSync(caffe.Layer):
    """
    This is a simple synchronous datalayer for training a network on the
    SegMNISTShapes dataset.
    """

    def setup(self, bottom, top):

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = ast.literal_eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        self.batch_size = params['batch_size']

        # ex. mnist-training or mnist-validation
        self.mnist_dataset_name = params['mnist_dataset']

        # multiplier for number of background pixels that are not masked
        # replaces prob_mask_bg
        if 'bg_pix_mul' in params.keys():
            self.bg_pix_mul = params['bg_pix_mul']
        else:
            self.bg_pix_mul = 1.0

        self.nclasses = params['nclasses']

        self.imshape = params['im_shape']

        if 'digit_positioning' in params.keys():
            self.positioning = params['digit_positioning']
        else:
            self.positioning = 'random'

        if 'nchannels' in params.keys():
            raise NotImplementedError(
                "Using nchannels is currently not "
                "implemented. Use im_shape instead.")

            # assumes imshape is a tuple
            self.imshape = (params['nchannels'],) + self.imshape

        # Create a batch loader to load the images.
        self.mnist = load_standard_MNIST(
            self.mnist_dataset_name, shuffle=True)  # BatchLoader(params, None)

        shapes = []
        if self.nclasses >= 11:
            shapes.append(SquareGenerator())
        if self.nclasses >= 12:
            shapes.append(RectangleGenerator())

        texturegen = TextureDispatcher()
        gratings = None
        if 'pgratings' in params.keys() and params['pgratings'] > 0:
            gratings = SinusoidalGratings(shape=self.imshape))
            texturegen.add_texturegen(
                params['pgratings'],
                gratings)

        defaultTexture = WhiteNoiseTexture(
            mean_dist=lambda: np.random.randint(256),
            var_dist=lambda: np.random.gamma(1, 25),
            shape=self.imshape,
        )
        if 'pwhitenoise' in params.keys() and params['pwhitenoise'] > 0:
            texturegen.add_texturegen(
                params['pwhitenoise'],
                defaultTexture)

        if 'pintermixed' in params.keys() and params['pintermixed'] > 0:
            randomtex = IntermixTexture()
            randomtex.add_texturegen(
                params['pgratings'] if 'pgratings' in params.keys() else 0,
                gratings)
            randomtex.add_texturegen(
                params['pwhitenoise'] if 'pwhitenoise' in params.keys() else 0,
                defaultTexture)

            texturegen.add_texturegen(
                params['pintermixed'],
                randomtex)

        self.batch_loader = SegMNISTShapes(
            self.mnist,
            imshape=self.imshape,
            bg_pix_mul=self.bg_pix_mul,
            positioning=self.positioning,
            shapes=shapes,
            texturegen=texturegen,
        )

        # if no texture generated added via parameters, add white noise
        if len(texturegen.generators()) == 0:
            texturegen.add_texturegen(1, defaultTexture)

        if 'max_digits' in params.keys():
            self.batch_loader.set_max_digits(params['max_digits'])

        if 'min_digits' in params.keys():
            self.batch_loader.set_min_digits(params['min_digits'])

        if 'scale_range' in params.keys():
            self.batch_loader.set_scale_range(params['scale_range'])
        else:
            self.batch_loader.set_scale_range((0.5, 1.5))

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        assert len(self.imshape) > 2
        top[0].reshape(self.batch_size, *params['im_shape'])

        # Note the N channels (for the 10 digits + n shapes).
        top[1].reshape(self.batch_size, self.nclasses, 1, 1)
        if len(top) == 3:  # to-do: deprecate this case!
            if len(params['im_shape']) == 2:
                top[2].reshape(
                    self.batch_size, 1,
                    params['im_shape'][0], params['im_shape'][1])
            else:
                top[2].reshape(
                    self.batch_size, 1, *params['im_shape'][1:])
        elif len(top) == 4:
            top[2].reshape(self.batch_size, self.nclasses, 1, 1)
            if len(params['im_shape']) == 2:
                top[3].reshape(
                    self.batch_size, 1,
                    params['im_shape'][0], params['im_shape'][1])
            else:
                top[3].reshape(
                    self.batch_size, 1, *params['im_shape'][1:])

        print_info("SegMNISTShapesLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        (img_data, cls_label, seg_label) = (
            self.batch_loader.create_batch(self.batch_size))
        top[0].data[...] = img_data
        if len(top) == 3:  # tops: (data, cls-label, seg-label), to-do: deprecate
            print('Using 3 tops for SegMNISTShapesLayerSync python layer '
                  'is deprecated!\n\n')

            top[1].data.fill(0)

            # set default values to be background
            # (used for the digits with other labels)
            top[2].data.fill(0)

            # for each example in batch
            for n in range(cls_label.shape[0]):
                # get indices (==labels) of classes that are in image
                labels = np.flatnonzero(cls_label[n])

                # randomly pick one of the labels
                lbl = random.sample(labels, 1)[0]
                top[1].data[n, lbl, 0, 0] = 1

                # retain masked out regions
                # (if mask_bg, this includes the original background)
                top[2].data[n, 0][seg_label[n, 0] == 255] = 255

                # set current label to foreground
                ind0 = seg_label[n, 0] == 255
                ind = seg_label[n, 0] == (lbl + 1)
                top[2].data[n, 0][seg_label[n, 0] == (lbl + 1)] = 1
        else:  # tops: (data, cls-label, [attend-label, seg-label])
            assert len(top) == 2 or len(top) == 4
            # cls_label should encode all classes
            top[1].data[...] = cls_label

            if len(top) > 2:
                top[2].data.fill(0)  # top: attend-label

                # set default values to be background
                # (used for the digits with other labels)
                top[3].data.fill(0)

                # for each example in batch
                for n in range(cls_label.shape[0]):
                    # get indices (==labels) of classes that are in image
                    labels = np.flatnonzero(cls_label[n])

                    # randomly pick one of the labels
                    lbl = random.sample(labels, 1)[0]
                    top[2].data[n, lbl, 0, 0] = 1

                    # retain masked out regions
                    # (if mask_bg, this includes the original background)
                    top[3].data[n, 0][seg_label[n, 0] == 255] = 255

                    # set current label to foreground
                    ind0 = seg_label[n, 0] == 255
                    ind = seg_label[n, 0] == (lbl + 1)
                    top[3].data[n, 0][seg_label[n, 0] == (lbl + 1)] = 1

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'mnist_dataset' in params.keys(
    ), 'Params must include mnist_dataset (mnist-training, mnist-validation).'

    required = ['batch_size', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for dataset: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['mnist_dataset'],
        params['batch_size'],
        params['im_shape'])
