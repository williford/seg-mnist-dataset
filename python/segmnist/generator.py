#!/usr/bin/env python

import os
import loader
import scipy.misc
import numpy as np
import errno
import pandas as pd
import itertools

import h5py
import math


def random_texture(shape,
                   mean=np.random.uniform(0, 255),
                   var=np.random.gamma(2, 4)):
    if var == 0:
        texture = np.ones(shape) * mean
    else:
        texture = np.random.normal(loc=mean, scale=math.sqrt(var), size=shape)

    texture[texture > 255] = 255
    texture[texture < 0] = 0
    return texture


def generate_textured_grid(mnist_iter, grid, mnist_shape=(28, 28),
                           bgmask=False):

    num_means = 10
    potential_means = np.arange(num_means) * (255. / (num_means - 1))
    np.random.shuffle(potential_means)

    h = mnist_shape[0]
    w = mnist_shape[1]
    new_data = random_texture((grid.shape[0] * h, grid.shape[1] * w), mean=potential_means[-1])
    potential_means = potential_means[0:-1]
    new_segm = np.zeros((grid.shape[0] * h, grid.shape[1] * w), dtype=np.uint8)
    labels = set()

    for j in range(grid.shape[1]): # column
        for i in range(grid.shape[0]):  # row
            # Position of "first" number
            offset_i = i * h
            offset_j = j * w

            if not grid[i, j]:
                if bgmask:
                    new_segm[offset_i:offset_i + h,
                             offset_j:offset_j + w] = 255
                continue

            label1, data1 = mnist_iter.next()
            data1 = data1.astype(dtype=np.float)
            labels.add(label1)
            label_texture = random_texture((h, w), mean=potential_means[-1])
            potential_means = potential_means[0:-1]

            new_data[offset_i:offset_i + h,
                     offset_j:offset_j + w] = (
                data1 / 255.0 * label_texture +
                (255.0 - data1) / 255.0 *
                         new_data[offset_i:offset_i + h, offset_j:offset_j + w]
            )

            # segmentation data
            new_segm[offset_i:offset_i + h,
                     offset_j:offset_j + w][data1 > 159] = label1 + 1

            # mask out intermediate values
            new_segm[offset_i:offset_i + h,
                     offset_j:offset_j + w][
                np.logical_and(data1 > 95,
                               data1 <= 159)] = 255

            # mask out real background (not any numbers) - if desired
            if bgmask:
                new_segm[
                    offset_i:offset_i + h,
                    offset_j:offset_j + w][data1 == 0] = 255
    new_data = new_data.astype(np.uint8)
    assert new_data.dtype == np.uint8
    return (new_data, new_segm, labels)
