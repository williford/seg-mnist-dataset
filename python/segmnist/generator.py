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
import random


def random_color_texture(shape, mean, var):
    assert shape[0] == len(mean)
    assert shape[0] == len(var)
    shape0 = list(shape)
    shape0[0] = 1
    textures = [random_texture(shape[1:], mean[i], var[i]).reshape(shape0)
                for i in range(shape[0])]
    ret = np.concatenate(textures)
    return ret


def random_texture(shape,
                   mean=np.random.uniform(0, 255),
                   var=np.random.gamma(1, 25)):
    if var == 0:
        texture = np.ones(shape) * mean
    else:
        texture = np.random.normal(loc=mean, scale=math.sqrt(var), size=shape)

    texture[texture > 255] = 255
    texture[texture < 0] = 0
    return texture


# Generate image where digits can appear anywhere, even overlapping.
def generate_textured_image(mnist_iter, grid, mnist_shape=(28, 28),
                           bgmask=False, nchannels=1):

    num_elem = np.sum(grid)
    # num_means = 10
    # potential_means = np.arange(num_means) * (255. / (num_means - 1))
    # np.random.shuffle(potential_means)

    H = grid.shape[0] * mnist_shape[0]
    W = grid.shape[1] * mnist_shape[1]
    new_data = random_color_texture(
        (nchannels, H, W),
        mean=np.random.randint(256, size=nchannels),
        var=np.random.gamma(1, 25, size=nchannels))

    new_segm = np.zeros((H, W), dtype=np.uint8)
    labels = set()

    for elem in range(num_elem):
        scale = random.uniform(0.5, 1.5)
        h = int(round(scale * mnist_shape[0]))
        w = int(round(scale * mnist_shape[1]))

        # allow digits to be partially outside image
        min_pos_i = - h // 4
        min_pos_j = - w // 4
        max_pos_i = H - h + h // 4
        max_pos_j = W - w + w // 4
        offset_i = random.randrange(min_pos_i, max_pos_i)
        offset_j = random.randrange(min_pos_j, max_pos_j)

        slice_chan = slice(nchannels)
        slice_dest_i = slice(max(0, offset_i), min(H, offset_i + h))
        slice_dest_j = slice(max(0, offset_j), min(W, offset_j + w))

        label1, data0 = mnist_iter.next()
        labels.add(label1)

        data1 = scipy.misc.imresize(data0, (h, w), 'bicubic')

        # Calculate indices within digit
        digit_offset_i = abs(min(0, offset_i))
        digit_offset_j = abs(min(0, offset_j))
        slice_src_i = slice(digit_offset_i, min(H, offset_i + h) - offset_i)
        slice_src_j = slice(digit_offset_j, min(W, offset_j + w) - offset_j)
        digit = data1[slice_src_i, slice_src_j].astype(dtype=np.float)

        digit_texture = random_color_texture(
            (nchannels, h, w),
            mean=np.random.randint(256, size=nchannels),
            var=np.random.gamma(1, 25, size=nchannels)
        )
        digit_texture = digit_texture[slice_chan, slice_src_i, slice_src_j]

        new_data[slice_chan, slice_dest_i, slice_dest_j] = (
            np.multiply(digit / 255.0, digit_texture) +
            np.multiply((255.0 - digit) / 255.0,
                        new_data[slice_chan, slice_dest_i, slice_dest_j])
        )

        # segmentation data
        new_segm[slice_dest_i, slice_dest_j][digit > 159] = label1 + 1
        assert new_segm.max() > 0 or num_elem == 0

        # mask out intermediate values
        new_segm[slice_dest_i, slice_dest_j][
            np.logical_and(digit > 95,
                           digit <= 159)] = 255

        # mask out real background (not any numbers) - if desired
        if bgmask:
            new_segm[slice_dest_i, slice_dest_j][digit == 0] = 255
    new_data = new_data.astype(np.uint8)
    assert new_data.dtype == np.uint8
    return (new_data, new_segm, labels)



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
            digit_texture = random_texture((h, w), mean=potential_means[-1])
            potential_means = potential_means[0:-1]

            new_data[offset_i:offset_i + h,
                     offset_j:offset_j + w] = (
                data1 / 255.0 * digit_texture +
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
