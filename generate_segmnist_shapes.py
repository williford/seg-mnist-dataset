#!/usr/bin/env python

import os
os.environ["MNIST_PATH"] = "/media/Data/Documents/Enkium/Projects/NIN/Data/"

import sys
sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python/')
sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python/segmnist')
sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python/segmnist/textures')
sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python/segmnist/loader')
sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python/segmnist/caffe')
sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python/segmnist/pytorch')
# sys.path.append('/media/Data/Documents/Enkium/Projects/NIN/seg-mnist-dataset/python/segmnist/loader')


# from segmnist.static_segmnistshapes import generate_caffe_segmnist_shapes # original
from python.segmnist.static_segmnistshapes import generate_caffe_segmnist_shapes


if __name__ == "__main__":
    # generate_segmnist_shapes()
    generate_caffe_segmnist_shapes()
