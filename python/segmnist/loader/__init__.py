import os
from segmnist.loader.mnist import MNIST
from segmnist.loader.shuffled_mnist import ShuffledMNIST

__all__ = ['MNIST', 'ShuffledMNIST']


def load_standard_MNIST(name, shuffle, path=None):
    if path is None:
        if "MNIST_PATH" in os.environ:
            path = os.environ["MNIST_PATH"]
        else:
            raise RuntimeError('Environment variable MNIST_PATH or ' +
                                'function parameter path must be defined')

    if shuffle:
        D = ShuffledMNIST
    else:
        D = MNIST

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
