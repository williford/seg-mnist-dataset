from . import MNIST
import random


class ShuffledMNIST(MNIST):
    """ Class to read MNIST dataset and iterate shuffled examples.
    """
    def __init__(self, *args, **kwargs):
        super(ShuffledMNIST, self).__init__(*args, **kwargs)

    def iter(self, max_iter=float("inf")):
        """ Randomly iterate over the image and file pairs (sampling with
            replacement), potentially forever.
        """
        if self.lbl is None or self.img is None:
            raise RuntimeError("Dataset must be loaded before " +
                               "calling random().")

        while max_iter > 0:
            # set seed before each random range call
            random.seed(self.seed_current)
            self.seed_current += 1

            i = random.randrange(len(self.lbl))
            yield (self.lbl[i], self.img[i])
            max_iter -= 1

    def is_shuffled(self):
        return True
