from abc import ABCMeta
from abc import abstractmethod


class TextureGenerator(object):
    __metaclass__ = ABCMeta

    def new_example(self, ntextures):
        """ Called once for every example """
        pass

    @abstractmethod
    def generate(self, mask=None):
        """ Called for every texture

        Args:
            mask: mask that will be used to apply the texture (doesn't need to
                be used).
        """
        pass
