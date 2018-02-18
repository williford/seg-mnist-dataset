from abc import ABCMeta
from abc import abstractmethod


class TextureGenerator(object):
    __metaclass__ = ABCMeta

    def new_example(self, ntextures):
        """ Called once for every example """
        pass

    @abstractmethod
    def generate(self, shape):
        """ Called for every texture """
        # def random_color_texture(self, shape, mean, var):
        pass
