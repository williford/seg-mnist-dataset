from abc import ABC
from abc import abstractmethod


class TextureGenerator(ABC):
    def new_example(self, ntextures):
        pass

    @abstractmethod
    def generate(self, shape):
        # def random_color_texture(self, shape, mean, var):
        pass
