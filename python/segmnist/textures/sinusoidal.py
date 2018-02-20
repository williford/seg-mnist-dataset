from generator import TextureGenerator
import numpy as np
import math

class Sinusoidal(TextureGenerator):
    """ Just starting to implement...
    """
    def __init__(self, freq_dist, shape, valid_range=(0, 255)):
        self._freq_dist = freq_dist
        self._shape = shape
        self._valid_range = valid_range
        self._ntextures = None
        self._curr_texture_num = 0
        self._orientations = None

    def new_example(self, ntextures):
        """ Called once for every example """
        self._ntextures = ntextures
        self._curr_texture_num = 0
        # pick a random orientation first then add offset
        self._orientations = np.random.choice([math.pi/4, 3*math.pi/4], size=2)

    def generate(self):
        """ Called for every texture.
        The first call after new_example is assumed to the be background and
        every other call a foreground. The background is set to be a different
        orientation than the foreground.
        """
        orient = self._orientations[max(self._curr_texture_num, 1)]
        # def random_color_texture(self, shape, mean, var):
        self._curr_texture_num += 1
