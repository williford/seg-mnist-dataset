from generator import TextureGenerator
import numpy as np
import math


class WhiteNoise(TextureGenerator):
    def __init__(self, mean_dist, var_dist, shape, valid_range=(0, 255)):
        self._mean_dist = mean_dist
        self._var_dist = var_dist
        self._valid_range = valid_range
        self._shape = shape

    def generate(self):

        shape0 = list(self._shape)
        shape0[0] = 1
        textures = [self.random_texture_1c(shape0)
                    for i in range(self._shape[0])]
        ret = np.concatenate(textures)
        return ret

    def random_texture_1c(self, shape):
        mean = self._mean_dist()
        var = self._var_dist()
        if var == 0:
            texture = np.ones(shape) * mean
        else:
            texture = np.random.normal(loc=mean, scale=math.sqrt(var), size=shape)

        texture[texture > self._valid_range[1]] = self._valid_range[1]
        texture[texture < self._valid_range[0]] = self._valid_range[0]
        return texture
