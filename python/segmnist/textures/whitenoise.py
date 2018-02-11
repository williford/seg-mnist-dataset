from generator import TextureGenerator
import numpy as np
import math


class WhiteNoise(TextureGenerator):
    def __init__(self, mean, var, valid_range=(0, 255)):
        self._mean_dist = mean
        self._var_dist = var
        self._valid_range = valid_range

    def generate(self, shape):

        assert shape[0] == len(mean)
        assert shape[0] == len(var)
        shape0 = list(shape)
        shape0[0] = 1
        textures = [self.random_texture(shape0)
                    for i in range(shape[0])]
        ret = np.concatenate(textures)
        return ret

    def random_texture_1c(self, shape):
        mean = self._mean_dist()
        var = self._var_dist()
        if var == 0:
            texture = np.ones(shape) * mean
        else:
            texture = np.random.normal(loc=mean, scale=math.sqrt(var), size=shape)

        texture[texture > 255] = 255
        texture[texture < 0] = 0
        return texture
