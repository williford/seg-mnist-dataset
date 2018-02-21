from generator import TextureGenerator
import numpy as np
import math
import ipdb as pdb

class TextureDispatcher(TextureGenerator):
    """ Contains other textures. For every example / image, it chooses a
        texture generator to use for that example.
    """
    def __init__(self):
        self._probtextgen = np.empty((0,))
        self._textgen = []
        self._example_gen = False
        self._curr_textgen = None

    def add_texturegen(self, prob, textgen):
        assert not self._example_gen, (
            'Adding texture generator after generating an '
            'example is not allowed')
        assert prob > 0, 'Probability of texture must be greater than 0.'
        self._probtextgen = np.append(self._probtextgen, prob)
        self._textgen.append(textgen)

    def new_example(self, ntextures):
        """ Called once for every example / image. """
        if not self._example_gen:  # first time function is called
            # normalize probabilities to add to 1
            self._probtextgen = self._probtextgen / np.sum(self._probtextgen)
            self._example_gen = True

        # choose which generator to use
        self._curr_textgen = np.random.choice(self._textgen,
                                              p=self._probtextgen)
        return self._curr_textgen.new_example(ntextures)

    def generate(self):
        """ Called for every texture.
        """
        return self._curr_textgen.generate()
