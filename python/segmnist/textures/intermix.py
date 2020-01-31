from segmnist.textures.generator import TextureGenerator
import numpy as np

class IntermixTexture(TextureGenerator):
    """ Random texture everytime a texture is generated (multiple
        times per example).
    """
    def __init__(self):
        self._probtextgen = np.empty((0,))
        self._textgens = []
        self._example_gen = False 
    def add_texturegen(self, prob, textgens):
        assert not self._example_gen, (
            'Adding texture generator after generating an '
            'example is not allowed')
        assert prob > 0, 'Probability of texture must be greater than 0.'
        self._probtextgen = np.append(self._probtextgen, prob)
        self._textgens.append(textgens)

    def new_example(self, ntextures):
        """ Called once for every example / image. """
        if not self._example_gen:  # first time function is called
            # normalize probabilities to add to 1
            self._probtextgen = self._probtextgen / np.sum(self._probtextgen)
            self._example_gen = True
        return

    def generate(self, mask=None):
        """ Called for every texture.
        """
        # choose which generator to use for each example,
        curr_textgen = np.random.choice(self._textgens,
                                        p=self._probtextgen)
        curr_textgen.new_example(1)

        return curr_textgen.generate(mask)

    def generators(self):
        return self._textgens
