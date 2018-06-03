from generator import TextureGenerator
import numpy as np


class FGModTexture(TextureGenerator):
    """ Oriented texture similar to that used in figure-ground modulation
    literature.
    """
    def __init__(self, shape,
                 independent_colors,
                 texture_alpha,
                 min_area_texture=0,
                 obj_rel_size=None,
                 valid_range=(0, 255),
                 fixed_colors=None, fixed_orientations=None):
        """ Args:
            independent_colors: each texture within an example can have a
                different color palette. This is a combination of
                color palette from the first texture (weight: 1 - alpha)
                and another random color palette (weight: alpha).
            min_area_texture: minimum area of mask in order for texture
                to be used (instead of just solid color). Smaller objects
                can be more difficult to see with the texture.
            fixed_colors: used to generate non-random textures
                (for external projects).
            fixed_orientations: prevent random orientations.
        """
        self._shape = shape
        self._valid_range = np.array(valid_range)
        self._ntextures = None
        self._curr_texture_num = 0
        self._linelen = 10
        self._slopes = None
        self._vert = None
        self._numlines = 50 * (shape[0] * shape[1] / self._linelen)
        self._step = 3
        self._step = 1
        assert 0 <= independent_colors <= 1, (
            "independent_colors must be in range [0,1]")
        self._independent_colors = independent_colors
        self._texture_alpha = texture_alpha
        self._min_area_texture = min_area_texture
        self.set_colors(fixed_colors)
        self.set_orientations(fixed_orientations)

    def set_colors(self, colors):
        self._fixed_colors = colors

    def set_orientations(self, orientations):
        self._fixed_orientations = orientations
        if self._fixed_orientations is not None:
            self._fixed_orientations = np.mod(self._fixed_orientations, np.pi)

            self._vert = np.full_like(self._fixed_orientations, np.nan)
            self._vert[
                np.isclose(self._fixed_orientations, np.pi / 2, 0.01)] = 1
            self._vert[
                np.isclose(self._fixed_orientations, 0, 0.01)] = 0
            assert not np.any(np.isnan(self._vert)), (
                'FGMod texture currently only support horizontal (0) and '
                'vertical orientations (pi/2).')

    def new_example(self, ntextures):
        """ Called once for every example """
        self._ntextures = ntextures
        self._curr_texture_num = 0

        # if self._fixed_orientations, then self._vert is already set
        if self._fixed_orientations is None:
            self._vert = np.random.choice([0, 1], size=2, replace=False)

        if self._fixed_colors is not None:
            if self._fixed_colors.ndim == 2:
                self._colors = np.repeat(
                    self._fixed_colors.reshape(
                        [1] + list(self._fixed_colors.shape)),
                    ntextures, axis=0)
            else:
                assert self._fixed_colors.ndim == 3, (
                    'FGMod textures must have 2 or 3 dimensions.')
                self._colors = self._fixed_colors
        else:
            low = np.min(self._valid_range)
            high = np.max(self._valid_range)
            mid1 = low * .55 + high * .45
            mid2 = low * .45 + high * .55
            self._colors = np.stack([
                np.stack([
                    np.random.uniform(low, mid1, size=3),
                    np.random.uniform(mid2, high, size=3),
                ]) for _ in range(ntextures)
            ])
            solid = np.random.uniform(low, high, size=3*ntextures).reshape(
                ntextures, 1, 3)
            texture_alpha = self._texture_alpha
            try:
                # allow texture alpha to be function
                texture_alpha = self._texture_alpha()
            except TypeError:
                # works for floats, tuple, etc.
                texture_alpha = np.random.uniform(
                    np.min(self._texture_alpha),
                    np.max(self._texture_alpha),
                )

            # initially calculate colors independently
            self._colors = (texture_alpha * self._colors +
                            (1-texture_alpha) * solid)
            # then treat "independent_colors" as alpha channel, with colors[0]
            # being the "background"
            for tex in range(1, len(self._colors)):
                self._colors[tex] = (
                    self._independent_colors * self._colors[tex] +
                    (1 - self._independent_colors) * self._colors[0])
        assert self._colors.ndim == 3
        assert not np.all(self._colors[0, 0] == self._colors[0, 1])

    def generate(self, mask=None):
        """ Called for every texture.
        The first call after new_example is assumed to the be background and
        every other call a foreground. The background is set to be a different
        orientation than the foreground.

        Args:
            mask: mask for the texture (or None, for background). Does not
                account for occlusion.
        """
        assert self._vert is not None, (
                'generate() must only be called after new_example()!')
        if mask is not None and self._min_area_texture > 0:
            num_pixels = np.sum(mask >= 1)
        else:
            num_pixels = self._shape[1] * self._shape[2]

        vert = self._vert[min(self._curr_texture_num, 1)]

        p = np.random.uniform()
        # color of the "canvas", texture painted over
        color = (
            p * self._colors[self._curr_texture_num, 0].reshape([3, 1]) +
            (1 - p) * self._colors[self._curr_texture_num, 1].reshape([3, 1]))
        texture = np.ones(self._shape) * color.reshape([3, 1, 1])

        if num_pixels < self._min_area_texture:
            self._curr_texture_num += 1
            return texture

        # "Bounding box"
        xmax = self._shape[2]
        ymax = self._shape[1]
        xmin = 0
        ymin = 0

        (x, y) = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

        p = np.random.uniform(size=self._numlines).reshape([1, self._numlines])
        colors = (p * self._colors[self._curr_texture_num, 0].reshape([3, 1]) +
                  (1 - p) * self._colors[self._curr_texture_num, 1].reshape([3, 1]))

        # centers
        if vert:
            x0 = np.random.choice(
                np.arange(xmin, xmax, self._step),
                size=self._numlines, replace=True)
            y0 = np.random.choice(
                np.arange(ymin - self._linelen, ymax, self._step),
                size=self._numlines, replace=True)
            x1 = x0
            y1 = y0 + self._linelen

            y0 = np.maximum(0, y0)
            y1 = np.minimum(ymax, y1)

            for i in range(self._numlines):
                texture[:, y0[i]:y1[i], x0[i]] = colors[:, i].reshape((3, 1))
        else:  # horizontal
            x0 = np.random.choice(
                np.arange(xmin - self._linelen, xmax, self._step),
                size=self._numlines, replace=True)
            y0 = np.random.choice(
                np.arange(ymin, ymax, self._step),
                size=self._numlines, replace=True)
            x1 = x0 + self._linelen
            y1 = y0

            x0 = np.maximum(0, x0)
            x1 = np.minimum(xmax, x1)

            for i in range(self._numlines):
                texture[:, y0[i], x0[i]:x1[i]] = colors[:, i].reshape((3, 1))

        self._curr_texture_num += 1

        return(texture)
