import numpy as np
from scipy.linalg import expm, norm
from skimage import color


class RandomRotation:

    def __init__(self, axis=None, max_theta=180):
        self.axis = axis
        self.max_theta = max_theta

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    @staticmethod
    def get_params():
        return np.random.rand(3), np.random.rand(1), np.random.rand(3), np.random.rand(1)

    def __call__(self, frames):
        out_frames = []
        axis_1, factor_1, axis_2, factor_2 = self.get_params()
        if self.axis is not None:
            axis = self.axis
        else:
            axis = axis_1 - 0.5

        for xyz in frames:

            R = self._M(axis, (np.pi * self.max_theta / 180) * 2 *
                        (factor_1 - 0.5))
            R_n = self._M(
                axis_2 - 0.5,
                (np.pi * 30 / 180) * 2 * (factor_2 - 0.5))

            out_frames.append(xyz @ R @ R_n)

        return out_frames


class RandomScale:

    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    @staticmethod
    def get_factor():
        return np.random.rand(1)

    def __call__(self, frames):
        factor = self.get_factor()
        out_frames = []
        for xyz in frames:
            s = self.scale * factor + self.bias
            out_frames.append(xyz * s)

        return out_frames


class RandomShear:

    @staticmethod
    def get_matrix():
        return np.random.randn(3, 3)

    def __call__(self, frames):
        matrix = self.get_matrix()
        out_frames = []
        for xyz in frames:
            T = np.eye(3) + 0.1 * matrix #original 0.1
            out_frames.append(xyz @ T)
        return out_frames


class RandomTranslation:
    def __init__(self, scale):
        self.scale = scale

    @staticmethod
    def get_factors():
        return np.random.randn(1, 3)

    def __call__(self, frames):
        factors = self.get_factors()
        out_frames = []
        for xyz in frames:
            trans = self.scale * factors
            out_frames.append(xyz + trans)
        return out_frames


class RandomGaussianNoise:

    def __init__(self, mean=0, var=0.001):
        self.mean = mean
        self.var = var

    def __call__(self, feats):
        out_colors = []

        for colors in feats:
            noise = np.random.normal(self.mean, self.var ** 0.5, colors.shape)
            out_colors.append(colors+noise)

        return out_colors


class RandomValue:

    def __init__(self, min=-0.2, max=0.2):
        self.scale = max-min
        self.bias = min

    @staticmethod
    def get_offset():
        return np.random.rand()

    def __call__(self, feats):

        out_colors = []
        offset = self.get_offset()

        for colors in feats:
            colors_hsv = color.rgb2hsv(colors) # transform colors to hsv space
            colors_hsv[..., -1] += self.scale * offset + self.bias # apply augmentation
            colors_rgb = color.hsv2rgb(colors_hsv) # transform colors back to rgb space
            out_colors.append(colors_rgb)

        return out_colors


class RandomSaturation:

    def __init__(self, min=-0.15, max=0.15):
        self.scale = max-min
        self.bias = min

    @staticmethod
    def get_offset():
        return np.random.rand()

    def __call__(self, feats):

        out_colors = []
        offset = self.get_offset()

        for colors in feats:
            colors_hsv = color.rgb2hsv(colors)  # transform colors to hsv space
            colors_hsv[:, 1] += self.scale * offset + self.bias  # apply augmentation
            colors_rgb = color.hsv2rgb(colors_hsv)  # transform colors back to rgb space
            out_colors.append(colors_rgb)

        return out_colors
