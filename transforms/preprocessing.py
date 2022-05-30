import numpy as np
import torch
from monai.transforms import Transform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from torchvision.io.image import read_image


class ReadImage(Transform):
    """
    Transform to read image, see torchvision.io.image.read_image
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, path: str) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return read_image(path)


class To01:
    """
    Convert the input to [0,1] scale

    """
    def __init__(self, max_val=255.0):
        self.max_val = max_val
        super(To01, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return img/self.max_val


class ToRGB:
    """
    Convert the input to an np.ndarray from grayscale to RGB

    """
    def __init__(self, r_val, g_val, b_val):
        self.r_val = r_val
        self.g_val = g_val
        self.b_val = b_val
        super(ToRGB, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        r = np.multiply(img, self.r_val).astype(np.uint8)
        g = np.multiply(img, self.g_val).astype(np.uint8)
        b = np.multiply(img, self.b_val).astype(np.uint8)

        img_color = np.dstack((r, g, b))
        return img_color


class AddChannelIfNeeded(Transform):
    """
    Adds a 1-length channel dimension to the input image, if input is 2D
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if len(img.shape) == 2:
            print(f'Added channel: {(img[None,...].shape)}')
            return img[None, ...]
        else:
            return img


class AssertChannelFirst(Transform):
    """
    Assert channel is first and permute otherwise
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        assert len(img.shape) == 3,  f'AssertChannelFirst:: Image should have 3 dimensions, instead of {len(img.shape)}'
        if img.shape[0] == img.shape[1] and img.shape[0] != img.shape[2]:
            print(f'Permuted channels {(img.permute(2,0,1)).shape}')
            return img.permute(2, 0, 1)
        else:
            return img


def visualize_slice(img):
    """
    Normalize for visualization
    """
    # img = img[0][0]
    img[0][0] = 0
    img[0][1] = 1
    return img