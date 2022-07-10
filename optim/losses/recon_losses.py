import torch
import torch.nn as nn

import kornia.metrics as metrics
import ignite.metrics.psnr


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"
        self.range = 1

    def __call__(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2)
        return 20 * torch.log10(self.range / torch.sqrt(mse))


def ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    reduction: str = 'mean',
    padding: str = 'same',
) -> torch.Tensor:
    r"""Function that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim` for details about SSIM.

    Args:
        img1: the first input image with shape :math:`(B, C, H, W)`.
        img2: the second input image with shape :math:`(B, C, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> loss = ssim_loss(input1, input2, 5)
    """
    # compute the ssim map
    ssim_map: torch.Tensor = metrics.ssim(img1, img2, window_size, max_val, eps, padding)

    # compute and reduce the loss
    loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        pass
    return loss


class SSIMLoss(nn.Module):
    r"""Create a criterion that computes a loss based on the SSIM measurement.

    The loss, or the Structural dissimilarity (DSSIM) is described as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    See :meth:`~kornia.losses.ssim_loss` for details about SSIM.

    Args:
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
        The loss based on the ssim index.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> criterion = SSIMLoss(5)
        >>> loss = criterion(input1, input2)
    """

    def __init__(
        self, window_size: int, max_val: float = 1.0, eps: float = 1e-12, reduction: str = 'mean', padding: str = 'same'
    ) -> None:
        super().__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps: float = eps
        self.reduction: str = reduction
        self.padding: str = padding

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim_loss(img1, img2, self.window_size, self.max_val, self.eps, self.reduction, self.padding)


## Histogram
import torch
from torch import nn, sigmoid


class HistLayerBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.K = 256
        self.L = 1 / self.K  # 2 / K -> if values in [-1,1] (Paper)
        self.W = self.L / 2.5

        self.mu_k = (self.L * (torch.arange(self.K) + 0.5)).view(-1, 1).cuda()

    def phi_k(self, x, L, W):
        return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)

    def compute_pj(self, x, mu_k, K, L, W):
        # we assume that x has only one channel already
        # flatten spatial dims (256*256)
        x = x.reshape(x.size(0), 1, -1)
        x = x.repeat(1, K, 1)  # construct K channels

        # apply activation functions
        return self.phi_k(x - mu_k, L, W)


class SingleDimHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N = x.size(1) * x.size(2)
        pj = self.compute_pj(x, self.mu_k, self.K, self.L, self.W)
        return pj.sum(dim=2) / N


class JointHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        N = x.size(1) * x.size(2)
        p1 = self.compute_pj(x, self.mu_k, self.K, self.L, self.W)
        p2 = self.compute_pj(y, self.mu_k, self.K, self.L, self.W)
        return torch.matmul(p1, torch.transpose(p2, 1, 2)) / N

class EarthMoversDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # input has dims: (Batch x Bins)
        bins = x.size(1)
        r = torch.arange(bins)
        s, t = torch.meshgrid(r, r)
        tt = t >= s

        cdf_x = torch.matmul(x, tt.float().cuda())
        cdf_y = torch.matmul(y, tt.float().cuda())

        return -torch.mean(torch.sum(torch.square(cdf_x - cdf_y), dim=1))


class HistogramLoss(nn.Module):

    def __init__(self, l1=1.0, l2=1.0, l3=1.0,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True):
        super(HistogramLoss, self).__init__()
        self.emd = EarthMoversDistanceLoss()
        # self.mi = MutualInformationLoss()
        # self.mi = LNCCLoss()
        # self.mi = MILossGaussian(vmin, vmax, num_bins, sample_ratio, normalised)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def __call__(self, tar, warped_src, tar_mask = None, src_mask = None, src=None):

        hist_tar = SingleDimHistLayer()(tar[:, 0, ...])
        hist_wr_s = SingleDimHistLayer()(warped_src[:, 0, ...])
        # hist_src = SingleDimHistLayer()(src[:, 0, ...])

        # hist1_loss = self.emd(hist_tar, hist_wr_s)
        hist1_loss = self.emd(hist_tar[..., 2:], hist_wr_s[..., 2:])

        return hist1_loss

        # joint_hist = JointHistLayer()(tar[:, 0, ...], warped_src[:, 0, ...])
        # joint_hist2 = JointHistLayer()(src[:, 0, ...], warped_src[:, 0, ...])

        # mi_loss = self.mi(hist_tar, hist_wr_s, joint_hist)
        # mi_loss2 = self.mi(hist_src, hist_wr_s, joint_hist2)
        # # mi_loss = self.mi(tar, warped_src, tar_mask, src_mask)
        #
        # loss = self.l1*mi_loss + self.l2*hist1_loss + self.l3*mi_loss2
        #
        # return loss