import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import math
from model_zoo import VGGEncoder
from torch.nn.modules.loss import _Loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pystrum.pynd.ndutils as nd
from abc import ABC, abstractmethod
from enum import Enum
import os
from collections import defaultdict, OrderedDict
from model_zoo.vgg import PretrainedVGG19FeatureExtractor


class PerceptualLoss2(torch.nn.Module):
    """
    https://github.com/ninatu/anomaly_detection/
    """
    def __init__(self,
                 reduction='mean',
                 img_weight=0,
                 feature_weights=None,
                 use_feature_normalization=False,
                 use_L1_norm=False,
                 use_relative_error=False):
        super(PerceptualLoss2, self).__init__()
        """
        We assume that input is normalized with 0.5 mean and 0.5 std
        """

        assert reduction in ['none', 'sum', 'mean', 'pixelwise']

        MEAN_VAR_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'vgg19_ILSVRC2012_object_detection_mean_var.pt')

        self.vgg19_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.vgg19_std = torch.Tensor([0.229, 0.224, 0.225])

        if use_feature_normalization:
            self.mean_var_dict = torch.load(MEAN_VAR_ROOT)
        else:
            self.mean_var_dict = defaultdict(
                lambda: (torch.tensor([0.0], requires_grad=False), torch.tensor([1.0], requires_grad=False))
            )

        self.reduction = reduction
        self.use_L1_norm = use_L1_norm
        self.use_relative_error = use_relative_error

        self.model = PretrainedVGG19FeatureExtractor()

        self.set_new_weights(img_weight, feature_weights)

    def set_reduction(self, reduction):
        self.reduction = reduction

    def forward(self, x, y):
        loss = 0
        ct = 1
        if len(x.shape) == 5:
            ct = 3
            input_ = x[0].permute(1, 0, 2, 3)
            target_ = y[0].permute(1, 0, 2, 3)

            loss += self._forward_(input_, target_)

            input_ = x[0].permute(2, 0, 1, 3)
            target_ = y[0].permute(2, 0, 1, 3)

            loss += self._forward_(input_, target_)

            x = x[0].permute(3, 0, 1, 2)
            y = y[0].permute(3, 0, 1, 2)

        loss += self._forward_(x, y)
        return loss / ct

    def _forward_(self, x, y):
        # pixel-wise prediction is implemented only if loss is obtained from one layer of vgg
        if self.reduction == 'pixelwise':
            assert (len(self.feature_weights) + (self.img_weight != 0)) == 1

        layers = list(self.feature_weights.keys())
        weights = list(self.feature_weights.values())

        x = self._preprocess(x)
        y = self._preprocess(y)

        f_x = self.model(x, layers)
        f_y = self.model(y, layers)

        loss = None

        if self.img_weight != 0:
            loss = self.img_weight * self._loss(x, y)

        for i in range(len(f_x)):
            # put mean, var on right device
            mean, var = self.mean_var_dict[layers[i]]
            mean, var = mean.to(f_x[i].device), var.to(f_x[i].device)
            self.mean_var_dict[layers[i]] = (mean, var)

            # compute loss
            norm_f_x_val = (f_x[i] - mean) / var
            norm_f_y_val = (f_y[i] - mean) / var

            cur_loss = self._loss(norm_f_x_val, norm_f_y_val)

            if loss is None:
                loss = weights[i] * cur_loss
            else:
                loss += weights[i] * cur_loss

        loss /= (self.img_weight + sum(weights))

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'pixelwise':
            loss = loss.unsqueeze(1)
            scale_h = x.shape[2] / loss.shape[2]
            scale_w = x.shape[3] / loss.shape[3]
            loss = F.interpolate(loss, scale_factor=(scale_h, scale_w), mode='bilinear')
            return loss
        else:
            raise NotImplementedError('Not implemented reduction: {:s}'.format(self.reduction))

    def set_new_weights(self, img_weight=0, feature_weights=None):
        self.img_weight = img_weight
        if feature_weights is None:
            self.feature_weights = OrderedDict({})
        else:
            self.feature_weights = OrderedDict(feature_weights)

    def _preprocess(self, x):
        assert len(x.shape) == 4

        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)

        # denormalize
        vector = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).to(x.device)
        x = x * vector + vector

        # normalize
        x = (x - self.vgg19_mean.reshape(1, 3, 1, 1).to(x.device)) / self.vgg19_std.reshape(1, 3, 1, 1).to(x.device)
        return x

    def _loss(self, x, y):
        if self.use_L1_norm:
            norm = lambda z: torch.abs(z)
        else:
            norm = lambda z: z * z

        diff = (x - y)

        if not self.use_relative_error:
            loss = norm(diff)
        else:
            means = norm(x).mean(3).mean(2).mean(1)
            means = means.detach()
            loss = norm(diff) / means.reshape((means.size(0), 1, 1, 1))

        # perform reduction
        if self.reduction == 'pixelwise':
            return loss.mean(1)
        else:
            return loss.mean(3).mean(2).mean(1)


class ProgGrowStageType(Enum):
    trns = 'trns'  # translation stage - increasing resolution twice
    stab = 'stab'  # stabilization stage - training at a fixed resolution


class AbstractPGLoss(torch.nn.Module, ABC):
    def __init__(self, max_resolution):
        super().__init__()

        self._resolution = max_resolution
        self._stage = ProgGrowStageType.stab
        self._progress = 0

    @abstractmethod
    def set_stage_resolution(self, stage, resolution):
        pass

    @abstractmethod
    def set_progress(self, progress):
        pass

    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def set_reduction(self, reduction):
        pass


class PGPerceptualLoss(AbstractPGLoss):
    def __init__(self,  weights_per_resolution, max_resolution=128,
                 reduction='mean',
                 use_smooth_pg=False,
                 use_feature_normalization=False,
                 use_L1_norm=False,
                 use_relative_error=False):
        super(PGPerceptualLoss, self).__init__(max_resolution)
        self._max_resolution = max_resolution
        weights_per_resolution = dict()
        weights_per_resolution[128] = dict()
        weights_per_resolution[128]['img_weight'] = 0
        weights_per_resolution[128]['feature_weights'] = {'r22': 1, 'r32': 1, 'r42': 1, 'r52': 1}
        self._weights_per_resolution = weights_per_resolution
        self._use_smooth_pg = use_smooth_pg
        self._loss = PerceptualLoss2(reduction=reduction,
                                     use_feature_normalization=use_feature_normalization,
                                     use_L1_norm=use_L1_norm,
                                     use_relative_error=use_relative_error)

        self._resolution = self._max_resolution
        self._stage = ProgGrowStageType.stab
        self._progress = 0

    def set_stage_resolution(self, stage, resolution):
        self._stage = stage
        self._resolution = resolution
        self._progress = 0

    def set_progress(self, progress):
        self._progress = progress

    def set_reduction(self, reduction):
        self._loss.reduction = reduction

    def forward(self, x, y):
        self._loss.set_new_weights(**self._weights_per_resolution[self._resolution])
        loss = self._loss(x, y)

        if self._use_smooth_pg:
            if self._stage == ProgGrowStageType.trns and self._progress < 1:
                prev_res = int(self._resolution / 2)
                self._loss.set_new_weights(**self._weights_per_resolution[prev_res])

                x = torch.nn.functional.upsample(x, scale_factor=0.5, mode='bilinear')
                y = torch.nn.functional.upsample(y, scale_factor=0.5, mode='bilinear')

                prev_loss = self._loss(x, y)
                loss = (1 - self._progress) * prev_loss + self._progress * loss

        return loss


class PGRelativePerceptualL1Loss(PGPerceptualLoss):
    def __init__(self, weights_per_resolution, max_resolution=128, reduction='mean', use_smooth_pg=False):
        super().__init__(
            max_resolution, weights_per_resolution,
            reduction=reduction,
            use_smooth_pg=use_smooth_pg,
            use_feature_normalization=False,
            use_L1_norm=True,
            use_relative_error=True)


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    code from https://github.com/voxelmorph/voxelmorph

    Licence :
        Apache License Version 2.0, January 2004 - http://www.apache.org/licenses/
    """

    def __init__(self, win=None):
        self.win = win

    def __call__(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1-torch.mean(cc)


class DisplacementRegularizer(torch.nn.Module):
    """
        code from https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/
    """

    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:,i,...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms)/3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, disp):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


class DisplacementRegularizer2D(torch.nn.Module):
    """
    code from https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/

    License:
        MIT License
    """
    def __init__(self, energy_type='gradient-l2'):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:,i,...]) for i in [0, 1]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy)
        else:
            norms = dTdx**2 + dTdy**2
        return torch.mean(norms)/2.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        return torch.mean(dTdxx**2 + dTdyy**2 + 2*dTdxy**2)

    def forward(self, disp):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


class JacobianDeterminant(torch.nn.Module):
    def __init__(self):
        super(JacobianDeterminant, self).__init__()

    def forward(self, disp):
        """
        jacobian determinant of a displacement field.
        NB: to compute the spatial gradients, we use np.gradient.
        Parameters:
            disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
                  where vol_shape is of len nb_dims
        Returns:
            jacobian determinant (scalar)
        """

        # check inputs
        volshape = disp.shape[:-1]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

        # compute grid
        grid_lst = nd.volsize2ndgrid(volshape)
        grid = np.stack(grid_lst, len(volshape))

        # compute gradients
        J = np.gradient(disp + grid)

        # 3D glow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

            return Jdet0 - Jdet1 + Jdet2

        else:  # must be 2

            dfdx = J[0]
            dfdy = J[1]

            return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


class PerceptualLoss(_Loss):
    """
    """

    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.loss_network = VGGEncoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        loss_pl = 0
        ct_pl = 0
        # input = (input + 1) / 2
        # target = (target + 1) / 2
        if len(input.shape) == 5:
            input_ = input[0].permute(1, 0, 2, 3)
            target_ = target[0].permute(1, 0, 2, 3)

            input_features = self.loss_network(input_.repeat(1, 3, 1, 1))
            output_features = self.loss_network(target_.repeat(1, 3, 1, 1))

            for output_feature, input_feature in zip(output_features, input_features):
                loss_pl += F.mse_loss(output_feature, input_feature)
                ct_pl += 1

            input_ = input[0].permute(2, 0, 1, 3)
            target_ = target[0].permute(2, 0, 1, 3)

            input_features = self.loss_network(input_.repeat(1, 3, 1, 1))
            output_features = self.loss_network(target_.repeat(1, 3, 1, 1))

            for output_feature, input_feature in zip(output_features, input_features):
                loss_pl += F.mse_loss(output_feature, input_feature)
                ct_pl += 1

            input = input[0].permute(3, 0, 1, 2)
            target = target[0].permute(3, 0, 1, 2)

        input_features = self.loss_network(input.repeat(1, 3, 1, 1))
        output_features = self.loss_network(target.repeat(1, 3, 1, 1))

        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
            ct_pl += 1

        return loss_pl / ct_pl



class VGGLoss(torch.nn.Module):
    def __init__(self, device, feature_layer=35):
        super(VGGLoss, self).__init__()
        # Feature extracting using vgg19
        cnn = torchvision.models.vgg19(pretrained=True).to(device)
        self.features = torch.nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])
        self.MSE = torch.nn.MSELoss().to(device)

    def normalize(self, tensors, mean, std):
        if not torch.is_tensor(tensors):
            raise TypeError('tensor is not a torch image.')
        for tensor in tensors:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return tensors

    def forward(self, input, target):
        ct = 1e-8
        mse_loss = 0
        if len(input.shape) == 5:
            # 3D case:
            for axial_slice in range(input.shape[-1]):
                x = input[:, :, :, :, axial_slice]
                y = target.detach()[:, :, :, :, axial_slice]
                if x.shape[1] == 1:
                    x = x.expand(-1, 3, -1, -1)
                    y = y.expand(-1, 3, -1, -1)
                # [-1: 1] image to  [0:1] image---------------------------------------------------(1)
                x = (x+1) * 0.5
                y = (y+1) * 0.5
                # https://pytorch.org/docs/stable/torchvision/models.html
                x.data = self.features(self.normalize(x.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                y.data = self.features(self.normalize(y.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                mse_loss += F.mse_loss(x, y.data)
        return mse_loss / ct


class SSIMLoss(_Loss):
    """
    """
    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, z_dict = None):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        loss_ssim = 0
        ct_ssim = 0
        if len(input.shape) == 5:
            input_ = input[0].permute(1, 0, 2, 3)
            target_ = target[0].permute(1, 0, 2, 3)

            loss_ssim += ssim(input_, target_, data_range=1., size_average=True)  # return a scalar
            ct_ssim += 1
            input_ = input[0].permute(2, 0, 1, 3)
            target_ = target[0].permute(2, 0, 1, 3)

            loss_ssim += ssim(input_, target_, data_range=1., size_average=True)  # return a scalar
            ct_ssim += 1

            input = input[0].permute(3, 0, 1, 2)
            target = target[0].permute(3, 0, 1, 2)

        loss_ssim += ssim(input, target, data_range=1., size_average=True)  # return a scalar
        ct_ssim += 1
        return 1 - (loss_ssim / ct_ssim)
