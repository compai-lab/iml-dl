#!/usr/bin/env python

"""
Loss functions from Voxelmorph

If you use this code, please cite the following, and read function docs for further info/citations.
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 
    or
    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 
Copyright 2020 Adrian V. Dalca
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    Correction adapted from:
    https://github.com/voxelmorph/voxelmorph/pull/358/files
    """

    def __init__(self, win=None,  eps=1e-5):
        self.win = win
        self.eps = 1e-5

    def __call__(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, nb_feats, *vol_shape]
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
        # u_I = I_sum / win_size
        # u_J = J_sum / win_size
        #
        # cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        # I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        # J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        cross = IJ_sum - I_sum * J_sum / win_size
        cross = torch.clamp(cross, min=self.eps)
        I_var = I2_sum - I_sum * I_sum / win_size
        I_var = torch.clamp(I_var, min=self.eps)
        J_var = J2_sum - J_sum * J_sum / win_size
        J_var = torch.clamp(J_var, min=self.eps)
        cc = (cross / I_var) * (cross / J_var)

        # cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def __call__(self, y_true, y_pred):
        if y_true is None:
            y_true = torch.zeros_like(y_pred)
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def __call__(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def __call__(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad_2D:
    """
    2-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def __call__(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        # dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            # dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) # + torch.mean(dz)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

from scipy.ndimage import sobel

class Tenengrad():

    def __call__(self, _ ,  img):
        '''
            Parameters
            ----------
            img : numpy array
                image for which the metrics should be calculated.
            brainmask_fl : numpy array or list, optional
                If a non-empty numpy array is given, this brainmask will be used to
                mask the images before calculating the metrics. If an empty list is
                given, no mask is applied. The default is [].
            Returns
            -------
            tg : float
                Tenengrad measure of the input image.
            '''
        # image needs to be in floating point numbers in order for gradient to be
        # correctly calculated
        img = img.astype(np.float)

        # calulate gradients:
        grad_x = sobel(img, axis=0, mode='reflect')
        grad_y = sobel(img, axis=1, mode='reflect')
        grad_z = sobel(img, axis=2, mode='reflect')
        nabla_ab = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
        nabla_abs = nabla_ab.flatten()

        # apply flattened brainmask:
        # if len(brainmask) > 0:
        #     nabla_abs = nabla_abs[brainmask.flatten() > 0]

        return np.mean(nabla_abs ** 2)

#### Jacobian Determinant
# Code adapted from
# https://github.com/voxelmorph/voxelmorph/issues/82#issuecomment-523447568
# and
# https://github.com/adalca/pystrum/blob/master/pystrum/pynd/ndutils.py

def jacdet_loss(flow_tensor):

        jacdet = flow_to_jacdet(flow_tensor)

        nb_neg = (jacdet < 0).sum()
        nb_total = np.prod(jacdet.shape)

        return nb_neg / nb_total


def flow_to_jacdet(flow):

    vol_size = flow.shape[:-1]
    n_dims = len(vol_size)

    assert n_dims in (2,3)

    grid = np.stack(volsize2ndgrid(vol_size), len(vol_size))
    J = np.gradient(flow + grid)


    if n_dims == 3:

        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1])
        Jdet1 = dx[:,:,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0])
        Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0])

        Jdet = Jdet0 - Jdet1 + Jdet2

        return Jdet

    else:

        dfdx = J[0]
        dfdy = J[1]

        Jdet = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

    return Jdet

def eval_jacdet(jacdet):
    '''
    Input:
    jadet: numpy array, Jacobian determinant
    Output:
    overall percentage of values below zero
    '''

    nb_neg = (jacdet <0).sum()
    nb_total = np.prod(jacdet.shape)

    return nb_neg / nb_total


def volsize2ndgrid(volsize):
    """
    return the dense nd-grid for the volume with size volsize
    essentially return the ndgrid fpr
    """
    ranges = [np.arange(e) for e in volsize]
    return ndgrid(*ranges)


def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)


### MeanStream (for groupwise registration, regularization of multiple displacement fields) ###
class MeanStream:
    """
    Maintain stream of data mean.

    Adapted from:
    https://github.com/adalca/neurite/blob/741c95d794f82f91d568eb28ef0c8d1f36d509cd/neurite/tf/layers.py#L1721

        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019.

    """

    def __init__(self, inshape, cap = 100, **kwargs): #needs to be called before training
        super(MeanStream, self).__init__(**kwargs)

        self.mean = torch.zeros(*inshape).to('cuda') # needs to be redefined for 3D
        self.cap = cap
        self.count = 0
        self.last_deformations = []

    def get_mean_and_count(self):
        return self.mean, self.count

    def get_last_deformations(self):
        return self.last_deformations

    def append_deformation(self, phi):

        self.last_deformations.append(phi)

        if len(self.last_deformations) >= self.cap:
            self.last_deformations.pop()
            assert (len(self.last_deformations) == self.cap-1) # -1 because in training loop a further one is added

    def update_with_mean(self, x, training = None):

        # Get batch size_
        batch_size = x.shape[0]

        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        # if training is False:
        #     return np.minimum(1., self.count / self.cap) * self.mean

        # Only if training is going on, update the class mean:
        self.mean = new_mean
        self.count = new_count

        # First inputs should not matter that much (if count below cap -> weight the loss less)
        # return np.minimum(1., self.count / self.cap) * self.mean


def _mean_update(pre_mean, pre_count, x, pre_cap=None):
    '''
    Compute new mean
    '''
    this_sum = torch.sum(x, dim=0, keepdim=True)
    this_bs = x.shape[0]

    # increase count and compute weights
    new_count = pre_count + this_bs
    alpha = this_bs / np.minimum(new_count, pre_cap)

    # compute new mean. Note that once we reach self.cap (e.g. 1000),
    # the 'previous mean' matters less
    new_mean = pre_mean * (1 - alpha) + (this_sum / this_bs) * alpha

    return (new_mean, new_count)

### MIND metric and helper functions
# from pull request for VxmMorph https://github.com/voxelmorph/voxelmorph/pull/145
def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = torch.nn.ReplicationPad3d(dilation)
    rpad2 = torch.nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                       kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind


def mind_loss(x, y):
    return torch.mean((MINDSSC(x) - MINDSSC(y)) ** 2)

class MINDLoss():

    def __call__(self, y_pred, y_true) :
        # pad with one dimension since MIND expects 3D
        if len(y_pred.shape) == 4:
            y_pred, y_true = y_pred[..., None], y_true[..., None]

        return mind_loss(y_pred, y_true)


## JacDet
import numpy as np

## Code adapted from
# https://github.com/voxelmorph/voxelmorph/issues/82#issuecomment-523447568
# and
# https://github.com/adalca/pystrum/blob/master/pystrum/pynd/ndutils.py

def flow_to_jacdet(flow):

    vol_size = flow.shape[:-1]
    n_dims = len(vol_size)

    assert n_dims in (2,3)

    grid = np.stack(volsize2ndgrid(vol_size), len(vol_size))
    J = np.gradient(flow + grid)


    if n_dims == 3:

        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1])
        Jdet1 = dx[:,:,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0])
        Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0])

        Jdet = Jdet0 - Jdet1 + Jdet2

        return Jdet

    else:

        dfdx = J[0]
        dfdy = J[1]

        Jdet = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

    return Jdet

def eval_jacdet(jacdet):
    '''
    Input:
    jadet: numpy array, Jacobian determinant
    Output:
    overall percentage of values below zero
    '''

    nb_neg = (jacdet <0).sum()
    nb_total = np.prod(jacdet.shape)

    return nb_neg / nb_total


def volsize2ndgrid(volsize):
    """
    return the dense nd-grid for the volume with size volsize
    essentially return the ndgrid fpr
    """
    ranges = [np.arange(e) for e in volsize]
    return ndgrid(*ranges)


def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)

### REcon loss
# LPIPS
import lpips

class LPIPS:

    def __init__(self, net = 'alex', eval_mode=True, device = 'cuda'):
        # For training eval_mode needs to be False
        self.lpips = lpips.LPIPS(net=net, eval_mode=eval_mode, verbose=False).to(device)

    def __call__(self, x, y):
        return torch.squeeze(self.lpips(x,y))



