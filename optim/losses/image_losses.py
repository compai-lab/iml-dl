import torch
import torch.nn.functional as F
from torch.nn import BCELoss
import numpy as np
import math
from model_zoo import VGGEncoder
from torch.nn.modules.loss import _Loss
from optim.losses.ln_losses import L2


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

        return -torch.mean(cc)

class BCE_loss:
    def __init__(self):
        super(BCE_loss, self).__init__()
        self.loss_ = BCELoss(reduction="sum")

    def __call__(self, x_recon, x , z=None):
        return self.loss_(x_recon, x)


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
        input_features = self.loss_network(input.repeat(1, 3, 1, 1))
        output_features = self.loss_network(target.repeat(1, 3, 1, 1))

        loss_pl = 0
        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
        return loss_pl

def compute_reg_loss(z, attr,factor):
    reg_loss = 0.0
    reg_dim_real = attr.size()[1]
    for dim in range(reg_dim_real):
        x_ = z[:, dim]
        reg_loss += reg_loss_sign(x_, attr[:, dim], factor)

    return reg_loss

def reg_loss_sign(latent_code, attribute, factor):
    """
    Computes the regularization loss given the latent code and attribute
    Args:
        latent_code: torch Variable, (N,)
        attribute: torch Variable, (N,)
        factor: parameter for scaling the loss
    Returns
        scalar, loss
    """
    # compute latent distance matrix
    latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
    lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

    # compute attribute distance matrix
    attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
    attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

    # compute regularization loss
    loss_fn = torch.nn.L1Loss()
    lc_tanh = torch.tanh(lc_dist_mat * factor).cpu()
    attribute_sign = torch.sign(attribute_dist_mat)
    sign_loss = loss_fn(lc_tanh, attribute_sign.float())

    return sign_loss


class AR_VAEPatiLoss:

    def __init__(self, beta, gamma, factor, reg_dim):
        super(AR_VAEPatiLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.factor = factor
        self.reg_dim = reg_dim

    def __call__(self, x_recon, x, z, attr, all= False):

        kld_weight = 1 #0.0128  # Account for the minibatch samples from the dataset
        batch_size = x.size(0)
        # Rec Loss
        recons_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
        # pl_loss = PerceptualLoss()
        # l2_loss = L2()
        # recons_loss = pl_loss(x_recon,x) + self.alpha * l2_loss(x_recon,x)

        # KLD Loss
        kld_loss = torch.distributions.kl.kl_divergence(z['z_dist'], z['prior_dist'])
        kld_loss = kld_loss.sum(1).mean()

        c = 0.0
        beta_loss = self.beta * kld_weight * (kld_loss - c).abs()

        # Reg loss
        reg_loss = 0.0
        reg_dim_real = attr.size()[1]
        for dim in range(reg_dim_real):
            x_ = z['z_tilde'][:, dim]
            reg_loss += self.reg_loss_sign(x_, attr[:,dim], self.factor)

        global_loss = recons_loss + beta_loss + self.gamma * reg_loss #

        #print(beta_loss, reg_loss ,pl_loss(x_recon,x), l2_loss(reconstructed_images, transformed_images))

        if all:
            return global_loss, recons_loss, beta_loss, reg_loss
        else:
            return global_loss

    @staticmethod
    def reg_loss_sign(latent_code, attribute, factor):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor).cpu()
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss
