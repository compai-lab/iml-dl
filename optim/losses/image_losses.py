import torch
import torch.nn.functional as F
import numpy as np
import math
from model_zoo import VGGEncoder
from torch.nn.modules.loss import _Loss
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn as nn
from lpips import LPIPS
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

        num_channels = y_true.shape[1]
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else [self.win]* ndims

        # compute filters
        # print(f'Num channels: {num_channels}')
        sum_filt = torch.ones([1, num_channels, *win]).to("cuda")

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

        var_prod = torch.clamp(I_var * J_var, min=0)  # For stability, avoid infinity
        cc = cross * cross / (var_prod + 1e-5)

        return 1-torch.mean(cc)

        #return 1-cc

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

class DisplacementRegularizer(torch.nn.Module):
    """
    code from https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/

    License:
        MIT License
    """
    def __init__(self, energy_type='gradient-l2'):
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
        
class MedicalNetPerceptualSimilarity(_Loss):
    """
    Component to perform the perceptual evaluation with the networks pretrained by Chen, et al. "Med3D: Transfer
    Learning for 3D Medical Image Analysis". This class uses torch Hub to download the networks from
    "Warvito/MedicalNet-models".
    Args:
        net: {``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``}
            Specifies the network architecture to use. Defaults to ``"medicalnet_resnet10_23datasets"``.
        verbose: if false, mute messages from torch Hub load function.
    """

    def __init__(self, net: str = "medicalnet_resnet10_23datasets", verbose: bool = False, device: str = 'cuda') -> None:
        super().__init__()
        self.device=device
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load("Warvito/MedicalNet-models", model=net, verbose=verbose)
        self.eval().to(self.device)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using MedicalNet 3D networks. The input and target tensors are inputted in the
        pre-trained MedicalNet that is used for feature extraction. Then, these extracted features are normalised across
        the channels. Finally, we compute the difference between the input and target features and calculate the mean
        value from the spatial dimensions to obtain the perceptual loss.
        Args:
            input: 3D input tensor with shape BCDHW.
            target: 3D target tensor with shape BCDHW.
        """
        input = medicalnet_intensity_normalisation(input)
        target = medicalnet_intensity_normalisation(target)

        # Get model outputs
        outs_input = self.model.forward(input)
        outs_target = self.model.forward(target)

        # Normalise through the channels
        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        results = (feats_input - feats_target) ** 2
        results = spatial_average_3d(results.sum(dim=1, keepdim=True), keepdim=True)

        return results
    
def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def medicalnet_intensity_normalisation(volume):
    """Based on https://github.com/Tencent/MedicalNet/blob/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b/datasets/brains18.py#L133"""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std

class RadImageNetPerceptualSimilarity(nn.Module):
    """
    Component to perform the perceptual evaluation with the networks pretrained on RadImagenet (pretrained by Mei, et
    al. "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"). This class
    uses torch Hub to download the networks from "Warvito/radimagenet-models".

    Args:
        net: {``"radimagenet_resnet50"``}
            Specifies the network architecture to use. Defaults to ``"radimagenet_resnet50"``.
        verbose: if false, mute messages from torch Hub load function.
    """

    def __init__(self, net: str = "radimagenet_resnet50", verbose: bool = False,device='cuda') -> None:
        super().__init__()
        self.model = torch.hub.load("Warvito/radimagenet-models", model=net, verbose=verbose)
        self.eval().to(device)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        We expect that the input is normalised between [0, 1]. Given the preprocessing performed during the training at
        https://github.com/BMEII-AI/RadImageNet, we make sure that the input and target have 3 channels, reorder it from
         'RGB' to 'BGR', and then remove the mean components of each input data channel. The outputs are normalised
        across the channels, and we obtain the mean from the spatial dimensions (similar approach to the lpips package).
        """
        # If input has just 1 channel, repeat channel to have 3 channels
        results=0
        for k in range(0,input.shape[2]-2,3):
           # print(k,input.shape[2])
            input_2d=input[:,:,k:k+3,:,:].squeeze()
            target_2d=target[:,:,k:k+3,:,:].squeeze()
            input_2d=input_2d[np.newaxis,:,:,:]
            target_2d=target_2d[np.newaxis,:,:,:]
            # Subtract mean used during training
         #  input_2d = subtract_mean(input_2d)
          #  target_2d = subtract_mean(target_2d)

            # Get model outputs
            outs_input = self.model.forward(input_2d)
            outs_target = self.model.forward(target_2d)

            # Normalise through the channels
            feats_input = normalize_tensor(outs_input)
            feats_target = normalize_tensor(outs_target)

            results_curr = (feats_input - feats_target) ** 2
            results =results+ spatial_average(results_curr.sum(dim=1, keepdim=True), keepdim=True)

        return results

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def torchvision_zscore_norm(x: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
    x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
    x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
    return x


def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[ 0, :, :] -= mean[0]
    x[ 1, :, :] -= mean[1]
    x[ 2, :, :] -= mean[2]
    return x
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
        if len(input.shape)==4:

            input_features = self.loss_network(input.repeat(1, 3, 1, 1)) if input.shape[1] == 1 else input
            output_features = self.loss_network(target.repeat(1, 3, 1, 1)) if target.shape[1] == 1 else target

            loss_pl = 0
            for output_feature, input_feature in zip(output_features, input_features):
                loss_pl += F.mse_loss(output_feature, input_feature)
            return loss_pl
        else:
            loss_pl = 0
            for k in range(0,input.shape[2],16):
                input_2d=input[:,:,k:k+16,:,:].squeeze()
                target_2d=target[:,:,k:k+16,:,:].squeeze()
                input_2d=input_2d[:,np.newaxis,:,:]   
                target_2d=target_2d[:,np.newaxis,:,:]         
                input_features = self.loss_network(input_2d.repeat(1, 3, 1, 1)) if input.shape[1] == 1 else input
                output_features = self.loss_network(target_2d.repeat(1, 3, 1, 1)) if target.shape[1] == 1 else target


                for output_feature, input_feature in zip(output_features, input_features):
                    for l in range(0,16):
                        loss_pl += F.mse_loss(output_feature[l,:,:,:], input_feature[l,:,:,:]) 
            return loss_pl/input.shape[2]

class CosineSimLoss():
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

    def norm(self, input):
        input = input/np.max(input)
        return input

    def __call__(self, input: torch.Tensor, target: torch.Tensor, out_size=128, amap_mode='mul'):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        input_features = self.loss_network(input.repeat(1, 3, 1, 1)) if input.shape[1] == 1 else input
        output_features = self.loss_network(target.repeat(1, 3, 1, 1)) if target.shape[1] == 1 else target
        anomaly_maps = []
        for b in range(input.shape[0]):
            if amap_mode == 'mul':
                anomaly_map = np.ones([out_size, out_size])
            else:
                anomaly_map = np.zeros([out_size, out_size])
            a_map_list = []
            for i in range(len(output_features)):
                fs = input_features[i]
                ft = output_features[i]
                # fs_norm = F.normalize(fs, p=2)
                # ft_norm = F.normalize(ft, p=2)
                a_map = 1 - F.cosine_similarity(fs, ft)
                a_map = torch.unsqueeze(a_map, dim=1)
                a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
                a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
                a_map_list.append(a_map)
                if amap_mode == 'mul' or amap_mode == 'mulsum':
                    anomaly_map *= a_map
                else:
                    anomaly_map += a_map
            anomaly_map = anomaly_map * 10 + gaussian_filter((np.max(np.asarray(a_map_list), axis=0)/10), 2)
            # if amap_mode == 'mulsum':
            #     anomaly_map = (self.norm(anomaly_map) + self.norm(np.max(np.asarray(a_map_list), axis=0)) / 2) / 2
            # else:
            #     anomaly_map = self.norm(anomaly_map)

            anomaly_maps.append(np.expand_dims(anomaly_map,  0))
        anomaly_maps = np.asarray(anomaly_maps)
        return anomaly_maps


class EmbeddingLoss(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, teacher_embeddings, student_embeddings):
        # print(f'LEN {len(output_real)}')
        layer_id = 0
        # teacher_embeddings = teacher_embeddings[:-1]
        # student_embeddings = student_embeddings[3:-1]
        # print(f' Teacher: {len(teacher_embeddings)}, Student: {len(student_embeddings)}')
        for teacher_feature, student_feature in zip(teacher_embeddings, student_embeddings):
            if layer_id == 0:
                total_loss = 0.5 * self.criterion(teacher_feature, student_feature)
            else:
                total_loss += 0.5 * self.criterion(teacher_feature, student_feature)
            total_loss += torch.mean(1 - self.similarity_loss(teacher_feature.view(teacher_feature.shape[0], -1),
                                                         student_feature.view(student_feature.shape[0], -1)))
            layer_id += 1
        return total_loss