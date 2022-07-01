import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import SimpleITK as sitk
import torch

class PlotResults():

    def __init__(self, spatial_data_info):
        self.spatial_data_info = spatial_data_info

    def transform_all_inputs(self, inputs = list):
        inputs_detached = [i.cpu().detach() for i in inputs]
        inputs_sitk = [torchToSitk(i, self.spatial_data_info) for i in inputs_detached]
        inputs_np = [sitk.GetArrayFromImage(i) for i in inputs_sitk]

        return inputs_np


    def plot_template_difference(self, template_before, template_after, vis_factor=1):

        inputs_np = self.transform_all_inputs(inputs = [template_before, template_after])

        temp_diff = inputs_np[1] - inputs_np[0]

        fig, ax = plt.subplots(1, 3)
        # Assuming 2D - cut first two dimensions for plotting
        plot_with_colorbar(inputs_np[0], ax[0])
        plot_with_colorbar(inputs_np[1], ax[1])
        plot_with_colorbar(temp_diff * vis_factor, ax[2], vmin=-0.1, vmax=0.1)

        fig.tight_layout()

        return fig

    def plot_registration_results(self, x, y_pred, y_true):
        # ToDo: plot colorbar with given range (without changing data points)

        [x, y_pred, y_true] = self.transform_all_inputs(inputs = [x, y_pred, y_true])

        img_diff_before = y_true - x
        img_diff_after = y_true - y_pred
        improvement = img_diff_before - img_diff_after

        # img_pred[0, 0], img_pred[0, 1] = 0, 1
        # img_true[0, 0], img_true[0, 1] = 0, 1
        # img_start[0, 0], img_start[0, 1] = 0, 1

        improvement_vis_factor = 100

        grid_image = np.hstack([x, y_true, y_pred, img_diff_before, img_diff_after, improvement*improvement_vis_factor])
        # grid_image_ = stack_registration_results(x, y_pred, y_true)
        fig, ax = plt.subplots()
        img = ax.imshow(grid_image[:, :], cmap='gray', clim=[0, 1])
        plt.title('Source, Target, Warped, Diff_before, Diff_after, Improvement x {}'.format(improvement_vis_factor))
        # cb = fig.colorbar(img, ax = ax)
        # img.clim = [-1, 1]

        return fig

    def plot_deformation_fields(self,flow_tensor):
        # ToDo: plot colorbar with given range (without changing data points)

        dim = flow_tensor.shape[1]
        img_list = []

        for d in range(dim):
            img_list.append(flow_tensor[0, d, :, :])

        # Total deformation:
        img_list.append(get_deformation_magnitude(flow_tensor))

        img_list = self.transform_all_inputs(img_list)

        # Subplots
        fig, ax = plt.subplots(1, len(img_list))
        for i in range(len(img_list)):
            plot_with_colorbar(img_list[i], ax_handle=ax[i], vmin=-1, vmax=1)
        fig.tight_layout

        return fig


# Helper functions
def plot_with_colorbar(data, ax_handle, title=None, cmap='gray', vmin=0, vmax=1, **kwargs):
    im = ax_handle.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if title:
        ax_handle.title.set_text(title)

    divider = make_axes_locatable(ax_handle)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    return ax_handle

def detach_for_plot(tensor):
    return tensor.detach().cpu().numpy()


def sitkToTorch(sitk_image, transpose = True):

    np_image = sitk.GetArrayFromImage(sitk_image)  # need to permute indices for consistency
    if transpose is True:
        np_image = np.transpose(np_image)
        tensor_info = {'spacing': sitk_image.GetSpacing(), 'origin': sitk_image.GetOrigin(),
                       'size': sitk_image.GetSize()}
    else:
        tensor_info = {'spacing': sitk_image.GetSpacing()[::-1], 'origin': sitk_image.GetOrigin()[::-1],
                       'size': sitk_image.GetSize()[::-1]}

    image_tensor = torch.from_numpy(np_image)
    return image_tensor, tensor_info

# Transform tto world coordinate system
def torchToSitk(torch_tensor, tensor_info, transpose = True):
    """    Args:
        torch_tensor: torch tensor (cpu)
        tensor_info: dictionary
        transpose: if data was transposed at load time or not
        Returns:
    """
    np_image = torch_tensor.numpy()
    if transpose is True:
        np_image = np.transpose(np_image)
        # if data was transposed at load time, then spacing/origin were not
        spacing = tensor_info['spacing']
        origin = tensor_info['origin']
    else:
        spacing = tensor_info['spacing'][::-1]
        origin = tensor_info['origin'][::-1]

    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    return sitk_image

class plot_template_development():

    def __init__(self, epochs):
        self.templates = []
        self.index = 0

    def __call__(self, template, backup_idx = 10):
        self.templates.append(template)

        if self.index % backup_idx == 0:
            print('save')

        self.index += 1

# def save_list_to_figures(list, path):


def get_deformation_magnitude(flow):
        '''
        Input: tensor with x, y(, z) deformation field
        Output: numpy array of image size with magnitude of deformation per pixel
        '''
        # Total deformation:
        flow_mag = flow.pow(2).sum(dim=1).sqrt()

        return flow_mag

def stack_template_difference(template_before, template_after):

    template_before_, template_after_ = detach_for_plot(template_before), detach_for_plot(template_after)

    temp_diff = template_after_ - template_before_

    template_grid = np.hstack([template_before_, template_after_, temp_diff])

    return template_grid

def stack_registration_results(x, y_pred, y_true):
        # Source, source_warped, target

        ## Plots
    img_pred = y_pred.detach().cpu()[0].numpy()
    img_true = y_true.detach().cpu()[0].numpy()
    img_start = x.detach().cpu()[0].numpy()

    img_diff_before = img_true - img_start
    img_diff_after = img_true - img_pred

    # img_pred[0, 0], img_pred[0, 1] = 0, 1
    # img_true[0, 0], img_true[0, 1] = 0, 1
    # img_start[0, 0], img_start[0, 1] = 0, 1

    grid_image = np.dstack([img_start, img_true, img_diff_before, img_pred, img_diff_after])

    return grid_image

def stack_deformation_results(flow_tensor):

        dim = flow_tensor.shape[1]
        img_list = []

        for d in range(dim):

                img_list.append(flow_tensor[0,d,:,:].detach().cpu().numpy())

        # Total deformation:
        img_list.append(get_deformation_magnitude(flow_tensor))

        def_image = np.concatenate(img_list, axis = -1)

        return def_image

