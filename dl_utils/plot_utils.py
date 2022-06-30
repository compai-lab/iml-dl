import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def detach_for_plot(tensor):

    return tensor.detach().cpu().numpy()

# def plot_aligned()

def plot_with_colorbar(data, ax_handle, title = None, cmap='gray', vmin=0, vmax=1, **kwargs):

    im = ax_handle.imshow(data, cmap = cmap, vmin  = vmin, vmax = vmax, **kwargs)

    if title:
        ax_handle.title.set_text(title)

    divider = make_axes_locatable(ax_handle)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    return ax_handle

def plot_template_difference(template_before, template_after, vis_factor = 1):

    template_before_, template_after_ = detach_for_plot(template_before), detach_for_plot(template_after)
    temp_diff = template_after_ - template_before_

    fig, ax = plt.subplots(1,3)

    # Assuming 2D - cut first two dimensions for plotting
    plot_with_colorbar(template_before_[0,0,:,:], ax[0])
    plot_with_colorbar(template_after_[0,0,:,:], ax[1])
    plot_with_colorbar(temp_diff[0,0,:,:]*vis_factor, ax[2], vmin=-0.1, vmax=0.1)

    fig.tight_layout()

    return fig

def stack_template_difference(template_before, template_after):

    template_before_, template_after_ = detach_for_plot(template_before), detach_for_plot(template_after)

    temp_diff = template_after_ - template_before_

    template_grid = np.hstack([template_before_, template_after_, temp_diff])

    return template_grid

def get_deformation_magnitude(flow):
        '''
        Input: tensor with x, y(, z) deformation field
        Output: numpy array of image size with magnitude of deformation per pixel
        '''

        # Total deformation:
        flow_mag = flow.pow(2).sum(dim=1).sqrt().detach().cpu()[0].numpy()

        return flow_mag


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

def plot_registration_results(x, y_pred, y_true):

        #ToDo: plot colorbar with given range (without changing data points)

        grid_image_ = stack_registration_results(x, y_pred, y_true)
        fig, ax = plt.subplots()
        img = ax.imshow(grid_image_[0,:,:], cmap = 'gray', clim = [0, 1])
        #cb = fig.colorbar(img, ax = ax)
        
        # img.clim = [-1, 1]

        return fig


def plot_deformation_fields(flow_tensor):

        #ToDo: plot colorbar with given range (without changing data points)

        def_image = stack_deformation_results(flow_tensor)
        fig, ax = plt.subplots()
        img = ax.imshow(def_image)
        cb = fig.colorbar(img, ax = ax)
        
        # img.clim = [-1, 1]

        return fig
