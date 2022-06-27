import numpy as np
import matplotlib.pyplot as plt

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
