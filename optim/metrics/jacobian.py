import numpy as np
import scipy.ndimage
import nibabel as nib
import matplotlib as plt
def jacobian_determinant(disp,rec):
    #code from https://github.com/MDL-UzL/L2R/blob/main/evaluation/utils.py
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian2=jacobian
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    jacdet2= jacobian2[0, 0, :, :, :] * (jacobian2[1, 1, :, :, :] * jacobian2[2, 2, :, :, :] - jacobian2[1, 2, :, :, :] * jacobian2[2, 1, :, :, :]) -\
             jacobian2[1, 0, :, :, :] * (jacobian2[0, 1, :, :, :] * jacobian2[2, 2, :, :, :] - jacobian2[0, 2, :, :, :] * jacobian2[2, 1, :, :, :]) +\
             jacobian2[2, 0, :, :, :] * (jacobian2[0, 1, :, :, :] * jacobian2[1, 2, :, :, :] - jacobian2[0, 2, :, :, :] * jacobian2[1, 1, :, :, :])
    min_jac_det = jacdet.min()
    max_jac_det = jacdet.max()

    perc_neg_jac_det = (jacdet <= 0).astype(float).sum()*100/(np.prod(jacdet.shape)-(jacdet[rec[:, :, 2:-2, 2:-2, 2:-2].squeeze()<0.005]).sum())   
    
    return min_jac_det,max_jac_det,perc_neg_jac_det,jacdet2
class MidpointNormalize(plt.colors.Normalize):
    """
    class to help renormalize the color scale
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        plt.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
