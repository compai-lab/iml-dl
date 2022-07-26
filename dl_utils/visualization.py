import matplotlib.pyplot as plt
import numpy as np


def visualize_slice(img, max=1):
    """
    Normalize for visualization
    """
    # img = img[0][0]
    img[0][0] = 0
    img[0][1] = max
    return img


def slice_3d_volume(volume, slice_id, axis):
    if axis == 0:
        return volume[slice_id, :, :]
    elif axis == 1:
        return volume[:,  slice_id, :]
    else:
        return volume[:, :, slice_id]


def vis_3d_reconstruction(img, rec, slice_id=0, prior=None, gt=None):
    axis = ['Coronal', 'Sagittal', 'Axial']
    figs = []
    for i_ax, ax in enumerate(axis):
        img_slice = slice_3d_volume(img, slice_id, i_ax).T
        max_value = np.max(img_slice)
        prior_slice = slice_3d_volume(prior, slice_id, i_ax).T if prior is not None else None
        rec_slice = slice_3d_volume(rec, slice_id, i_ax).T
        elements = [img_slice, prior_slice, rec_slice, np.abs(prior_slice - img_slice), np.abs(rec_slice-img_slice)] \
            if prior is not None else [img_slice, rec_slice, np.abs(rec_slice-img_slice)]

        if gt is not None:
            elements.append(slice_3d_volume(gt, slice_id, i_ax).T)

        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)

        for i in range(len(axarr)):
            axarr[i].axis('off')
            c_map = 'gray' if i < np.ceil(len(elements)/2) else 'inferno'
            max = max_value if i < np.ceil(len(elements)/2) else 0.5
            # print(f'max:  {max}')
            axarr[i].imshow(visualize_slice(elements[i], max), cmap=c_map, origin='lower')
        figs.append(diffp)
    return figs


def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\phi$", fontsize=30, color=(0.85, 0.27, 0.41, 0.75), linewidth=1.0, flip=False):
    """disp shape (2, H, W)
            code from: https://github.com/qiuhuaqi/midir
    """
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": linewidth, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray', origin='lower') if flip else ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return ax