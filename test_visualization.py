from dl_utils.visualization import plot_warped_grid
import numpy as np
from matplotlib import pyplot as plt



img = np.load("img.npy")
# print(np.min(img), np.max(img))
rec_ = np.load("rec.npy")
gl_prior = np.load("gl_prior.npy")
# deff=rec_
deff = np.load("deff.npy")
# print(f'rec: {np.min(rec)}, {np.max(rec)}')
elements = [img,gl_prior, rec_,np.abs(gl_prior-img), np.abs(rec_ - img),deff]
v_maxs = [1, 1, 1,0.5,0.5,0.5]
diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
diffp.set_size_inches(len(elements) * 4, 3 * 4)
w=160
for i in range(len(elements)):
    for axis in range(3):
        if i!=len(elements)-1:
            axarr[axis, i].axis('off')
            v_max = v_maxs[i]
            c_map = 'gray' if v_max == 1 else 'plasma'
            # print(elements[i].shape)
            if axis == 0:
                el = np.squeeze(elements[i])[int(w / 2), :, :]
            elif axis == 1:
                el = np.squeeze(elements[i])[:, int(w / 2), :]
            else:
                el = np.squeeze(elements[i])[:, :, int(w / 2)]

            axarr[axis, i].imshow(np.squeeze(el).T, vmin=0, vmax=v_max, cmap=c_map, origin='lower')
        else:
            
                if axis == 0:
                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                    plot_warped_grid(ax=axarr[axis, i],disp=temp) # .rot90(axes=(2,3)
                elif axis == 1:
                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                else:
                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
plt.savefig('trial_vis22.png')