import logging
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
#
from optim.losses import MedicalNetPerceptualSimilarity

from torch.nn import L1Loss, MSELoss
#
from dl_utils.visualization import plot_warped_grid,save_warped_grid
from optim.losses import PerceptualLoss

from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from sklearn.metrics import roc_auc_score, roc_curve
from skimage import exposure
from skimage.measure import label, regionprops
from scipy.ndimage.filters import gaussian_filter
import nibabel as nib
from PIL import Image
#
from optim.metrics.jacobian import jacobian_determinant,MidpointNormalize

import lpips
#
from dl_utils import *
from optim.metrics import *
from optim.losses.image_losses import NCC
from core.DownstreamEvaluator import DownstreamEvaluator
import subprocess
import os
import copy
from model_zoo import VGGEncoder
from optim.losses.image_losses import CosineSimLoss
from transforms.synthetic import *


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_= True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = MSELoss().to(self.device)
        self.criterion_MSE = MSELoss().to(self.device)
       # self.auprc = AUPRC()
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)
        self.l_cos = CosineSimLoss(device='cuda')
        self.l_ncc = NCC(win=[9, 9])
        self.experiment="no_deformer_g_0.01_upsample_dc_1e-4_lr_5e-4"
        # self.l_pips_vgg = lpips.LPIPS(pretrained=True, net='vgg', use_dropout=False, eval_mode=False, spatial=False, lpips=True).to(self.device)
        # self.l_pips_alex = lpips.LPIPS(pretrained=True, net='alex', use_dropout=False, eval_mode=False, spatial=False, lpips=True).to(self.device)

        self.global_= True

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        # AE-S: 0.169 | 0.146 | 0.123
        # VAE: 0.464 | 0.402 | 0.294
        # DAE: 0.138 | 0.108 | 0.083
        # SI-VAE: 0.644 | 0.51 | 0.319
        # RA: 0.062| 0.049 | 0.032
        # RA: 0.273 | 0.212 | 0.136
        # RA: 0 | 0.9 | 0.822
        # RA: 0.015 | 0.011 | 0.007
        # th = 0.033
        # _ = self.thresholding(global_model)
        # if self.global_:
        #     self.global_detection(global_model)
        # else:
        #     self.object_localization(global_model, th)
        # self.pathology_localization(global_model)
        # self.curate_dataset(global_model)
        #
        # self.umap_plot(global_model)
        self.global_detection2(global_model)


    def global_detection(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        task='Test_Alzheimer'
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        loss_mse_array=[]
        loss_mse_array=np.array(loss_mse_array)
        loss_rec_array=[]
        loss_rec_array=np.array(loss_rec_array)
        
        stdlogjacdet_array=[]
        stdlogjacdet_array=np.array(stdlogjacdet_array)
        meanjacdet_array=[]
        meanjacdet_array=np.array(meanjacdet_array)
        kk=0
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            test_total=0
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                if len(x.shape)==4:
                    b, c, h, w = x.shape
                else:
                    b, c, h, w,d = x.shape

                test_total += b
                x_ = x_rec_dict['x_prior']
                deformation = x_rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_rec, x)
                self.criterion_PL = MedicalNetPerceptualSimilarity(device=device)
                loss_pl = self.criterion_PL(x_, x)
                loss_mse_array= np.append(loss_mse_array,loss_mse.detach().cpu().numpy())
                loss_rec_array= np.append(loss_rec_array,loss_rec.detach().cpu().numpy())
                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)
              #  _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
               # jacdet_def = (jacdet + 3).clip(1e-10,1e10)
               # log_jac_det = np.log(jacdet_def)
               # single_value = log_jac_det.std()
               # mean= jacdet_def.mean()
               # stdlogjacdet_array= np.append(stdlogjacdet_array,single_value)
               # meanjacdet_array= np.append(meanjacdet_array,mean)

                w=116
                if idx%260==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_rec.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze(x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_gl_prior.nii.gz')
                   # x_nifti = nib.Nifti1Image(np.squeeze(x.detach().cpu()[0].numpy()) , np.eye(4))
                   # nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_img.nii.gz')
                  #  x_nifti = nib.Nifti1Image(np.squeeze(deformation.detach().cpu()[0].numpy()) , np.eye(4))
                    #nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_deff.nii.gz')
                   # x_nifti = nib.Nifti1Image(np.squeeze(jacdet) , np.eye(4))
                   # nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_jacdet.nii.gz')

                if idx<=20:
                    if len(x.shape)==4:
                        global_prior = x_.detach().cpu()[0].numpy()
                        # global_prior[0, 0], global_prior[0, 1] = 0, 1
                        rec = x_rec.detach().cpu()[0].numpy()
                        # rec[0, 0], rec[0, 1] = 0, 1
                        img = x.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        deff = deformation.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        elements = [img, global_prior, rec, np.abs(global_prior-img), np.abs(rec-img),deff]
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for i in range(len(axarr)):
                            axarr[i].axis('off')
                            if i!=len(axarr)-1:
                                v_max = 1 if i < np.floor(((len(elements)-1) / 2)) + 1 else 0.5
                                c_map = 'gray' if i < np.floor(((len(elements)-1) / 2)) + 1 else 'inferno'               
                                axarr[i].imshow(np.squeeze(elements[i].transpose(1, 2, 0)), vmin=0, vmax=v_max, cmap=c_map)
                            else:
                                plot_warped_grid(ax=axarr[i],disp=deff)
                    else:
                        global_prior = x_.detach().cpu()[0].numpy()
                        # global_prior[0, 0], global_prior[0, 1] = 0, 1
                        rec = x_rec.detach().cpu()[0].numpy()
                        # rec[0, 0], rec[0, 1] = 0, 1
                        img = x.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        deff = deformation[0,:,:,:,:].detach().cpu().numpy()
                        # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            
            
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),deff]
                    v_maxs = [1, 1, 1,0.5,0.5,0.5,0.5]
                    diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 3 * 4)
                    for i in range(len(elements)):
                        for axis in range(3):
                            if i<=len(elements)-2:
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

                    wandb.log({task + '/Example_': [
                            wandb.Image(diffp, caption="Test_Alzheimer_" + str(idx))]})
                    

        fig, ax = plt.subplots()
        ax.hist(loss_mse_array, bins=len(loss_mse_array)) 
#        wandb.log({"Test_Alzheimer_histogram": fig})
        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': idx})
        np.save("./results/"+self.experiment+"/alzheimer_loss_morphed.npy", loss_mse_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/alzheimer_loss_prior.npy", loss_rec_array, allow_pickle=True, fix_imports=True)

    def global_detection2(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        task='Test_Healthy'
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        loss_mse_array=[]
        loss_mse_array=np.array(loss_mse_array)
        loss_rec_array=[]
        loss_rec_array=np.array(loss_rec_array)
        stdlogjacdet_array=[]
        stdlogjacdet_array=np.array(stdlogjacdet_array)
        meanjacdet_array=[]
        meanjacdet_array=np.array(meanjacdet_array)
        kk=0
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            test_total=0
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                if len(x.shape)==4:
                    b, c, h, w = x.shape
                else:
                    b, c, h, w,d = x.shape

                test_total += b
                x_ = x_rec_dict['x_prior']
                deformation = x_rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_rec, x)
                self.criterion_PL = MedicalNetPerceptualSimilarity(device=device)
                loss_pl = self.criterion_PL(x_, x)
                if loss_mse.detach().cpu().numpy()<0:
                    print("loss_mse is smalelr than 0")
                    loss_mse=0
                loss_mse_array= np.append(loss_mse_array,loss_mse.detach().cpu().numpy())
                loss_rec_array= np.append(loss_rec_array,loss_rec.detach().cpu().numpy())
                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)


                if idx%260==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+self.experiment+'/test_healthy_'+str(idx)+'_rec.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze( x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+self.experiment+'/test_healthy_'+str(idx)+'_gl_prior.nii.gz')


                if idx<=20:
                    if len(x.shape)==4:
                        global_prior = x_.detach().cpu()[0].numpy()
                        # global_prior[0, 0], global_prior[0, 1] = 0, 1
                        rec = x_rec.detach().cpu()[0].numpy()
                        # rec[0, 0], rec[0, 1] = 0, 1
                        img = x.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        deff = deformation.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        elements = [img, global_prior, rec, np.abs(global_prior-img), np.abs(rec-img),deff]
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for i in range(len(axarr)):
                            axarr[i].axis('off')
                            if i!=len(axarr)-1:
                                v_max = 1 if i < np.floor(((len(elements)-1) / 2)) + 1 else 0.5
                                c_map = 'gray' if i < np.floor(((len(elements)-1) / 2)) + 1 else 'inferno'               
                                axarr[i].imshow(np.squeeze(elements[i].transpose(1, 2, 0)), vmin=0, vmax=v_max, cmap=c_map)
                            else:
                                plot_warped_grid(ax=axarr[i],disp=deff)
                    else:
                        global_prior = x_.detach().cpu()[0].numpy()
                        # global_prior[0, 0], global_prior[0, 1] = 0, 1
                        rec = x_rec.detach().cpu()[0].numpy()
                        # rec[0, 0], rec[0, 1] = 0, 1
                        img = x.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        deff = deformation[0,:,:,:,:].detach().cpu().numpy()
                        # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            
            
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),deff]
                    v_maxs = [1, 1, 1,0.5,0.5,0.5,0.5]
                    diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 3 * 4)
                    for i in range(len(elements)):
                        for axis in range(3):
                            if i<=len(elements)-2:
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

                    wandb.log({task + '/Example_': [
                            wandb.Image(diffp, caption="Test_" + str(idx))]})
                    

        fig, ax = plt.subplots()
        ax.hist(loss_mse_array, bins=len(loss_mse_array)) 
#        wandb.log({"Test_healthy_histogram": fig})
        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': idx})
        np.save("./results/"+self.experiment+"/healthy_loss_morphed.npy", loss_mse_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/healthy_loss_prior.npy", loss_rec_array, allow_pickle=True, fix_imports=True)


    def global_detection3(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        task='Test_MCI'
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        loss_mse_array=[]
        loss_mse_array=np.array(loss_mse_array)
        loss_rec_array=[]
        loss_rec_array=np.array(loss_rec_array)
        
        stdlogjacdet_array=[]
        stdlogjacdet_array=np.array(stdlogjacdet_array)
        meanjacdet_array=[]
        meanjacdet_array=np.array(meanjacdet_array)
        kk=0
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            test_total=0
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                if len(x.shape)==4:
                    b, c, h, w = x.shape
                else:
                    b, c, h, w,d = x.shape

                test_total += b
                x_ = x_rec_dict['x_prior']
                deformation = x_rec_dict['deformation']
                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_rec, x)
                self.criterion_PL = MedicalNetPerceptualSimilarity(device=device)
                loss_pl = self.criterion_PL(x_, x)
                loss_mse_array= np.append(loss_mse_array,loss_mse.detach().cpu().numpy())
                loss_rec_array= np.append(loss_rec_array,loss_rec.detach().cpu().numpy())
                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)


                if idx%260==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+self.experiment+'/test_mci_'+str(idx)+'_rec.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze(x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+self.experiment+'/test_mci_'+str(idx)+'_gl_prior.nii.gz')


                if idx<=20:
                    if len(x.shape)==4:
                        global_prior = x_.detach().cpu()[0].numpy()
                        # global_prior[0, 0], global_prior[0, 1] = 0, 1
                        rec = x_rec.detach().cpu()[0].numpy()
                        # rec[0, 0], rec[0, 1] = 0, 1
                        img = x.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        deff = deformation.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        elements = [img, global_prior, rec, np.abs(global_prior-img), np.abs(rec-img),deff]
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for i in range(len(axarr)):
                            axarr[i].axis('off')
                            if i!=len(axarr)-1:
                                v_max = 1 if i < np.floor(((len(elements)-1) / 2)) + 1 else 0.5
                                c_map = 'gray' if i < np.floor(((len(elements)-1) / 2)) + 1 else 'inferno'               
                                axarr[i].imshow(np.squeeze(elements[i].transpose(1, 2, 0)), vmin=0, vmax=v_max, cmap=c_map)
                            else:
                                plot_warped_grid(ax=axarr[i],disp=deff)
                    else:
                        global_prior = x_.detach().cpu()[0].numpy()
                        # global_prior[0, 0], global_prior[0, 1] = 0, 1
                        rec = x_rec.detach().cpu()[0].numpy()
                        # rec[0, 0], rec[0, 1] = 0, 1
                        img = x.detach().cpu()[0].numpy()
                        # img[0, 0], img[0, 1] = 0, 1
                        # grid_image = np.hstack([img, global_prior, rec])
                        deff = deformation[0,:,:,:,:].detach().cpu().numpy()
                        # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            
                    
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),deff]
                    v_maxs = [1, 1, 1,0.5,0.5,0.5,0.5]
                    diffp, axarr = plt.subplots(3, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                    diffp.set_size_inches(len(elements) * 4, 3 * 4)
                    for i in range(len(elements)):
                        for axis in range(3):
                            if i<=len(elements)-2:
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

                    wandb.log({task + '/Example_': [
                            wandb.Image(diffp, caption="Test_MCI_" + str(idx))]})
                    

        fig, ax = plt.subplots()
        ax.hist(loss_mse_array, bins=len(loss_mse_array)) 
#        wandb.log({"Test_MCI_histogram": fig})
        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': idx})
        np.save("./results/"+self.experiment+"/mci_loss_morphed.npy", loss_mse_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/mci_loss_prior.npy", loss_rec_array, allow_pickle=True, fix_imports=True)


        plot_roc_curve_no_deformer(self.experiment,True)

