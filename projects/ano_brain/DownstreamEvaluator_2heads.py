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
        self.experiment="b_10_b_1_b01"
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
        loss_rec_array_b1=[]
        loss_rec_array_b1=np.array(loss_rec_array_b1)
        loss_rec_array_b01=[]
        loss_rec_array_b01=np.array(loss_rec_array_b01)                
        stdlogjacdet_array=[]
        stdlogjacdet_array=np.array(stdlogjacdet_array)
        stdlogjacdet_array_b1=[]
        stdlogjacdet_array_b1=np.array(stdlogjacdet_array_b1)        
        stdlogjacdet_array_b01=[]
        stdlogjacdet_array_b01=np.array(stdlogjacdet_array_b01)        
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
                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean= log_jac_det.mean()
                stdlogjacdet_array= np.append(stdlogjacdet_array,single_value)
                meanjacdet_array= np.append(meanjacdet_array,mean)

                    # heads #

                imagess = data[0].to(self.device)
                transformed_imagess = imagess
                encode_historyy = [x.detach().clone() for x in x_rec_dict['encode_history']]
                    
                decode_historyy = [x.detach().clone() for x in x_rec_dict['decode_history']]
                gl_priorr = x_rec_dict['x_prior'].detach().clone()
                reconstruction_b1,result_dict_b1= self.model.forward_b1(transformed_imagess,gl_priorr,encode_historyy,decode_historyy,False)
                deformation_b1 = result_dict_b1['deformation']
                reversed_b1 = result_dict_b1['x_reversed']

                reconstruction_b01,result_dict_b01= self.model.forward_b01(transformed_imagess,gl_priorr,encode_historyy,decode_historyy,False)
                deformation_b01 = result_dict_b01['deformation']
                reversed_b01 = result_dict_b01['x_reversed']
                loss_mse_b1 = self.criterion_MSE(reconstruction_b1, x)
                loss_rec_array_b1= np.append(loss_rec_array_b1,loss_mse_b1.detach().cpu().numpy())
                loss_mse_b01 = self.criterion_MSE(reconstruction_b01, x)
                loss_rec_array_b01= np.append(loss_rec_array_b01,loss_mse_b01.detach().cpu().numpy())    

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation_b1.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean= log_jac_det.mean()
                stdlogjacdet_array_b1= np.append(stdlogjacdet_array_b1,single_value)

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation_b01.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean= log_jac_det.mean()
                stdlogjacdet_array_b01= np.append(stdlogjacdet_array_b01,single_value)            
                w=116
                if idx%260==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_rec.nii.gz')
                    rec2_nifti = nib.Nifti1Image(np.squeeze(reconstruction_b1.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec2_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_rec_b1.nii.gz')
                    rec3_nifti = nib.Nifti1Image(np.squeeze(reconstruction_b01.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec3_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_rec_b01.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze(x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_gl_prior.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(x.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_img.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(deformation.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_deff.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(jacdet) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_a_'+str(idx)+'_jacdet.nii.gz')

                if idx<=20 or idx%260==0:
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
            
                    
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),jacdet_def,deff]
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
                                
                            elif i==len(elements)-1:
                                
                                if axis == 0:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp) # .rot90(axes=(2,3)
                                    if idx%260==0:
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx)
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=2)
                                elif axis == 1:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                                    if idx%260==0:
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx)
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=2)
                                else:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp)
                                    if idx%260==0:
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=1)
                                        save_warped_grid(disp=temp,save= self.experiment,idx=idx,interval=2)
                            elif i==len(elements)-2:    
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'bwr'
                                # print(elements[i].shape)
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)] 
                                axarr[axis, i].imshow(np.squeeze(el).T, cmap=c_map, norm = MidpointNormalize(midpoint=0))
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
        np.save("./results/"+self.experiment+"/alzheimer_stdlogjacdet_array.npy", stdlogjacdet_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/alzheimer_meanjacdet_array.npy", meanjacdet_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/alzheimer_loss_morphed_b1.npy", loss_rec_array_b1, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/alzheimer_loss_morphed_b01.npy", loss_rec_array_b01, allow_pickle=True, fix_imports=True)   
        np.save("./results/"+self.experiment+"/alzheimer_stdlogjacdet_array_b1.npy", stdlogjacdet_array_b1, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/alzheimer_stdlogjacdet_array_b01.npy", stdlogjacdet_array_b01, allow_pickle=True, fix_imports=True)          
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
        loss_rec_array_b1=[]
        loss_rec_array_b1=np.array(loss_rec_array_b1)
        loss_rec_array_b01=[]
        loss_rec_array_b01=np.array(loss_rec_array_b01)                
        stdlogjacdet_array=[]
        stdlogjacdet_array=np.array(stdlogjacdet_array)
        stdlogjacdet_array_b1=[]
        stdlogjacdet_array_b1=np.array(stdlogjacdet_array_b1)        
        stdlogjacdet_array_b01=[]
        stdlogjacdet_array_b01=np.array(stdlogjacdet_array_b01)        
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
                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean=log_jac_det.mean()
                meanjacdet_array= np.append(meanjacdet_array,mean)
                stdlogjacdet_array= np.append(stdlogjacdet_array,single_value)
                    # heads #

                imagess = data[0].to(self.device)
                transformed_imagess =  imagess
                encode_historyy = [x.detach().clone() for x in x_rec_dict['encode_history']]
                    
                decode_historyy = [x.detach().clone() for x in x_rec_dict['decode_history']]
                gl_priorr = x_rec_dict['x_prior'].detach().clone()
                reconstruction_b1,result_dict_b1= self.model.forward_b1(transformed_imagess,gl_priorr,encode_historyy,decode_historyy,False)
                deformation_b1 = result_dict_b1['deformation']
                reversed_b1 = result_dict_b1['x_reversed']

                reconstruction_b01,result_dict_b01= self.model.forward_b01(transformed_imagess,gl_priorr,encode_historyy,decode_historyy,False)
                deformation_b01 = result_dict_b01['deformation']
                reversed_b01 = result_dict_b01['x_reversed']
                loss_mse_b1 = self.criterion_MSE(reconstruction_b1, x)
                loss_rec_array_b1= np.append(loss_rec_array_b1,loss_mse_b1.detach().cpu().numpy())
                loss_mse_b01 = self.criterion_MSE(reconstruction_b01, x)
                loss_rec_array_b01= np.append(loss_rec_array_b01,loss_mse_b1.detach().cpu().numpy())    

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation_b1.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean= log_jac_det.mean()
                stdlogjacdet_array_b1= np.append(stdlogjacdet_array_b1,single_value)

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation_b01.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean= log_jac_det.mean()
                stdlogjacdet_array_b01= np.append(stdlogjacdet_array_b01,single_value)    

                if idx%260==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+self.experiment+'/test_healthy_'+str(idx)+'_rec.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze( x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+self.experiment+'/test_healthy_'+str(idx)+'_gl_prior.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(x.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_healthy_'+str(idx)+'_img.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(deformation.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_healthy_'+str(idx)+'_deff.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(jacdet) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_healthy_'+str(idx)+'_jacdet.nii.gz')

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
            
            
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),jacdet_def,deff]
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
                                
                            elif i==len(elements)-1:
                                
                                if axis == 0:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_healthy_'+str(idx)+'deff3.png') # .rot90(axes=(2,3)
                                elif axis == 1:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_healthy_'+str(idx)+'deff2.png')
                                else:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_healthy_'+str(idx)+'deff1.png')
                            elif i==len(elements)-2:    
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'bwr'
                                # print(elements[i].shape)
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)] 
                                axarr[axis, i].imshow(np.squeeze(el).T, cmap=c_map, norm = MidpointNormalize(midpoint=0))
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
        np.save("./results/"+self.experiment+"/healthy_stdlogjacdet.npy", stdlogjacdet_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/healthy_meanjacdet.npy", meanjacdet_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/healthy_loss_morphed_b1.npy", loss_rec_array_b1, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/healthy_loss_morphed_b01.npy", loss_rec_array_b01, allow_pickle=True, fix_imports=True)    
        np.save("./results/"+self.experiment+"/healthy_stdlogjacdet_b1.npy", stdlogjacdet_array_b1, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/healthy_stdlogjacdet_b01.npy", stdlogjacdet_array_b01, allow_pickle=True, fix_imports=True)    

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
        loss_rec_array_b1=[]
        loss_rec_array_b1=np.array(loss_rec_array_b1)
        loss_rec_array_b01=[]
        loss_rec_array_b01=np.array(loss_rec_array_b01)                
        stdlogjacdet_array=[]
        stdlogjacdet_array=np.array(stdlogjacdet_array)
        stdlogjacdet_array_b1=[]
        stdlogjacdet_array_b1=np.array(stdlogjacdet_array_b1)        
        stdlogjacdet_array_b01=[]
        stdlogjacdet_array_b01=np.array(stdlogjacdet_array_b01)        
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
                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean=log_jac_det.mean()
                stdlogjacdet_array= np.append(stdlogjacdet_array,single_value)
                meanjacdet_array= np.append(meanjacdet_array,mean)
                    # heads #

                imagess = data[0].to(self.device)
                transformed_imagess = imagess
                encode_historyy = [x.detach().clone() for x in x_rec_dict['encode_history']]
                    
                decode_historyy = [x.detach().clone() for x in x_rec_dict['decode_history']]
                gl_priorr = x_rec_dict['x_prior'].detach().clone()
                reconstruction_b1,result_dict_b1= self.model.forward_b1(transformed_imagess,gl_priorr,encode_historyy,decode_historyy,False)
                deformation_b1 = result_dict_b1['deformation']
                reversed_b1 = result_dict_b1['x_reversed']

                reconstruction_b01,result_dict_b01= self.model.forward_b01(transformed_imagess,gl_priorr,encode_historyy,decode_historyy,False)
                deformation_b01 = result_dict_b01['deformation']
                reversed_b01 = result_dict_b01['x_reversed']
                loss_mse_b1 = self.criterion_MSE(reconstruction_b1, x)
                loss_rec_array_b1= np.append(loss_rec_array_b1,loss_mse_b1.detach().cpu().numpy())
                loss_mse_b01 = self.criterion_MSE(reconstruction_b01, x)
                loss_rec_array_b01= np.append(loss_rec_array_b01,loss_mse_b1.detach().cpu().numpy())    

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation_b1.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean= log_jac_det.mean()
                stdlogjacdet_array_b1= np.append(stdlogjacdet_array_b1,single_value)

                _,_,perc_neg_jac_det,jacdet=jacobian_determinant(deformation_b01.cpu().detach().numpy(),x_rec.cpu().detach().numpy())
                jacdet_def = (jacdet + 3).clip(1e-10,1e10)
                log_jac_det = np.log(jacdet_def)
                single_value = log_jac_det.std()
                mean= log_jac_det.mean()
                stdlogjacdet_array_b01= np.append(stdlogjacdet_array_b01,single_value)    

                if idx%260==0:
                    rec_nifti = nib.Nifti1Image(np.squeeze(x_rec.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(rec_nifti, './results/'+self.experiment+'/test_mci_'+str(idx)+'_rec.nii.gz')
                    gl_prior_nifti = nib.Nifti1Image(np.squeeze(x_.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(gl_prior_nifti, './results/'+self.experiment+'/test_mci_'+str(idx)+'_gl_prior.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(x.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_mci_'+str(idx)+'_img.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(deformation.detach().cpu()[0].numpy()) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_mci_'+str(idx)+'_deff.nii.gz')
                    x_nifti = nib.Nifti1Image(np.squeeze(jacdet) , np.eye(4))
                    nib.save(x_nifti, './results/'+self.experiment+'/test_mci_'+str(idx)+'_jacdet.nii.gz')

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
            
                    
                    elements = [img,global_prior, rec,np.abs(global_prior-img), np.abs(rec - img),jacdet_def,deff]
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
                                
                            elif i==len(elements)-1:
                                
                                if axis == 0:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, int(w / 2),:,:],axes=(1,2)), np.rot90(elements[i][np.newaxis,1, int(w / 2),:,:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_mci_'+str(idx)+'deff3.png') # .rot90(axes=(2,3)
                                elif axis == 1:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,2, :, int(w / 2),:],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :, int(w / 2),:],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_mci_'+str(idx)+'deff2.png')
                                else:
                                    temp=np.concatenate((np.rot90(elements[i][np.newaxis,1, :,:, int(w / 2)],axes=(1,2)), np.rot90(elements[i][np.newaxis,0, :,:, int(w / 2)],axes=(1,2))), 0)
                                    plot_warped_grid(ax=axarr[axis, i],disp=temp,save= './results/'+self.experiment+'/test_mci_'+str(idx)+'deff1.png')
                            elif i==len(elements)-2:    
                                axarr[axis, i].axis('off')
                                v_max = v_maxs[i]
                                c_map = 'bwr'
                                # print(elements[i].shape)
                                if axis == 0:
                                    el = np.squeeze(elements[i])[int(w / 2), :, :]
                                elif axis == 1:
                                    el = np.squeeze(elements[i])[:, int(w / 2), :]
                                else:
                                    el = np.squeeze(elements[i])[:, :, int(w / 2)] 
                                axarr[axis, i].imshow(np.squeeze(el).T, cmap=c_map, norm = MidpointNormalize(midpoint=0))
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
        np.save("./results/"+self.experiment+"/mci_stdlogjacdet_array.npy", stdlogjacdet_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/mci_meanjacdet_array.npy", meanjacdet_array, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/mci_loss_morphed_b1.npy", loss_rec_array_b1, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/mci_loss_morphed_b01.npy", loss_rec_array_b01, allow_pickle=True, fix_imports=True)    
        np.save("./results/"+self.experiment+"/mci_stdlogjacdet_b1.npy", stdlogjacdet_array_b1, allow_pickle=True, fix_imports=True)
        np.save("./results/"+self.experiment+"/mci_stdlogjacdet_b01.npy", stdlogjacdet_array_b01, allow_pickle=True, fix_imports=True)      
        plot_roc_curve_2heads(self.experiment,True)

        # if self.compute_scores:
        #     normal_key = 'Normal'
        #     for key in pred_dict.keys():
        #         if 'Normal' in key:
        #             normal_key = key
        #             break
        #     pred_cxr, label_cxr = pred_dict[normal_key]
        #     for dataset_key in self.test_data_dict.keys():
        #         print(f'Running evaluation for {dataset_key}')
        #         if dataset_key == normal_key:
        #             continue
        #         pred_ood, label_ood = pred_dict[dataset_key]
        #         predictions = np.asarray(pred_cxr + pred_ood)
        #         labels = np.asarray(label_cxr + label_ood)
        #         print('Negative Classes: {}'.format(len(np.argwhere(labels == 0))))
        #         print('Positive Classes: {}'.format(len(np.argwhere(labels == 1))))
        #         print('total Classes: {}'.format(len(labels)))
        #         print('Shapes {} {} '.format(labels.shape, predictions.shape))

        #         auprc = average_precision_score(labels, predictions)
        #         print('[ {} ]: AUPRC: {}'.format(dataset_key, auprc))
        #         auroc = roc_auc_score(labels, predictions)
        #         print('[ {} ]: AUROC: {}'.format(dataset_key, auroc))

        #         fpr, tpr, ths = roc_curve(labels, predictions)
        #         th_95 = np.squeeze(np.argwhere(tpr >= 0.95)[0])
        #         th_99 = np.squeeze(np.argwhere(tpr >= 0.99)[0])
        #         fpr95 = fpr[th_95]
        #         fpr99 = fpr[th_99]
        #         print('[ {} ]: FPR95: {} at th: {}'.format(dataset_key, fpr95, ths[th_95]))
        #         print('[ {} ]: FPR99: {} at th: {}'.format(dataset_key, fpr99, ths[th_99]))
        # logging.info('Writing plots...')

        # for metric in metrics:
        #     fig_bp = go.Figure()
        #     x = []
        #     y = []
        #     for idx, dataset_values in enumerate(metrics[metric]):
        #         dataset_name = list(self.test_data_dict)[idx]
        #         for dataset_val in dataset_values:
        #             y.append(dataset_val)
        #             x.append(dataset_name)

        #     fig_bp.add_trace(go.Box(
        #         y=y,
        #         x=x,
        #         name=metric,
        #         boxmean='sd'
        #     ))
        #     title = 'score'
        #     fig_bp.update_layout(
        #         yaxis_title=title,
        #         boxmode='group',  # group together boxes of the different traces for each value of x
        #         yaxis=dict(range=[0, 1]),
        #     )
        #     fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

        #     wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

    def pathology_localization(self, global_model):
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ MANIFOLD LEARNING TEST #################")
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'AUPRC': [],
        }
        pred_dict = dict()
        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
                'AUPRC': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                    data1 = [0]
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                # print(data[1].shape)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks>0]=1
                # print(torch.min(masks), torch.max(masks))
                x_rec, x_rec_dict = self.model(x)#, mask=masks)
                saliency = None
                x_rec = torch.clamp(x_rec, 0, 1)
                x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())
                # if 'embeddings' in x_rec_dict.keys():
                #     #     # x_res = gaussian_filter(x_res, sigma=2)
                #     #     saliency = self.get_saliency(x_rec_i.detach(), x_i.detach())
                #     x_res, saliency = self.compute_residual(x_rec, x)
                    # x_res = x_res.detach().numpy()

                for i in range(len(x)):
                    if torch.sum(masks[i][0]) > 1:
                        count = str(idx * len(x) + i)
                        x_i = x[i][0]
                        x_rec_i = x_rec[i][0]
                        x_res_i = x_res[i][0]
                        saliency_i = saliency[i][0] if saliency is not None else None

                        #
                        loss_mse = self.criterion_rec(x_rec_i, x_i)
                        test_metrics['MSE'].append(loss_mse.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        #
                        x_ = x_i.cpu().detach().numpy()
                        x_rec_ = x_rec_i.cpu().detach().numpy()
                        np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                        ssim_ = ssim(x_rec_, x_, data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

                        if 'embeddings' in x_rec_dict.keys():
                            x_res_i, saliency_i = self.compute_residual(x_rec_i, x_i)
                            x_res_orig = copy.deepcopy(x_res)
                            # saliency = self.get_saliency(x_rec.detach(), x.detach())
                            x_res_i = x_res_i * saliency_i
                            # x_res = saliency
                            if 'saliency' in x_rec_dict.keys():  # PHANES
                            #     eps = 1e-8
                                x_coarse_res = x_rec_dict['residual'][i][0].cpu().detach().numpy()
                                # saliency_coarse = x_rec_dict['saliency'][i][0].cpu().detach().numpy()
                                #
                                masked_x = x_rec_dict['masked'][i][0].cpu().detach().numpy()
                            #
                                np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec_c.npy',
                                        x_coarse_res)
                                np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_mask.npy',
                                        masked_x)
                                # saliency_i *= saliency_coarse
                            #     # x_res_i = x_res_i * saliency_i  # np.max(saliency)
                            #     # x_res_i = x_res_i / (np.max(x_res_i)+eps) * np.max(saliency_i)
                            #     # x_res_i = x_res_i * x_coarse_res

                        res_pred = np.max(x_res_i)
                        # print(x_res_i.shape, np.min(x_res_i), np.max(x_res_i))
                        # print(x_res_i.shape)
                        label = 0 if 'Normal' in dataset_key else 1
                        pred_.append(x_res_i)
                        label_.append(masks[i][0].cpu().detach().numpy())
                        auprc_slice, _, _, _ = compute_auprc(x_res_i, masks[i][0].cpu().detach().numpy())
                        test_metrics['AUPRC'].append(auprc_slice)
                        if int(count) in [0, 321, 325, 329, 545, 548, 607, 609, 616, 628]:#254, 539, 543, 545, 550, 609, 616, 628, 630, 636, 651]: # or int(count)==539: #(idx % 50) == 0:  # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                            elements = [x_, x_rec_, x_res_i, masks[i][0].cpu().detach().numpy()]
                            v_maxs = [1, 1, 0.5, 0.999]
                            titles = ['Input', 'Rec', str(res_pred), 'GT']
                            if 'embeddings' in x_rec_dict.keys():
                                if 'saliency' in x_rec_dict.keys():  # PHANES
                                    coarse_y = x_rec_dict['y_coarse'][i][0].cpu().detach().numpy()
                                    masked_x = x_rec_dict['masked'][i][0].cpu().detach().numpy()
                                    x_coarse_res = x_rec_dict['residual'][i][0].cpu().detach().numpy()
                                    saliency_coarse = x_rec_dict['saliency'][i][0].cpu().detach().numpy()
                                    elements = [x_, coarse_y, x_coarse_res, saliency_coarse, masked_x, x_rec_, x_, saliency_i, x_res_i, masks[i][0].cpu().detach().numpy()]
                                    v_maxs = [1, 1, np.max(x_coarse_res), 0.5, 1, 1, 1, 0.5, np.max(x_res_i), 0.99]  # , 0.99, 0.25]
                                    titles = ['Input', 'CR', 'CRes_'+ str(np.round(np.max(x_coarse_res), 3)), 'CSAl_' + str(np.round(np.max(saliency_coarse), 3)), 'Masked', 'Rec','Input', str(np.max(saliency)), str(res_pred), 'GT Mask']
                                else:
                                    elements = [x_, x_rec_, saliency, x_res_i, masks[i][0].cpu().detach().numpy()]
                                    v_maxs = [1, 1, 0.5, 0.1, 0.99]  # , 0.99, 0.25]
                                    titles = ['Input', 'Rec', str(np.max(saliency_i)), str(res_pred), 'GT']


                            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                            diffp.set_size_inches(len(elements) * 4, 4)
                            for idx_arr in range(len(axarr)):
                                axarr[idx_arr].axis('off')
                                v_max = v_maxs[idx_arr]
                                c_map = 'gray' if v_max == 1 else 'plasma'
                                axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                                axarr[idx_arr].set_title(titles[idx_arr])

                                wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                    wandb.Image(diffp, caption="Sample_" + str(count))]})

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        if self.compute_scores:
            for dataset_key in self.test_data_dict.keys():
                pred_ood, label_ood = pred_dict[dataset_key]
                predictions = np.asarray(pred_ood)
                labels = np.asarray(label_ood)
                predictions_all = np.reshape(np.asarray(predictions), (len(predictions), -1))  # .flatten()
                labels_all = np.reshape(np.asarray(labels), (len(labels), -1))  # .flatten()
                print(f'Nr of preditions: {predictions_all.shape}')
                aurocs = []
                auprcs = []
                dice_scores = []
                print(np.min(predictions_all), np.mean(predictions_all), np.max(predictions_all))
                print(np.min(labels_all), np.mean(labels_all), np.max(labels_all))

                auprc_, _, _, _ = compute_auprc(predictions_all, labels_all)
                print('Shapes {} {} '.format(labels.shape, predictions.shape))
                # auprc_, _, _, _ = compute_auprc(predictions, labels)
                logging.info('Total AUPRC: ' + str(auprc_))
                # print('[ {} ]: AUPRC: {}'.format(dataset_key, auprc))
                # auroc = roc_auc_score(labels, predictions)
                # print('[ {} ]: AUROC: {}'.format(dataset_key, auroc))
                #
                # fpr, tpr, ths = roc_curve(labels, predictions)
                # th_95 = np.squeeze(np.argwhere(tpr >= 0.95)[0])
                # th_99 = np.squeeze(np.argwhere(tpr >= 0.99)[0])
                # fpr95 = fpr[th_95]
                # fpr99 = fpr[th_99]
                # print('[ {} ]: FPR95: {} at th: {}'.format(dataset_key, fpr95, ths[th_95]))
                # print('[ {} ]: FPR99: {} at th: {}'.format(dataset_key, fpr99, ths[th_99]))
        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

