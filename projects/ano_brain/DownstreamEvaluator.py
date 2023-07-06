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
from torch.nn import L1Loss
#
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from sklearn.metrics import roc_auc_score, roc_curve
from skimage import exposure
from skimage.measure import label, regionprops
from scipy.ndimage.filters import gaussian_filter

from PIL import Image
#
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

        self.criterion_rec = L1Loss().to(self.device)
       # self.auprc = AUPRC()
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)
        self.l_cos = CosineSimLoss(device='cuda')
        self.l_ncc = NCC(win=[9, 9])

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
        self.global_detection(global_model)

    def compute_residual(self, x_rec, x):
        saliency = self.get_saliency(x_rec, x)
        # x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
        # x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
        # saliency2 = self.get_saliency(torch.Tensor(x_rec_rescale).to(x_rec.device), torch.Tensor(x_rescale).to(x_rec.device))
        # saliency = (saliency + saliency2) / 2
        # saliency = (saliency * saliency2)
        # saliency = saliency2
        x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())
        # x_res = np.abs(x_rec_rescale - x_rescale)
        # x_res += x_res_2
        # perc95 = np.quantile(x_res, 0.99)
        # eps=1e-8
        # x_res = x_res / (perc95 + eps)
        # x_res[x_res > 1] = 1
        # x_res /= 2
        return x_res, saliency


    def lpips_loss(self, ph_img, anomaly_img, mode=0):
        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        loss_lpips = self.l_pips_sq(anomaly_img, ph_img, normalize=True, retPerLayer=False)
        return loss_lpips.cpu().detach().numpy()


    def convert_to_grayscale(self, im_as_arr):
        grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
        im_max = np.percentile(grayscale_im, 99)
        # im_max = np.max(grayscale_im)
        im_min = np.min(grayscale_im)
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
        grayscale_im = np.expand_dims(grayscale_im, axis=0)
        return grayscale_im


    def get_saliency(self, x_rec, x):

        # saliency = self.lpips_loss(x_rec, x)
        saliency = self.lpips_loss(2*x_rec-1, 2*x-1)
        # saliency[saliency>1]=1
        saliency = gaussian_filter(saliency, sigma=2)

        # saliency = self.convert_to_grayscale(saliency)
        return saliency

    def curate_dataset(self, global_model):
        logging.info("################ Curating Dataset #################")
        # lpips_alex = lpips.LPIPS(net='vgg')  # best forward scores

        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            if not os.path.exists(self.image_path + dataset_key):
                os.makedirs(self.image_path + dataset_key)
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                    data1 = [0]
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                masks_bool = True if len(data1) > 2 else False
                x = data0.to(self.device)
                masks = data[1].to(self.device) if masks_bool else None

                im = Image.fromarray((torch.squeeze(x).cpu().detach().numpy() * 255).astype(np.uint8))
                im.save(self.image_path + dataset_key + '/' + dataset_key + str(idx) + '.png')
                if masks_bool:
                    masks[masks > 0] = 1
                    im = Image.fromarray((torch.squeeze(masks).cpu().detach().numpy() * 255).astype(np.uint8))
                    im.save(self.image_path + dataset_key + '/' + dataset_key + str(idx) + '_mask.png')
                x_rec, x_rec_dict = self.model(x)
                im = Image.fromarray((torch.squeeze(x_rec).cpu().detach().numpy() * 255).astype(np.uint8))
                if not os.path.exists(self.image_path + dataset_key):
                    os.makedirs(self.image_path + dataset_key)
                im.save(self.image_path + dataset_key + '/' + dataset_key + str(idx) + '_ph.png')


    def pseudo_healthy(self, global_model):
        # synth_ = SyntheticRect()
        synth_ = SyntheticSprites()
        # synth_ = CopyPaste()
        """
               Validation of downstream tasks
               Logs results to wandb

               :param global_model:
                   Global parameters
               """
        logging.info("################ Pseudo Healthy TEST #################")
        # lpips_alex = lpips.LPIPS(net='vgg')  # best forward scores
        lpips_alex = lpips.LPIPS(pretrained=True, net='alex', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)

        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'MSE_an': [],
            'LPIPS': [],
            'LPIPS_an': [],
            'SSIM': [],
            'AUPRC': [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'MSE_an': [],
                'LPIPS': [],
                'LPIPS_an': [],
                'SSIM': [],
                'AUPRC': [],
            }
            pred = []
            gt = []
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                # x_synth = x
                # mask_synth = np.ones(x.shape)
                x_synth, mask_synth = synth_(copy.deepcopy(x).cpu().numpy())
                x_synth = torch.from_numpy(x_synth).to(self.device)
                x_rec, x_rec_dict = self.model(x_synth)
                # x_rec, x_rec_dict = self.model(x_synth, torch.tensor(mask_synth.astype(np.float32)).to(self.device))
                # ims = Image.fromarray((torch.squeeze(x_synth).cpu().detach().numpy() * 255).astype(np.uint8))
                # ims.save(self.image_path + '/Brain_' + str(idx) + '_synth.png')
                # imm = Image.fromarray((np.squeeze(mask_synth) * 255).astype(np.uint8))
                # imm.save(self.image_path + '/Brain_' + str(idx) + '_mask.png')
                # imgt = Image.fromarray((torch.squeeze(x).cpu().detach().numpy() * 255).astype(np.uint8))
                # imgt.save(self.image_path + '/Brain_' + str(idx) + '_gt.png')

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)

                    loss_lpips = np.squeeze(lpips_alex(x, x_rec, normalize=True).cpu().detach().numpy())
                    loss_lpips_an = copy.deepcopy(loss_lpips) * np.squeeze(np.abs(1-mask_synth))
                    loss_lpips_an[loss_lpips_an == 0] = np.nan
                    test_metrics['LPIPS'].append(np.nanmean(loss_lpips_an))

                    loss_lpips *= np.squeeze(mask_synth)
                    loss_lpips[loss_lpips == 0] = np.nan
                    loss_lpips = np.nanmean(loss_lpips)
                    test_metrics['LPIPS_an'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_synth_ = x_synth.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    masked_mse = np.sum(((x_rec_ - x_) * np.squeeze(mask_synth)) ** 2.0) / np.sum(mask_synth)
                    test_metrics['MSE_an'].append(masked_mse)
                    masked_mse_an = np.sum(((x_rec_ - x_) * np.squeeze(np.abs(1-mask_synth))) ** 2.0) / np.sum(mask_synth)
                    test_metrics['MSE'].append(masked_mse_an)

                    anomaly_map = np.abs(x_rec_ - x_synth_)
                    if 'embeddings' in x_rec_dict.keys():
                        anomaly_map, saliency = self.compute_residual(x_rec_i, x_synth[i][0])
                        x_res_orig = copy.deepcopy(anomaly_map)
                        #     # saliency = self.get_saliency(x_rec.detach(), x.detach())
                        # anomaly_map = anomaly_map * saliency
                        # saliency_coarse = x_rec_dict['saliency'][i][0].cpu().detach().numpy()
                        # saliency *= saliency_coarse
                        anomaly_map = anomaly_map * saliency
                        # x_coarse_res = x_rec_dict['residual'][i][0].cpu().detach().numpy()
                        # anomaly_map = anomaly_map * x_coarse_res
                    # x_bin = copy.deepcopy(anomaly_map)
                    # x_bin[x_bin < th] = 0
                    # x_bin[x_bin > 0] = 1
                    # dice_ = compute_dice(x_bin , np.squeeze(mask_synth))
                    auprc_, _, _,  _ =  compute_auprc(anomaly_map, mask_synth)
                    pred.append(anomaly_map)
                    gt.append(mask_synth)
                    test_metrics['AUPRC'].append(auprc_)

                    ssim_val, ssim_map = ssim(x_rec_, x_, data_range=1., full=True)
                    ssim_map *= np.squeeze(mask_synth)
                    ssim_map[ssim_map == 0] = np.nan
                    ssim_ = np.nanmean(ssim_map)
                    test_metrics['SSIM'].append(ssim_)


                    if (idx % 1) == 0:  # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        elements = [x_synth_, x_rec_, x_, anomaly_map, mask_synth]
                        v_maxs = [1, 1, 1, np.max(anomaly_map), 0.99]
                        titles = ['Input', 'Rec', 'GT',  str(np.round(loss_lpips,2)), 'Mask']
                        # if 'embeddings' in x_rec_dict.keys():
                        #     # elements = [x_synth_, x_rec_, x_, x_res_orig, saliency, anomaly_map, mask_synth]
                        #     # v_maxs = [1, 1, 1, 0.5, 0.5, 0.125, 1]
                        #     # titles = ['Input', 'Rec', 'GT', 'Res', 'Sal', 'Combo', str(np.round(loss_lpips,2)), 'Mask']
                        #     coarse_y = x_rec_dict['y_coarse'][i][0].cpu().detach().numpy()
                        #     masked_x = x_rec_dict['masked'][i][0].cpu().detach().numpy()
                        #     x_coarse_res = x_rec_dict['residual'][i][0].cpu().detach().numpy()
                        #     saliency_coarse = x_rec_dict['saliency'][i][0].cpu().detach().numpy()
                        #     elements = [x_synth_, coarse_y, x_coarse_res, saliency_coarse, masked_x, x_rec_, x_, anomaly_map, mask_synth]
                        #     # elements = [x_synth_, coarse_y, x_coarse_res, saliency_coarse, masked_x, x_rec_, x_, np.abs(x_rec_ - x_), x_res_orig, saliency, anomaly_map]
                        #     # v_maxs = [1, np.max(coarse_y), np.max(x_coarse_res), 0.5, 1, 1, 1, 0.5, 0.5, 0.5, 0.25]  # , 0.99, 0.25]
                        #     v_maxs = [1, 1, np.max(x_coarse_res), 0.5, 1, 1, 1, np.max(anomaly_map), 0.999]# 0.5, 0.5, 0.25]  # , 0.99, 0.25]
                        #     titles = ['Input', 'Rec Coarse', str(np.max(x_coarse_res)), 'CSAl', 'Masked', 'Rec','Input', str(np.max(anomaly_map)), 'GT']#str(np.max(np.abs(x_rec_ - x_))), str(np.max(x_res_orig)), str(np.max(saliency)), str(np.max(anomaly_map))]

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

            auprc_, _, _, _ = compute_auprc(np.asarray(pred), np.asarray(gt))
            auroc = roc_auc_score(y_true= np.asarray(gt).flatten(), y_score=np.asarray(pred).flatten())
            logging.info('Total AUROC: ' + str(auroc))

            dices = []
            dice_ranges = np.linspace(0, np.max(np.asarray(pred)), 1000)
            logging.info('Total AUPRC: ' + str(auprc_))
            for i in range(1000):
                th = dice_ranges[i]
                dice_i = compute_dice(copy.deepcopy(np.asarray(pred)), np.asarray(gt), th)
                dices.append(dice_i)
                # print(i, th, dice_i)
            dice_ = np.max(np.asarray(dices))
            logging.info('Total DICE: ' + str(dice_))

            # test_metrics['AUPRC'].append(auprc_)
            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

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

    def global_detection(self, global_model):
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
            'SSIM': []
        }
        pred_dict = dict()
        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': []
            }
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                saliency = None
                x_res = np.abs(x_rec - x.detach().cpu().numpy())
                x_rec=torch.from_numpy(x_rec)
                x_res.to(self.device)
                if 'embeddings' in x_rec_dict.keys():
                    #     # x_res = gaussian_filter(x_res, sigma=2)
                    #     saliency = self.get_saliency(x_rec_i.detach(), x_i.detach())
                    x_res, saliency = self.compute_residual(x_rec, x)
                    # x_res = x_res.detach().numpy()

                for i in range(len(x)):
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
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    x_res = np.abs(x_rec_ - x_)
                    # if 'embeddings' in x_rec_dict.keys():
                    #     x_res, saliency = self.compute_residual(x_rec, x)
                    #     x_res_orig = copy.deepcopy(x_res)
                    #     # saliency = self.get_saliency(x_rec.detach(), x.detach())
                    #     # x_res = x_res * saliency
                    #     x_res = saliency
                    if 'embeddings' in x_rec_dict.keys():
                        x_res_orig = copy.deepcopy(x_res_i)
                        x_coarse_res = x_rec_dict['residual'][i][0]
                        saliency_coarse = x_rec_dict['saliency'][i][0]
                        # x_res_i[x_res_i>0.1] = 1
                        # x_coarse_res[x_coarse_res>0.1] = 1
                        # x_res_i = gaussian_filter((x_res_i * x_coarse_res), sigma=2) * (
                        #             saliency_i * saliency_coarse)
                        x_ress = x_res_i
                        # x_ress[x_ress>0.15] = 1
                        # x_res_i = x_ress + (saliency_i)
                        x_res_i = (x_res_i * x_coarse_res)

                    res_pred = np.mean(x_res_i)
                    label = 0 if 'Normal' in dataset_key else 1
                    pred_.append(res_pred)
                    label_.append(label)

                    if (idx % 30) == 0:  # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        elements = [x_, x_rec_, x_res]
                        v_maxs = [1, 1, 0.5]
                        titles = ['Input', 'Rec', str(res_pred)]
                        # if 'embeddings' in x_rec_dict.keys():
                        #     coarse_y = x_rec_dict['y_coarse'][i][0].cpu().detach().numpy()
                        #     masked_x = x_rec_dict['masked'][i][0].cpu().detach().numpy()
                        #     x_coarse_res = x_rec_dict['residual'][i][0]
                        #     saliency_coarse = x_rec_dict['saliency'][i][0]
                        #     elements = [x_, coarse_y, x_coarse_res, saliency_coarse, masked_x, x_rec_, x_, np.abs(x_rec_ - x_), x_res_orig, saliency, x_res]
                        #     v_maxs = [1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.5, 0.25]  # , 0.99, 0.25]
                        #     titles = ['Input', 'CR', 'CRes', 'CSAl', 'Masked', 'Rec','Input', str(np.max(np.abs(x_rec_ - x_))), str(np.max(x_res_orig)), str(np.max(saliency)), str(res_pred)]

                        if 'embeddings' in x_rec_dict.keys():
                            coarse_y = x_rec_dict['y_coarse'][i][0].cpu().detach().numpy()
                            masked_x = x_rec_dict['masked'][i][0].cpu().detach().numpy()
                            x_coarse_res = x_rec_dict['residual'][i][0]
                            saliency_coarse = x_rec_dict['saliency'][i][0]
                            elements = [x_, coarse_y, x_coarse_res, saliency_coarse, masked_x, x_rec_, x_,
                                        x_res_orig, x_res_orig * x_coarse_res, saliency_i, saliency_i * saliency_coarse,
                                        x_res_i]
                            v_maxs = [1, 1, np.max(x_coarse_res), np.max(saliency_coarse), 1, 1, 1, np.max(x_res_orig),
                                      np.max(x_res_orig * x_coarse_res), np.max(saliency_i), np.max(saliency_i * saliency_coarse), np.max(x_res_i)]
                            titles = ['Input', 'C_Rec', str(np.max(x_coarse_res)), str(np.max(saliency_coarse)), 'Masked', 'Rec', 'Input',
                                      str(np.round(np.max(x_res_orig), 3)), str(np.round(np.max(x_res_orig * x_coarse_res), 3)),
                                      str(np.round(np.max(saliency_i), 3)), str(np.round(np.max(saliency_i*saliency_coarse), 3)),
                                      str(np.round(np.max(x_res_i), 3)) + '/' + str(np.round(np.sum(x_res_i), 3))]

                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'jet'
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
            normal_key = 'Normal'
            for key in pred_dict.keys():
                if 'Normal' in key:
                    normal_key = key
                    break
            pred_cxr, label_cxr = pred_dict[normal_key]
            for dataset_key in self.test_data_dict.keys():
                print(f'Running evaluation for {dataset_key}')
                if dataset_key == normal_key:
                    continue
                pred_ood, label_ood = pred_dict[dataset_key]
                predictions = np.asarray(pred_cxr + pred_ood)
                labels = np.asarray(label_cxr + label_ood)
                print('Negative Classes: {}'.format(len(np.argwhere(labels == 0))))
                print('Positive Classes: {}'.format(len(np.argwhere(labels == 1))))
                print('total Classes: {}'.format(len(labels)))
                print('Shapes {} {} '.format(labels.shape, predictions.shape))

                auprc = average_precision_score(labels, predictions)
                print('[ {} ]: AUPRC: {}'.format(dataset_key, auprc))
                auroc = roc_auc_score(labels, predictions)
                print('[ {} ]: AUROC: {}'.format(dataset_key, auroc))

                fpr, tpr, ths = roc_curve(labels, predictions)
                th_95 = np.squeeze(np.argwhere(tpr >= 0.95)[0])
                th_99 = np.squeeze(np.argwhere(tpr >= 0.99)[0])
                fpr95 = fpr[th_95]
                fpr99 = fpr[th_99]
                print('[ {} ]: FPR95: {} at th: {}'.format(dataset_key, fpr95, ths[th_95]))
                print('[ {} ]: FPR99: {} at th: {}'.format(dataset_key, fpr99, ths[th_99]))
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

