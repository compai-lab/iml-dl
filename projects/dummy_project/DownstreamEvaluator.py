import logging
#
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
#
from torch.nn import L1Loss
#
from skimage.metrics import structural_similarity as ssim
#
import lpips
#
from dl_utils import *
from optim.metrics import *
from core.DownstreamEvaluator import DownstreamEvaluator


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.auprc = AUPRC()

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        self.test_reconstruction(global_model)

    def test_reconstruction(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ Reconstruction test #################")
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': []
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': []
            }
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if dataset_key == 'celeba':
                    data0 = data['images']
                else:
                    data0 = data[0]
                nr_batches, nr_slices, width, height = data0.shape
                x = data0.view(nr_batches * nr_slices, 1, width, height)

                x = x.to(self.device)
                x_rec, x_rec_dict = self.model(x)

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    if idx % 10 == 0 and i % 10 == 0:  # plot some images
                        grid_image = np.hstack([x_, x_rec_])

                        wandb.log({"Reconstruction_Examples/" + '_' + dataset_key + '_' + str(count) + '_' +
                                   str(ssim_): [wandb.Image(grid_image, caption="Sample_" + str(count))]})

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