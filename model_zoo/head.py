import glob

import pytorch_lightning as pl
import torch

import yaml

from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import confusion_matrix, accuracy

from dl_utils.config_utils import check_config_file, import_module


class MLP(pl.LightningModule):

    def __init__(
        self,
        folder_name: str,
        data_dir: str,
        latent_dim: int,
        dict_classes: dict,
        blocked_latent_features: int,
        fix_weights=True,
    ):

        super().__init__()

        self.dict_classes = dict_classes
        self.num_classes = len(self.dict_classes)
        self.results_folder = data_dir + folder_name

        path_pt = data_dir + folder_name + "/best_model.pt"
        weights = torch.load(path_pt)

        #config_folder = glob.glob(f'./wandb/*{folder_name}/files/')
        model_config = data_dir + folder_name + "/config.yaml"
        stream_file = open(model_config, 'r')
        config = yaml.load(stream_file, Loader=yaml.FullLoader)
        self.dl_config = check_config_file(config)

        model_class = import_module(self.dl_config['model']['module_name'], self.dl_config['model']['class_name'])
        self.encoder = model_class(**(self.dl_config['model']['params']))

        self.model_name = self.dl_config['model']['module_name'].split('.')[1]

        if 'latent_dim' in self.dl_config['model']['params'].keys():
            self.latent_dim = self.dl_config['model']['params']['latent_dim'] # parameters of the encoder
        elif 'z_dim' in self.dl_config['model']['params'].keys():
            self.latent_dim = self.dl_config['model']['params']['z_dim'] # parameters of the encoder
        elif 'zdim' in self.dl_config['model']['params'].keys():
            self.latent_dim = self.dl_config['model']['params']['zdim']
        else:
            self.latent_dim = latent_dim # parameters in head config

        self.blocked_latent_features = list(range(blocked_latent_features[0],blocked_latent_features[1]))

        self.kept_latent_features = torch.tensor(
            [x for x in list(range(0, self.latent_dim)) if x not in blocked_latent_features]
        )

        self.encoder.load_state_dict(weights['model_weights'])

        #layer_enc_names = [name.split('.')[0] for name, param in self.model.named_parameters() if 'enc' in name]
        if fix_weights:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder.eval()

        self.fc1 = nn.Linear(len(self.kept_latent_features), 512, bias=True)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc3 = nn.Linear(512, self.num_classes, bias=True)

    def forward(self, x):
        """Predicts from encoded or not encoded image.

        Parameters
        ----------
        x : torch.Tensor
            Image or latent representation.

        Returns
        -------
        torch.Tensor
            Prediction.
        """


        if len(x.shape) >=3:
            if self.model_name == 'beta_vae_higgings':
                _, f_results = self.encoder.encode(x)
                x = f_results['z']

            elif self.model_name == 'soft_intro_vae_daniel':
                #len(x.shape) >= 3:

                mu, logvar = self.encoder.encode(x)
                """
                z_dist = self.encoder.encode(x)
                if len(z_dist.shape) == 4:
                    x = torch.squeeze(z_dist)
                elif len(z_dist)>1:
                    x,_,_,_ = self.encoder.reparameterize(z_dist[0],z_dist[1],z_dist[2])
                else:
                    x, _, _ = self.encoder.reparametrize(z_dist)
                """
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                x =  mu + eps * std
            else:
                pass
        else:
            pass

        if len(self.blocked_latent_features) > 0:
            x = x.index_select(1, self.kept_latent_features.to(x.device))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = F.softmax(self.fc3(x), dim=1)

        return y_hat





