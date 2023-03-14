import torch
from torch import nn, distributions

from model_zoo.ar_vae import ArVAE


class ArVAECardiac(ArVAE):
    def __init__(self, z_dim=16, input_size=784):
        super(ArVAE, self).__init__()

        self.input_size = input_size
        self.z_dim = z_dim
        self.inter_dim = int(input_size[0] / 16) # image reduce 4 times

        self.linear_input_size = 256 * self.inter_dim * self.inter_dim

        k_s = 3 # kernel_size

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, k_s, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, k_s, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, k_s, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, k_s, 2, 1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(32, 32, k_s, 2, 1),
            #nn.ReLU(inplace=True)
        )
        self.enc_lin = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.enc_mean = nn.Linear(256, self.z_dim)
        self.enc_log_std = nn.Linear(256, self.z_dim)
        self.dec_lin = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.linear_input_size),
            nn.ReLU(inplace=True),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, k_s, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, k_s, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, k_s, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, k_s, 2, 1, 1),
            #nn.Sigmoid()
        )

        self.xavier_initialization()
    def __repr__(self):
        """
        String representation of class
        :return: string
        """
        return 'Cardiac' #"+ self.trainer_config