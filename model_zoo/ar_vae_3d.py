import torch
from torch import nn, distributions
import torch.nn.functional as F

"""
To DO:
    - Citation paper
    - __repr__: deal with the trainer config
    - do I need to update the file_path ? 
"""

class ArVAE(torch.nn.Module):
    """
    Class defining a variational auto-encoder (VAE) for MNIST images
    """
    def __init__(self,z_dim=16,input_size=784):
        super(ArVAE, self).__init__()

        self.input_size = input_size
        self.z_dim = z_dim
        self.inter_dim = 8

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.enc_lin = nn.Sequential(
            nn.Linear(2048, 256), # For [64,64] images use (512,256)
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
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.xavier_initialization()

        #self.update_filepath()

    def __repr__(self):
        """
        String representation of class
        :return: string
        """
        return 'MnistVAE' #+ self.trainer_config

    def encode(self, x):
        hidden = self.enc_conv(x)
        hidden = hidden.view(x.size(0), -1)
        hidden = self.enc_lin(hidden)
        z_mean = self.enc_mean(hidden)
        z_log_std = self.enc_log_std(hidden)
        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution

    def decode(self, z):
        hidden = self.dec_lin(z)
        hidden = hidden.view(z.size(0), -1, self.inter_dim, self.inter_dim)
        hidden = self.dec_conv(hidden)
        return hidden

    def reparametrize(self, z_dist):
        """
        Implements the reparametrization trick for VAE
        """
        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()
        return z_tilde, z_prior, prior_dist

    def forward(self, x):
        """
        Implements the forward pass of the VAE
        :param x: minist image input
            (batch_size, 28, 28)

        """
        # compute distribution using encoder
        z_dist = self.encode(x)
        # reparametrize
        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)
        # compute output of decoding layer
        output = self.decode(z_tilde).view(x.size())

        return output, {'z_mu': z_dist.mean, 'z_logvar': z_dist.scale, 'z_tilde': z_tilde,'z_dist': z_dist,'prior_dist': prior_dist}

    def xavier_initialization(self):
        """
        Initializes the network params
        :return:
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

class AR_VAEPatiLoss:

    def __init__(self, beta, gamma, factor, reg_dim):
        super(AR_VAEPatiLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.factor = factor
        self.reg_dim = reg_dim

    def __call__(self, x_recon, x, z, labels):

        #mu = z['z_mu']
        #log_var = z['z_logvar']
        #z_tilde  = z['z_tilde']

        kld_weight = 0.0128 # 128/10 000  # Account for the minibatch samples from the dataset

        batch_size = x.size(0)
        recons_loss = F.mse_loss(x_recon, x).div(batch_size)
        recons_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
        kld_loss = torch.distributions.kl.kl_divergence(z['z_dist'], z['prior_dist'])
        kld_loss = kld_loss.sum(1).mean()

        c = 0.0
        beta_loss = recons_loss + self.beta * kld_weight * (kld_loss - c).abs()

        reg_loss = 0.0
        if self.gamma != 0.0:
            reg_dim_real = labels.size(1)
            for dim in range(reg_dim_real):
                x_ = z['z_tilde'][:, dim]
                reg_loss += self.reg_loss_sign(x_, labels[:,dim], self.factor)

        global_loss = beta_loss + self.gamma * reg_loss

        #print(f'Recons_loss: {recons_loss}')
        #print(f'KLD_loss: {kld_loss}')
        #print(f'Reg_loss: {reg_loss}')
        #print(f'Bernouilli: {recons_loss}')
        return global_loss

    @staticmethod
    def reg_loss_sign(latent_code, attribute, factor):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor).cpu()
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss

if __name__ == '__main__':
    pass