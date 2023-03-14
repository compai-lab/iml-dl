import abc
import math

import torch

from torch.nn import functional as F

class BaseLoss(abc.ABC):
    """Base class for loss."""

    def __init__(self, rec_dist="bernoulli", steps_anneal=0):
        """Called upon initialization.

        Parameters
        ----------
        record_loss_every: int, optional
            Every how many steps to recorsd the loss.
        rec_dist: str: {"bernoulli", "gaussian", "laplace"}, optional
            Reconstruction distribution istribution of the likelihood on the each pixel.
            Implicitely defines the reconstruction loss. Bernoulli corresponds to a
            binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
            corresponds to L1.
        steps_anneal: bool, optional
            Number of annealing steps where gradually adding the regularisation.
        """
        self.n_train_steps = 0
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, **kwargs):
        """Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).
        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).
        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).
        is_train : bool
            Whether currently in train mode.
        storer : dict
            Dictionary in which to store important variables for vizualisation.
        kwargs:
            Loss specific arguments
        """


class betatc_loss(BaseLoss):
    """Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]. Inherits from BaseLoss.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1.0, beta=6.0, gamma=1.0, is_mss=True, **kwargs):
        """Called upon initialization.

        Parameters
        ----------
        n_data: int
            Number of data in the training set
        alpha : float
            Weight of the mutual information term.
        beta : float
            Weight of the total correlation term.
        gamma : float
            Weight of the dimension-wise KL term.
        is_mss : bool
            Whether to use minibatch stratified sampling instead of minibatch
            weighted sampling.
        kwargs:
            Additional arguments for `BaseLoss`, e.g. rec_dist`.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling
        self.n_data = n_data

    def __call__(self, data, recon_batch, latent_dist, is_train, latent_sample=None):
        """Returns KL Divergence and reconstruction loss upon call.

        Parameters
        ----------
        data : torch.Tensor
            Original image(s).
        recon_batch : torch.Tensor
            Reconstructed image(s).
        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).
        is_train : bool
            Whether currently in train mode.
        latent_sample : torch.Tensor, optional
            Sample from latent distribution, by default None.

        Returns
        -------
        torch.Tensor. torch.Tensor
            Returns reconstruction loss and KL-Divergence based on beta-TCVAE loss.
        """
        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch, distribution=self.rec_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
            latent_sample, latent_dist, self.n_data, is_mss=self.is_mss
        )
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (
            linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
            if is_train
            else 1
        )

        kld = (
            self.alpha * mi_loss
            + self.beta * tc_loss
            + anneal_reg * self.gamma * dw_kl_loss
        )

        return rec_loss, kld


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).
    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).
    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.
    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """

    if len(recon_data.size()) > 4:
        batch_size, n_chan, height, width, depth = recon_data.size()
    else:
        batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = (
            loss * 3
        )  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        #assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    return loss


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    """Approximates all distributions via minibatch weighted or stratified sampling.

    Parameters
    ----------
    latent_sample : torch.Tensor, optional
        Sample from latent distribution, by default None.
    latent_dist : tuple of torch.tensor
        sufficient statistics of the latent dimension. E.g. for gaussian
        (mean, log_var) each of shape : (batch_size, latent_dim).
    n_data: int
        Number of data in the training set.
    is_mss : bool, optional
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling, by default True.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        All distributions to approximate the KL-Divergence in the beta-TCVAE loss.
    """
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False) - math.log(
        batch_size * n_data
    )
    log_prod_qzi = (
        torch.logsumexp(mat_log_qz, dim=1, keepdim=False)
        - math.log(batch_size * n_data)
    ).sum(1)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(
            latent_sample.device
        )
        log_qz = torch.logsumexp(log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(
            log_iw_mat.view(batch_size, batch_size, 1) + mat_log_qz,
            dim=1,
            keepdim=False,
        ).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


## Utils


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: torch.Tensor or np.ndarray or float
        Mean.
    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = -0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu) ** 2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch
    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[:: M + 1] = 1 / N
    W.view(-1)[1 :: M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()
