import torch
from torch.nn import L1Loss, MSELoss
from merlinth.losses.pairwise_loss import mse


class L2:
    def __init__(self):
        super(L2, self).__init__()
        self.loss_ = MSELoss()

    def __call__(self, x, x_recon, z=None):
        return self.loss_(x, x_recon)


class L1:
    def __init__(self):
        super(L1, self).__init__()
        self.loss_ = L1Loss()

    def __call__(self, x, x_recon, z=None):
        return self.loss_(x, x_recon)


class MSE:
    def __init__(self, batch=True, reduce=True):
        super(MSE, self).__init__()
        self.batch = batch
        self.reduce = reduce

    def __call__(self, gt, pred):
         return mse(gt, pred, batch=self.batch, reduce=self.reduce)


class MagnPhaseL2:
    def __init__(self):
        super(MagnPhaseL2, self).__init__()
        self.loss_ = MSELoss()

    def __call__(self, gt, pred):
        return self.loss_(torch.abs(gt), torch.abs(pred)) + \
               self.loss_(torch.angle(gt), torch.angle(pred))


class MagnL2:
    def __init__(self):
        super(MagnL2, self).__init__()
        self.loss_ = MSELoss()

    def __call__(self, gt, pred):
        return self.loss_(torch.abs(gt), torch.abs(pred))


class RealImagL2:
    def __init__(self):
        super(RealImagL2, self).__init__()
        self.loss_ = MSELoss()

    def __call__(self, gt, pred):
        return (self.loss_(torch.real(gt), torch.real(pred)) +
                self.loss_(torch.imag(gt), torch.imag(pred)))
