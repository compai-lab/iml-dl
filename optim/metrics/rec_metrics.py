import torch

def compute_MMD(x, y, device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    bandwidth_range = [0.2, 0.5, 0.9, 1.3]
    for a in bandwidth_range:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1


    return torch.mean(XX + YY - 2. * XY)