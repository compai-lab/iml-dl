import torch
import merlinth
from merlinth.layers import ComplexConv2d, ComplexConv3d
from merlinth.layers.complex_act import cReLU
from merlinth.layers.module import ComplexModule


class ComplexUnrolledNetwork(ComplexModule):
    """ Unrolled network for iterative reconstruction

    Input to the network are zero-filled, coil-combined images, corresponding
    undersampling masks and coil sensitivtiy maps. Output is a reconstructed
    coil-combined image.

    """
    def __init__(self,
                 nr_iterations=10,
                 weight_sharing=True,
                 select_echo=False,
                 nr_filters=64,
                 kernel_size=3,
                 nr_layers=5,
                 activation="relu",
                 **kwargs):
        super(ComplexUnrolledNetwork, self).__init__()

        self.nr_iterations = nr_iterations
        self.T = 1 if weight_sharing else nr_iterations
        input_dim = 12 if not select_echo else 1

        # create layers
        self.denoiser = torch.nn.ModuleList([MerlinthComplexCNN(dim='2D',
                                                                input_dim=input_dim,
                                                                filters=nr_filters,
                                                                kernel_size=kernel_size,
                                                                num_layer=nr_layers,
                                                                activation=activation,
                                                                use_bias=True,
                                                                normalization=None,
                                                                weight_std=False,
                                                                **kwargs)
                                             for _ in range(self.T)])

        A = merlinth.layers.mri.MulticoilForwardOp(center=True, channel_dim_defined=False)
        AH = merlinth.layers.mri.MulticoilAdjointOp(center=True, channel_dim_defined=False)
        self.DC = torch.nn.ModuleList([merlinth.layers.data_consistency.DCGD(A, AH, weight_init=1.0)
                                       for _ in range(self.T)])

        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, img, y, mask, smaps):
        x = img
        for i in range(self.nr_iterations):
            ii = i % self.T
            x = x + self.denoiser[ii](x)
            x = self.DC[ii]([x, y, mask, smaps])
        return x


class MerlinthComplexCNN(ComplexModule):
    """
    This is a copy of merlinth.models.cnn.ComplexCNN since the module could not
    be loaded due to problems with incomplete optox installation.
    """
    def __init__(
        self,
        dim="2D",
        input_dim=1,
        filters=64,
        kernel_size=3,
        num_layer=5,
        activation="relu",
        use_bias=True,
        normalization=None,
        weight_std=False,
        **kwargs,
    ):
        super().__init__()
        # get correct conv operator
        if dim == "2D":
            conv_layer = ComplexConv2d
        elif dim == "3D":
            conv_layer = ComplexConv3d
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        if activation == "relu":
            act_layer = cReLU

        padding = kernel_size // 2
        # create layers
        self.ops = []
        self.ops.append(
            conv_layer(
                input_dim,
                filters,
                kernel_size,
                padding=padding,
                bias=use_bias,
                weight_std=weight_std,
                **kwargs,
            )
        )
        if normalization is not None:
            self.ops.append(normalization())

        self.ops.append(act_layer())

        for _ in range(num_layer - 2):
            self.ops.append(
                conv_layer(
                    filters,
                    filters,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if normalization is not None:
                self.ops.append(normalization())
            self.ops.append(act_layer())

        self.ops.append(
            conv_layer(
                filters,
                input_dim,
                kernel_size,
                bias=False,
                padding=padding,
                **kwargs,
            )
        )
        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # equivalent to tf.layers.xavier_initalization()
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.ops(x)
        return x
