import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from torch.autograd import Variable

import copy


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # zer = torch.cuda.FloatTensor(100, 1, 128, 128).fill_(0)
        # zer[:, :, ::8, :] = 1
        # zer[:, :, :, ::8] = 1
        # src = zer
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (1, 128, 128)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        nb_features = [
            [16, 32, 32, 32],  # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VoxelMorph(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (1, 128, 128)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        disp = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(disp)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, y_source, preint_flow)
        else:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, y_source, pos_flow)


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

# class AtlasLayer(nn.Module):
#
#     def __init__(self, inshape, initialization = None):
#         super().__init__()
#
#         self.atlas_list = nn.ParameterList()
#         # Atlas layer
#         if initialization == None:
#             self.atlas_list.append(nn.Parameter(torch.tensor(torch.randn(*inshape), requires_grad=True)))
#         else:
#             # self.atlas_list.append(nn.Parameter(data = initialization, requires_grad=True))
#             self.atlas_list.append(nn.Parameter(torch.tensor(initialization, requires_grad=True)))
#
#         ##### ToDo: Initialize and
#
#     def forward(self, x):
#
#         # Resize atlas layer to x?
#         for i, p in enumerate(self.params):
#             x_concat = torch.concat([p, x], dim=1)
#         # self.atlas_list[0].retain_grad()
#         # x_concat.retain_grad()
#
#         return x_concat


class VoxelMorph_TemplateCreation(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between input image and a learned template
    
    source image is the learned template which is supposed to be transformed to target
    """
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (1, 128, 128)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super(VoxelMorph_TemplateCreation, self).__init__()



        self.bidir = bidir

        # ensure correct dimensionalitys
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # Parameter creation dummy:
        # self.alpha = nn.Parameter(torch.tensor(torch.randn(1, 1, *inshape), requires_grad=True))

        ##### Atlas layer (theoretically not necessary if initialized)
        # self.atlas_layer = nn.ParameterList(parameters=[nn.Parameter(torch.randn(1, src_feats, *inshape))])
        # self.atlas_layer = nn.Parameter(torch.randn(1, src_feats, *inshape), requires_grad=True) 
        # self.atlas_layer = Variable(torch.randn(1, src_feats, *inshape), requires_grad=True)
        # self.atlas_layer = AtlasLayer(inshape=(1, src_feats, (*inshape)), initialization=None)
        self.atlas_layer = nn.Parameter(torch.randn(1, src_feats, *inshape), requires_grad=True)

        # Registration model
        self.vxm_dense = VoxelMorph(inshape,
                                    nb_unet_features=nb_unet_features,
                                    nb_unet_levels=nb_unet_levels,
                                    unet_feat_mult=unet_feat_mult,
                                    nb_unet_conv_per_level=nb_unet_conv_per_level,
                                    int_steps=int_steps,
                                    int_downsize=int_downsize,
                                    bidir=bidir,
                                    use_probs=use_probs,
                                    src_feats=src_feats,
                                    trg_feats=trg_feats,
                                    unet_half_res=unet_half_res,
                                    **kwargs)
    def get_atlas_tensor(self):

        return self.atlas_layer.clone().detach()

    def set_initial_atlas(self, atlas):
        ''' 
        Sets the atlas weights
        Parameters:
            inshape: shape of atlas tensor
            atlas: atlas tensor 
        '''
        # del(self.atlas_layer)
        # self.atlas_layer = AtlasLayer(inshape = atlas.shape, initialization=atlas)
        self.atlas_layer.data = torch.tensor(atlas, requires_grad=True)

    def forward(self, target, registration=True, source_for_pairwise = None):
        '''
        Parameters:
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
            source_for_pairwise: (Optional) Source image tensor for registration via template (only for inference)

        Output:
        '''
        # concatenate inputs and propagate unet
        # atlas_template = self.atlas_layer[0].cuda() # adjust depending on ParameterList
        # atlas_template = self.atlas_layer.cuda() # required if initializd as Parameter/Variable

        # ensure correct dimensionality (for torchsummary -> if no batch size is given)

        if source_for_pairwise is None:

            if len(target.shape) == 3:
                target = target[None, :] # Adds dimension in place of dimension for batch size (due to bug in torchsummary)

            assert(self.vxm_dense.bidir, "ToDo: Prediction output needs to be changed if NOT bidir")
            y_source, y_target, pos_flow = self.vxm_dense(source = self.atlas_layer, target = target, registration = registration)
            # else: y_source, y_source, pos_flow = self.vxm_dense(atlas_template, target, registration = registration)

            # template_pred = copy.deepcopy(self.atlas_layer.atlas_list[0])
            template_pred = self.atlas_layer.clone()

            return y_source, y_target, pos_flow, template_pred

        # if tensor is passed for pairwise registration (only for inference!)
        else:
            assert(target.shape == source_for_pairwise.shape)

            return self.register_pairwise(moving = source_for_pairwise, fixed = target)

            # Register

    def register_pairwise(self, moving, fixed):
        '''
       Parameters:
           source: Source image tensor.
           target: Target image tensor.
           registration: Return transformed source image and flow. Default is False.
       Output:

       Comment: Bidirectional training would help to learn?
       '''

        # template_pred = self.atlas_layer.detach().clone()

        # Transform source to template
        template = self.get_atlas_tensor()
        _, _, phi_m_t = self.vxm_dense(source = moving, target = template, registration = True)

        # Transform template to target
        _, _, phi_t_f = self.vxm_dense(source = template, target=fixed, registration=True)

        # Combine deformations
        transformer = SpatialTransformer(phi_m_t.shape[2:]) # shape of image
        pos_flow = transformer(phi_m_t, phi_t_f)

        # Warp source to target
        m_warped = self.vxm_dense.transformer(moving, pos_flow)

        return m_warped, pos_flow, phi_m_t, phi_t_f

