# This file contains the code for the encoder in VIP-CT framework.
# It is based on pixelNeRF source code ('https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/src/model/encoder.py#L180')
# Copyright (c) pixelNeRF, authors and thier affiliates.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Roi Ronen, Vadim Holodovsky and Yoav. Y. Schechner, "Variable Imaging Projection Cloud Scattering Tomography",
# Proc. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# authors. The authors are not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from ProbCT.util import nn_util as util
from .roi_align import ROIAlign
from .MyResNetFPN import resnet_fpn_backbone

class Backbone(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            backbone="resnet34",
            image_size=-1,
            pretrained=True,
            num_layers=4,
            index_interp="bilinear",
            index_padding="border",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
            sampling_output_size=8,
            sampling_support = 8,
            out_channels = 256,
            in_channels = 1,
            n_sampling_nets=1,
            to_flatten = False,
            modify_first_layer=True
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        assert sampling_output_size>0
        if norm_type != "batch":
            assert not pretrained
        self.name = backbone
        self.sampling_support = sampling_support
        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)
        self.num_layers = num_layers
        self.is_fpn = 'fpn' in backbone
        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        elif self.is_fpn:
            print("Using torchvision", backbone, "encoder")
            extractor =  backbone.split('_')[0]
            self.model = resnet_fpn_backbone(backbone_name=extractor,
                pretrained=pretrained, norm_layer=norm_layer, out_channels=out_channels, trainable_layers=5)
            if modify_first_layer:
                self.model.body.conv1 = nn.Conv2d(in_channels, self.model.body.conv1.out_channels, kernel_size=3,
                                             stride=1, padding=1,
                                             bias=self.model.body.conv1.bias != None)
                self.samplers = [ROIAlign((sampling_output_size, sampling_output_size), 0.5 ** scale, 0) for scale in
                                 range(5)]
            else:
                self.model.body.conv1 = nn.Conv2d(1, self.model.body.conv1.out_channels, kernel_size=self.model.body.conv1.kernel_size,
                                             stride=self.model.body.conv1.stride, padding=self.model.body.conv1.padding,
                                             bias=self.model.body.conv1.bias!=None)
                self.samplers = [ROIAlign((sampling_output_size, sampling_output_size), 0.5 ** scale, 0) for scale in
                                 range(5)]
            self.model.body.maxpool = nn.Sequential()
            if backbone == 'fasterrcnn_resnet50_fpn':
                self.latent_size = [0, 64, 320, 832, 1856][num_layers]
            elif backbone == 'resnet50_fpn':
                self.latent_size = out_channels * 4  # 256*5

        else:
            print("Using torchvision", backbone, "encoder")
            if 'resnet'in backbone:
                self.model = getattr(torchvision.models, backbone)(
                    pretrained=pretrained, norm_layer=norm_layer
                )
                if modify_first_layer:
                    self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=3,
                                                 stride=1, padding=1,
                                                 bias=self.model.conv1.bias != None)
                    self.model.maxpool = nn.Sequential()
                    self.samplers = [ROIAlign((sampling_output_size, sampling_output_size), 0.5 ** scale, 0) for scale
                                     in
                                     [0, 0, 1, 2]]

                else:
                    self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels,
                                                 kernel_size=self.model.conv1.kernel_size,
                                                 stride=self.model.conv1.stride, padding=self.model.conv1.padding,
                                                 bias=self.model.conv1.bias != None)
                    self.samplers = [ROIAlign((sampling_output_size, sampling_output_size), 0.5 ** scale, 0) for scale
                                     in
                                     range(1, 1 + self.num_layers)]

                    # Following 2 lines need to be uncommented for older cfgigs
                    self.model.fc = nn.Sequential()
                    self.model.avgpool = nn.Sequential()
                    if backbone == 'resnet34':
                        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
                    elif backbone == 'resnet50':
                        self.latent_size = [0, 64, 320, 832, 1856][num_layers]
                    elif backbone == 'resnet101':
                        self.latent_size = [0, 64, 320, 832, 1856][num_layers]

                    else:
                        NotImplementedError()
            elif 'swin' in backbone: #swin transformer
                init_weights = None
                if backbone == 'swin_v2_t':
                    if pretrained:
                        init_weights = torchvision.models.swin_transformer.Swin_V2_T_Weights.IMAGENET1K_V1
                    self.latent_size = [96, 96 + 192, 96 + 192 + 384, 96 + 192 + 384 + 768][num_layers - 1]

                elif backbone == 'swin_v2_b':
                    if pretrained:
                        init_weights = torchvision.models.swin_transformer.Swin_V2_B_Weights.IMAGENET1K_V1
                    self.latent_size = [128, 128 + 256, 128 + 256 + 512, 128 + 256 + 512 + 1024][num_layers - 1]

                self.model = getattr(torchvision.models, backbone)(weights=init_weights)
                self.model=self.model.features
                conv_in = self.model[0][0]

                if modify_first_layer:
                    self.model[0][0] = nn.Conv2d(1, conv_in.out_channels, kernel_size=3,
                                                 stride=1, padding=1,
                                                 bias=conv_in.bias != None)
                    self.model.maxpool = nn.Sequential()

                else:
                    self.model[0][0] = nn.Conv2d(1, conv_in.out_channels,
                                                 kernel_size=conv_in.kernel_size,
                                                 stride=conv_in.stride, padding=conv_in.padding,
                                                 bias=conv_in.bias != None)


                self.model.maxpool = nn.Sequential()
                self.layers = [1, 3, 5, 8][:self.num_layers]
                self.samplers = [ROIAlign((sampling_output_size, sampling_output_size), 0.5 ** scale, 0) for scale
                                 in
                                 [0, 1, 2, 3]]

            else:
                NotImplementedError()
        self.sampling_output_size = sampling_output_size

        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.to_flatten = to_flatten

        self.net = nn.Linear(sampling_output_size * sampling_output_size, 1, bias=True) if n_sampling_nets==1\
            else Sampling_Weighting(sampling_output_size,n_sampling_nets)

    def sample_roi(self, latents, box_centers):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param latent (B, C, H, W) images features
        :param box_center (B, N, 2) box center in pixels (x,y)
        :return (B, C, N) L is latent size
        """
        # if uv.shape[0] == 1 and latents.shape[0] > 1:
        #     uv = uv.expand(latents.shape[0], -1, -1)
        # image_size = torch.tensor(image_size, device=uv.device)
        # m = (image_size -1) / 2
        # uv /= m
        # uv -= 1

        assert len(box_centers)==1
        box_centers = box_centers[0]
        boxes = [self.make_boxes(box_center) for box_center in box_centers]
        samples = torch.empty(0,device=latents[0].device)
        for latent, sampler in zip(latents, self.samplers):
            latent = latent.view(-1,*latent.shape[2:])
            roi_features = sampler(latent,boxes)#.reshape(box_centers.shape[0],box_centers.shape[1],-1) #
            samples = torch.cat((samples, roi_features),dim=1)

        samples = [torch.squeeze(self.net(samples.reshape(samples.shape[0],samples.shape[1],-1)),-1).reshape(*box_centers.shape[:2],-1)]


        # samples = [torch.squeeze(self.net(samples.reshape(samples.shape[0],samples.shape[1],-1)),-1)]
        # chuncks =  [box.shape[0] for box in boxes]
        # samples = torch.split(samples,chuncks)
        # if self.to_flatten:
        #     samples = [sample.reshape(*box_center.shape[:-1], -1).transpose(0, 1).reshape(box_center.shape[1], -1) for
        #                sample, box_center in zip(samples, box_centers)]
        # else:
        #     samples = [sample.reshape(*box_center.shape[:-1], -1).transpose(0, 1) for
        #                sample, box_center in zip(samples, box_centers)]
            # samples.splitreshape(*box_centers.shape[:-1],-1)
        # samples = self.net(samples.reshape(samples.shape[0],samples.shape[1],-1)).reshape(*box_centers.shape[:-1],-1)
        return samples #.transpose(2,3) # (B, Cams,points,features)

    def sample_roi_debug(self, latents, box_centers):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param latent (B, C, H, W) images features
        :param box_center (B, N, 2) box center in pixels (x,y)
        :return (B, C, N) L is latent size
        """
        # if uv.shape[0] == 1 and latents.shape[0] > 1:
        #     uv = uv.expand(latents.shape[0], -1, -1)
        # image_size = torch.tensor(image_size, device=uv.device)
        # m = (image_size -1) / 2
        # uv /= m
        # uv -= 1

        assert len(box_centers)==1
        box_centers = box_centers[0]
        boxes = [self.make_boxes(box_center) for box_center in box_centers]
        samples = torch.empty(0,device=latents[0].device)
        for latent, sampler in zip(latents, self.samplers):
            latent = latent.view(-1,*latent.shape[2:])
            roi_features = sampler(latent,boxes)#.reshape(box_centers.shape[0],box_centers.shape[1],-1) #
            samples = torch.cat((samples, roi_features),dim=1)

        samples = [torch.squeeze(samples.reshape(samples.shape[0],samples.shape[1],-1),-1).reshape(*box_centers.shape[:2],-1)]

        return samples

    def make_boxes(self, box_centers):
        d = (self.sampling_support-1) / 2
        x1y1 = box_centers - d
        x2y2 = box_centers + d
        # boxes = list(torch.cat((x1y1,x2y2),dim=-1).view(-1,box_centers.shape[-2],4))
        try:
            boxes = torch.cat((x1y1,x2y2),dim=-1).view(-1,4)
        except:
            print()
        return boxes

    def sample_features(self, latents, uv):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param latent (B, C, H, W) images features
        :param uv (B, N, 2) image points (x,y)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :return (B, C, N) L is latent size
        """
        # if uv.shape[0] == 1 and latents.shape[0] > 1:
        #     uv = uv.expand(latents.shape[0], -1, -1)
        # image_size = torch.tensor(image_size, device=uv.device)
        # m = (image_size -1) / 2
        # uv /= m
        # uv -= 1

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = torch.empty(0,device=uv.device)
        for latent in latents:
            samples = torch.cat((samples, torch.squeeze(F.grid_sample(
                latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            ))),dim=1)
        return samples # (Cams,cum_channels, N)
    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """

        input_size = torch.tensor(x.shape[-2:])
        if self.use_custom_resnet or self.is_fpn:
            latents = self.model(x)
            del latents['pool']
            latents = [v for k,v in latents.items()]
        elif 'swin' in self.name:
            latents = []
            for layer_index, layer in enumerate(self.model):
                x=layer(x)
                if layer_index in self.layers:
                    latents.append(x.permute(0,3,1,2))
        else:#resnt
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

        self.latent_scaling = [(torch.tensor(latent.shape[-2:]) / input_size).to(device=x.device) for latent in latents]
        # return latent
        return latents
    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            cfg.backbone.name,
            cfg.data.image_size,
            pretrained=cfg.backbone.pretrained,
            num_layers=cfg.backbone.num_layers,
            index_interp=cfg.backbone.index_interp,
            index_padding=cfg.backbone.index_padding,
            upsample_interp=cfg.backbone.upsample_interp,
            feature_scale=cfg.backbone.feature_scale,
            use_first_pool=cfg.backbone.use_first_pool,
            sampling_output_size=cfg.backbone.sampling_output_size,
            sampling_support=cfg.backbone.sampling_support,
            out_channels = cfg.backbone.out_channels,
            in_channels = cfg.backbone.in_channels if hasattr(cfg.backbone,'in_channels') else 1,
            n_sampling_nets = cfg.backbone.n_sampling_nets,
            to_flatten = cfg.backbone.feature_flatten,
            modify_first_layer = cfg.backbone.modify_first_layer
        )


class ConvEncoder(nn.Module):
    """
    Basic, extremely simple convolutional encoder
    """

    def __init__(
        self,
        dim_in=1,
        norm_layer=util.get_norm_layer("group"),
        padding_type="reflect",
        use_leaky_relu=True,
        use_skip_conn=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.padding_type = padding_type
        self.use_skip_conn = use_skip_conn

        # TODO: make these cfgigurable
        first_layer_chnls = 64
        mid_layer_chnls = 128
        last_layer_chnls = 128
        n_down_layers = 3
        self.n_down_layers = n_down_layers

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim_in, first_layer_chnls, kernel_size=7, stride=2, bias=False),
            norm_layer(first_layer_chnls),
            self.activation,
        )

        chnls = first_layer_chnls
        for i in range(0, n_down_layers):
            conv = nn.Sequential(
                nn.Conv2d(chnls, 2 * chnls, kernel_size=3, stride=2, bias=False),
                norm_layer(2 * chnls),
                self.activation,
            )
            setattr(self, "conv" + str(i), conv)

            deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    4 * chnls, chnls, kernel_size=3, stride=2, bias=False
                ),
                norm_layer(chnls),
                self.activation,
            )
            setattr(self, "deconv" + str(i), deconv)
            chnls *= 2

        self.conv_mid = nn.Sequential(
            nn.Conv2d(chnls, mid_layer_chnls, kernel_size=4, stride=4, bias=False),
            norm_layer(mid_layer_chnls),
            self.activation,
        )

        self.deconv_last = nn.ConvTranspose2d(
            first_layer_chnls, last_layer_chnls, kernel_size=3, stride=2, bias=True
        )

        self.dims = [last_layer_chnls]

    def forward(self, x):
        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_in)
        x = self.conv_in(x)

        inters = []
        for i in range(0, self.n_down_layers):
            conv_i = getattr(self, "conv" + str(i))
            x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=conv_i)
            x = conv_i(x)
            inters.append(x)

        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_mid)
        x = self.conv_mid(x)
        x = x.reshape(x.shape[0], -1, 1, 1).expand(-1, -1, *inters[-1].shape[-2:])

        for i in reversed(range(0, self.n_down_layers)):
            if self.use_skip_conn:
                x = torch.cat((x, inters[i]), dim=1)
            deconv_i = getattr(self, "deconv" + str(i))
            x = deconv_i(x)
            x = util.same_unpad_deconv2d(x, layer=deconv_i)
        x = self.deconv_last(x)
        x = util.same_unpad_deconv2d(x, layer=self.deconv_last)
        return x


class Sampling_Weighting(nn.Module):
    def __init__(
        self,
        sampling_output_size=3,
        n_sampling_nets=1,
    ):
        super(Sampling_Weighting, self).__init__()
        self.n_sampling_nets = n_sampling_nets
        self.model = nn.ModuleList([nn.Linear(sampling_output_size * sampling_output_size, 1, bias=True)]*n_sampling_nets)


    def forward(self, x):
        if self.n_sampling_nets==1:
            return self.model[0](x)
        else:
            x = x.reshape(self.n_sampling_nets,-1,x.shape[-2],x.shape[-1])
            x = [net(b) for b, net in zip(x,self.model)]
            x = torch.vstack(x)
            return x
