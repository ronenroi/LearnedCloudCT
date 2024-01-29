# This file contains the code for the decoder in VIP-CT framework.
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
from ProbCT.mlp_function import MLPWithInputSkips2
import torch.nn.functional as F
from .LoRA import Linear as LoRALinear
def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)

class Decoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            type,
            average_cams,
            feature_flatten,
            latent_size,
            ce_bins,
            use_neighbours,
    ):
        """
        :param type Decoder network type.
        """
        super().__init__()
        self.average_cams = average_cams
        self.feature_flatten = feature_flatten
        out_size = 27 if use_neighbours else 1
        self.type = type
        self.mask = False
        if type == 'FixCT':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, out_size)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)

            self.decoder = torch.nn.Sequential(
                linear1,
                # torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(True),
                linear2,
                # torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv2':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, out_size)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)

            self.decoder = torch.nn.Sequential(
                linear1,
                torch.nn.LayerNorm(2048),
                torch.nn.ReLU(True),
                linear2,
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv3':
            linear1 = torch.nn.Linear(latent_size, 2048)
            linear2 = torch.nn.Linear(2048, 512)
            linear3 = torch.nn.Linear(512, 64)
            linear4 = torch.nn.Linear(64, out_size)
            _xavier_init(linear1)
            _xavier_init(linear2)
            _xavier_init(linear3)
            _xavier_init(linear4)
            nn.init.constant_(linear4.bias.data, 10)

            self.decoder = torch.nn.Sequential(
                linear1,
                torch.nn.LayerNorm(2048),
                torch.nn.ReLU(True),
                linear2,
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(True),
                linear3,
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(True),
                linear4
            )
        elif type == 'FixCTv4':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),
            torch.nn.Linear(512, out_size))
        elif type == 'FixCTv4_CE':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),
            torch.nn.Linear(512, ce_bins),
            # torch.nn.Softmax(dim=-1)
            )
        elif type == 'FixCTv4_CE_mask':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),

            )
            self.decoder_out = torch.nn.Linear(512, ce_bins)
            self.mask_decoder = nn.Sequential(torch.nn.Linear(512, 1),torch.nn.Sigmoid())
            self.mask = True

        elif type == 'FixCTv4_microphysics':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),
            torch.nn.Linear(512, 3*out_size))


    def forward(self, x):
        if self.average_cams:
            x = torch.mean(x,1)
        if self.feature_flatten:
            x = x.reshape(x.shape[0],-1)
        if self.mask:
            x = self.decoder(x)
            return self.decoder_out(x), self.mask_decoder(x)
        return self.decoder(x)

    @classmethod
    def from_cfg(cls, cfg, latent_size, ce_bins, use_neighbours=False):
        return cls(
            type = cfg.decoder.name,
            average_cams=cfg.decoder.average_cams,
            feature_flatten=cfg.decoder.feature_flatten,
            latent_size = latent_size,
            ce_bins=ce_bins,
            use_neighbours = use_neighbours,
        )


class LoRA_Decoder(nn.Module):
    def __init__(
            self,
            lora_dim: int,
            lora_alpha: int,
            lora_dropout: float,
            fan_in_fan_out: bool,
            merge_weights: bool,
            type,
            average_cams,
            feature_flatten,
            latent_size,
            ce_bins,
            use_neighbours,
    ):
        super().__init__()
        self.average_cams = average_cams
        self.feature_flatten = feature_flatten
        out_size = 27 if use_neighbours else 1
        self.type = type
        self.mask = False


        assert lora_dropout == 0 , 'The encoder must be in train mode, may conflict with eval/train mode for the dropout'
        self.mask_decoder = nn.Identity()

        if type == 'FixCTv4':

            self.decoder = nn.Sequential(

                LoRALinear(latent_size, 2048,lora_dim, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights,),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
                    apply_lora=True,
                lora_dim=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out, merge_weights=merge_weights,
            ),
                LoRALinear(512, out_size, lora_dim, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights, ),)
        elif type == 'FixCTv4_CE':

            self.decoder = nn.Sequential(
                LoRALinear(latent_size, 2048,lora_dim, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights,),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                    apply_lora=True,
                    lora_dim=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out,
                    merge_weights=merge_weights,
                ),
            LoRALinear(512,ce_bins, lora_dim, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights, ),)
        elif type == 'FixCTv4_CE_mask':

            self.decoder = nn.Sequential(
                LoRALinear(latent_size, 2048,lora_dim, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights,),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                    apply_lora=True,
                    lora_dim=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out,
                    merge_weights=merge_weights,
                ),

            )
            self.decoder_out = LoRALinear(512,ce_bins, lora_dim, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights, )


            self.mask_decoder = nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Sigmoid())
            # self.mask = False

        else:
            NotImplementedError()


    def forward(self, x):
        if self.average_cams:
            x = torch.mean(x,1)
        if self.feature_flatten:
            x = x.reshape(x.shape[0],-1)
        # if self.mask:
        return self.decoder_out(self.decoder(x))
            # return self.decoder_out(x)
            # assert 'mask is not supported with LoRA yet'
        # return self.decoder(x)

    # def decoder_lora_forward(self, x):
    #     for dec_layer, lora_A, lora_B in zip(self.decoder,self.lora_A,self.lora_B):
    #         x = self.decoder_lora_layer_forward(x, dec_layer, lora_A, lora_B)
    #     return x
    # def decoder_lora_layer_forward(self, x, decoder_layer, lora_A_layer, lora_B_layer):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
    #
    #     if self.merged:
    #         return F.linear(x, T(decoder_layer.weight), bias=decoder_layer.bias)
    #     else:
    #         result = F.linear(x, T(decoder_layer.weight), bias=decoder_layer.bias)
    #         if self.r > 0:
    #             after_A = F.linear(self.lora_dropout(x), lora_A_layer)
    #             after_B = F.conv1d(
    #                 after_A.transpose(-2, -1),
    #                 lora_B_layer.unsqueeze(-1),
    #                 groups=sum(self.enable_lora)
    #             ).transpose(-2, -1)
    #             result += self.zero_pad(after_B) * self.scaling
    #         return result


    @classmethod
    def from_cfg(cls, cfg, latent_size, ce_bins, use_neighbours=False):
        return cls(
            lora_dim=cfg.decoder.lora_dim,
            lora_alpha=cfg.decoder.lora_alpha,
            lora_dropout=cfg.decoder.lora_dropout,
            fan_in_fan_out=cfg.decoder.fan_in_fan_out,
            merge_weights=cfg.decoder.merge_weights,
            type=cfg.decoder.name,
            average_cams=cfg.decoder.average_cams,
            feature_flatten=cfg.decoder.feature_flatten,
            latent_size=latent_size,
            ce_bins=ce_bins,
            use_neighbours=use_neighbours,
        )
