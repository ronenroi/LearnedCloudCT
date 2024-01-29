# This file contains the main VIP-CT and ProbCT framework code.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# authors. The authors are not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from LearnedCloudCT.scene.volumes import Volumes
from typing import List, Optional, Tuple
import torch
from LearnedCloudCT.scene.cameras import PerspectiveCameras, AirMSPICameras
from LearnedCloudCT.ProbCT.mlp_function import MLPWithInputSkips, MLPIdentity
from LearnedCloudCT.ProbCT.encoder.encoder import Backbone
from LearnedCloudCT.ProbCT.decoder.decoder import Decoder, LoRA_Decoder
from .decoder.transformer import DecoderTransformer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CTnetV2(torch.nn.Module):

    def __init__(
        self,
        cfg,
        n_cam
    ):

        super().__init__()

        n_layers_xyz = cfg.ct_net.n_layers_xyz
        append_xyz = cfg.ct_net.append_xyz
        self.use_neighbours = cfg.ct_net.use_neighbours if hasattr(cfg.ct_net,'use_neighbours') else False
        self._image_encoder = Backbone.from_cfg(cfg)
        self.stop_encoder_grad = cfg.ct_net.stop_encoder_grad
        self.n_query = cfg.ct_net.n_query
        self.val_n_query = cfg.ct_net.val_n_query
        self.n_cam = n_cam
        self.query_point_method = cfg.ct_net.query_point_method
        self.decoder_masked = False if cfg.decoder.masked == 'None' else cfg.decoder.masked
        self.num_masked = 0 if cfg.decoder.num_masked == 'None' else cfg.decoder.num_masked
        self.query_point_val_method = cfg.ct_net.query_point_val_method if hasattr(cfg.ct_net,'query_point_val_method') else 'all'
        self.decoder_batchify = cfg.ct_net.decoder_batchify if hasattr(cfg.ct_net,'decoder_batchify') else False
        self.mask_type = None if cfg.ct_net.mask_type == 'None' else cfg.ct_net.mask_type
        self.val_mask_type = None if cfg.ct_net.val_mask_type == 'None' else cfg.ct_net.val_mask_type
        self.decoder_input_size = self._image_encoder.latent_size* n_cam
        self.ce_bins = cfg.cross_entropy.bins
        self.iter = 0
        self.train_mode = True
        if n_layers_xyz>0:
            if n_layers_xyz>1:
                self.mlp_xyz = MLPWithInputSkips(
                    n_layers_xyz,
                    3,
                    3,
                    cfg.ct_net.n_hidden_neurons_xyz,
                    input_skips=append_xyz,
                )
                self.mlp_cam_center = MLPWithInputSkips(
                    n_layers_xyz,
                    3,
                    3,
                    cfg.ct_net.n_hidden_neurons_dir,
                    input_skips=append_xyz,
                )
                self.decoder_input_size += cfg.ct_net.n_hidden_neurons_dir* n_cam + cfg.ct_net.n_hidden_neurons_xyz
            else:
                # insert raw coordinates
                self.mlp_xyz = MLPIdentity()
                self.mlp_cam_center = MLPIdentity()
                self.decoder_input_size += 3 * n_cam +3
        else:
            self.mlp_xyz = None
            self.mlp_cam_center = None
        # self.decoder_input_size *= n_cam
        if self.decoder_masked:
            len_mask = (cfg.ct_net.n_hidden_neurons_xyz+1) * self.num_masked
            self.decoder_input_size += len_mask

        if cfg.decoder.type == 'mlp':
            self.decoder =  LoRA_Decoder.from_cfg(cfg, self.decoder_input_size, self.ce_bins, self.use_neighbours) if cfg.decoder.apply_lora else Decoder.from_cfg(cfg, self.decoder_input_size, self.ce_bins, self.use_neighbours)
            self.decoder_type = 'mlp'
        elif cfg.decoder.type == 'transformer':
            self.decoder = DecoderTransformer.from_cfg(cfg, self.decoder_input_size, self.ce_bins)
            self.decoder_type = 'transformer'
            self.train_mode_seq_iter = cfg.transformer.training_seq_iter
        else:
            NotImplementedError()

    def batchify(self, image_features, embed_camera_center, query_points, uv, n_query):
        if self.decoder_type == 'mlp':
            return self.batchify_mlp(image_features, embed_camera_center, query_points, uv, n_query)
        elif self.decoder_type == 'transformer':
            return self.batchify_transformer(image_features, embed_camera_center, query_points, uv, n_query)
        else:
            NotImplementedError()

    def batchify_mlp(self, image_features, embed_camera_center, query_points, uv, n_query):
        assert self.decoder_type == 'mlp'
        max_n_query = self.n_query if self.train_mode else self.val_n_query
        n_chunk = int(torch.ceil(torch.tensor(n_query).sum() / max_n_query))
        uv = [torch.chunk(p, n_chunk, dim=1) for p in uv]
        query_points = torch.chunk(query_points, n_chunk) if query_points is not None else None
        pushbroom_camera = False
        if len(embed_camera_center.shape) == 4:  # pushbroom camera
            pushbroom_camera=True
            embed_camera_center = torch.chunk(embed_camera_center, n_chunk, dim=1) if embed_camera_center is not None else None

        output = [torch.empty(0, device=image_features[0].device)] * len(n_query)
        output_mask = None
        for chunk in range(n_chunk):
            uv_chunk = [p[chunk] for p in uv]
            n_split = [points.shape[1] for points in uv_chunk]
            latent_chunk = self._image_encoder.sample_roi(image_features, uv_chunk)
            latent_chunk = torch.vstack(latent_chunk).transpose(0, 1)
            latent_chunk = latent_chunk.reshape(latent_chunk.shape[0], -1)
            if query_points is not None:
                query_points_chunk = query_points[chunk]
                # query_points_chunk = query_points_chunk.unsqueeze(1).expand(-1, latent_chunk.shape[1], -1)
                latent_chunk = torch.cat((latent_chunk, query_points_chunk), -1)
                del query_points_chunk
                #with torch.cuda.device(device=image_features[0].device):
   #                 torch.cuda.empty_cache()
            if embed_camera_center is not None:
                if not pushbroom_camera:#perspective camera
                    embed_camera_center_chunk = embed_camera_center.unsqueeze(1).expand(-1,latent_chunk.shape[0], -1, -1)
                    embed_camera_center_chunk = embed_camera_center_chunk.reshape(latent_chunk.shape[0], -1)
                else:
                    embed_camera_center_chunk = embed_camera_center[chunk].reshape(latent_chunk.shape[0], -1)
                latent_chunk = torch.cat((latent_chunk, embed_camera_center_chunk), -1)

            output_chunk = self.decoder(latent_chunk)
            del latent_chunk
            # with torch.cuda.device(device=image_features[0].device):
            #     torch.cuda.empty_cache()
            if self.decoder.mask:
                if output_mask is None:
                    output_mask = [torch.empty(0, device=image_features[0].device)] * len(n_query)
                output_chunk_mask = torch.split(output_chunk[1], n_split)
                output_mask = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output_mask, output_chunk_mask)]
                output_chunk = output_chunk[0]
            output_chunk = torch.split(output_chunk, n_split)
            output = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output, output_chunk)]
        if self.decoder.mask:
            return output, output_mask
        return output

    def batchify_transformer(self, image_features, embed_camera_center, query_points, uv, n_query, z_size=32):
        assert self.decoder_type == 'transformer'
        n_query_z = int(torch.tensor(n_query).sum() / z_size)
        query_points = query_points.reshape(n_query_z,z_size,-1)
        uv = [p.reshape(-1,n_query_z,z_size,2) for p in uv]
        max_n_query = self.n_query if self.train_mode else self.val_n_query
        n_chunk = int(torch.ceil(torch.tensor(n_query_z / max_n_query)))
        uv = [torch.chunk(p, n_chunk, dim=1) for p in uv]
        query_points = torch.chunk(query_points, n_chunk) if query_points is not None else None
        output = [torch.empty(0, device=image_features[0].device)] * len(n_query)
        for chunk in range(n_chunk):
            uv_chunk = [p[chunk].reshape(p[chunk].shape[0],-1,2) for p in uv]
            n_split = [points.shape[1] for points in uv_chunk]
            latent_chunk = self._image_encoder.sample_roi(image_features, uv_chunk)
            latent_chunk = torch.vstack(latent_chunk).transpose(0, 1)
            latent_chunk = latent_chunk.reshape(latent_chunk.shape[0], -1)
            if query_points is not None:
                query_points_chunk = query_points[chunk].reshape(-1,query_points[chunk].shape[-1])
                # query_points_chunk = query_points_chunk.unsqueeze(1).expand(-1, latent_chunk.shape[1], -1)
                latent_chunk = torch.cat((latent_chunk, query_points_chunk), -1)
                del query_points_chunk
                #with torch.cuda.device(device=image_features[0].device):
   #                 torch.cuda.empty_cache()

            if embed_camera_center is not None:
                embed_camera_center_chunk = embed_camera_center.unsqueeze(1).expand(-1,
                                                                                    latent_chunk.shape[0],
                                                                                    -1, -1)
                embed_camera_center_chunk = embed_camera_center_chunk.reshape(latent_chunk.shape[0], -1)
                latent_chunk = torch.cat((latent_chunk, embed_camera_center_chunk), -1)


            latent_chunk = latent_chunk.reshape(int(latent_chunk.shape[0] / z_size), z_size, -1)
            seq_padded = torch.ones((latent_chunk.shape[0],1),device=latent_chunk.device) * 301  # 301 is bos
            output_chunk = self.decoder(latent_chunk, None, None, seq_padded, None) # Nchunk, Nz, bins(301)
            # output_chunk = torch.split(output_chunk, n_split)
            output = [torch.cat([output[0], output_chunk], dim=0)]

            # output = [torch.cat((out_i, out_chunk_i)) for out_i, out_chunk_i in zip(output, output_chunk)]
        return output

    def forward(
        self,
        cameras: PerspectiveCameras,
        image: torch.Tensor,
        volume: Volumes,
        masks: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """
        Args:
            cameras: Camera objects for geometry encoding and image feature projection & sampling.
            image: A batch of corresponding cloud images of shape
            ('batch_size', 'num_cameras', C, H, W).
            volume: A batch of corresponding ground truth 3D volumes of shape
            ('batch_size', Nx, Ny, Nz).
            masks: A batch of corresponding 3D masks of shape
            ('batch_size', Nx, Ny, Nz).
        """

        if len(image.shape)==4:
            image = image[:,:,None,...]
        Vbatch = len(volume)
        image = image.view(-1, *image.shape[2:])
        image_features = self._image_encoder(image)
        image_features = [features.view(Vbatch,self.n_cam,*features.shape[1:]) for features in image_features]

        del image
        #with torch.cuda.device(device=image_features[0].device):
          #  torch.cuda.empty_cache()

        if self.train_mode and not self.decoder_batchify:
            if self.use_neighbours:
                volume, query_points, query_indices = volume.get_query_points_and_neighbours(self.n_query, self.query_point_method, masks=masks)
            elif self.query_point_method == 'toa_random' or self.query_point_method == 'toa_all':
                volume, query_points, _ = volume.get_query_points_seq(self.n_query, self.query_point_method, masks=masks)
            else:
                volume, query_points, query_indices = volume.get_query_points(self.n_query, self.query_point_method, masks = masks)
        else:
            if self.query_point_method == 'toa_random':
                volume, query_points, _ = volume.get_query_points_seq(self.n_query, self.query_point_val_method, masks=masks)
            else:
                # volume, query_points, _ = volume.get_query_points(self.val_n_query, self.query_point_val_method, masks = masks)
                volume, query_points, query_indices = volume.get_query_points(self.val_n_query, self.query_point_val_method, masks = masks)
        n_query = [points.shape[0] for points in query_points]

        if cameras.__class__.__name__ == "AirMSPICameras":
            pushbroom_camera = True
            uv, cam_centers = cameras.project_pointsv2(query_indices, screen=True)
            # uv, cam_centers = cameras.project_pointsv2(None, screen=True)
            if self.mlp_cam_center:
                embed_camera_center = self.mlp_cam_center(cam_centers.view(-1,3),cam_centers.view(-1,3)).view(*cam_centers.shape[:-1],-1)
                embed_camera_center = embed_camera_center.transpose(1, 2)

            else:
                embed_camera_center = None

        else:
            pushbroom_camera = False
            uv = cameras.project_points(query_points, screen=True)
            if self.mlp_cam_center:
                cam_centers = cameras.get_camera_center()
                embed_camera_center = self.mlp_cam_center(cam_centers.view(-1,3),cam_centers.view(-1,3)).view(*cam_centers.shape[:-1],-1)
            else:
                embed_camera_center = None
        del cameras
        #with torch.cuda.device(device=image_features[0].device):
          #  torch.cuda.empty_cache()

        if self.mlp_xyz:
            query_points = torch.vstack(query_points).view(-1,3)
            query_points = self.mlp_xyz(query_points, query_points)
        else:
            query_points = None

        if self.train_mode and not self.decoder_batchify:
            latent = self._image_encoder.sample_roi(image_features, uv)#.transpose(1, 2)
            latent = torch.vstack(latent).transpose(0, 1)
            if query_points is not None:
                latent = latent.reshape(latent.shape[0],-1)
                latent = torch.cat((latent,query_points),-1)
                del query_points
                #with torch.cuda.device(device=image_features[0].device):
   #                 torch.cuda.empty_cache()

            if embed_camera_center is not None:
                if pushbroom_camera:
                    embed_camera_center = embed_camera_center.squeeze(0)
                    embed_camera_center = embed_camera_center.reshape(embed_camera_center.shape[0],-1)
                else:
                    embed_camera_center = embed_camera_center.reshape(embed_camera_center.shape[0],-1)
                    embed_camera_center = embed_camera_center.expand(latent.shape[0],-1)
                latent = torch.cat((latent, embed_camera_center), -1)

                del embed_camera_center
                #with torch.cuda.device(device=image_features[0].device):
   #                 torch.cuda.empty_cache()

            if self.decoder_type == 'mlp':
                output = self.decoder(latent)
            elif self.decoder_type == 'transformer':
                latent = latent.reshape(int(n_query[0]/32),32,-1)
                if self.train_mode_seq_iter<self.iter:
                    seq = volume[0].reshape(int(n_query[0]/32),32)
                    seq = torch.round(seq)
                    seq[seq>300] = 300
                    seq_padded = torch.ones_like(seq) * 301 #0-300 cloud values, 301 is bos
                    seq_padded[:, 1:] = seq[:, :-1]
                    output = self.decoder(latent,None,None,seq_padded,None)
                else:
                    seq_padded = torch.ones((latent.shape[0], 1), device=latent.device) * 301  # 301 is bos
                    output = self.decoder(latent,None,None,seq_padded,None,training_seq=True)

                output = output.reshape(-1,output.shape[-1])
            if self.decoder.mask:
                output_mask = torch.split(output[1], n_query)
                output = torch.split(output[0], n_query)
                out = {"output": output, "output_mask": output_mask, "volume": volume}
            else:
                output = torch.split(output, n_query)
                out = {"output": output, "volume": volume}
        else:
            assert Vbatch == 1
            if self.decoder.mask:
                output, output_mask = self.batchify(image_features, embed_camera_center, query_points, uv, n_query)
                out = {"output": output, "output_mask": output_mask, "volume": volume, 'query_indices': query_indices}
            else:
                output = self.batchify(image_features, embed_camera_center, query_points, uv, n_query)
                out = {"output": output, "volume": volume, 'query_indices': query_indices}
        return out


