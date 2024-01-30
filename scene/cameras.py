# This file contains the code for camera object for VIP-CT and ProbCT.
# It is based on PyTorch3D source code ('https://github.com/facebookresearch/pytorch3d') by FAIR
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# authors. The authors are not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from ProbCT.util.types import Device
from ProbCT.util.renderer_utils import TensorProperties


# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)
norm = lambda x: x / torch.linalg.norm(x, axis=0)


class PerspectiveCameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    def __init__(
            self,
            image_size: Union[List, Tuple, torch.Tensor],
            focal_length=1.0,
            camera_center: Optional[torch.Tensor] = None,
            R: Optional[torch.Tensor] = _R,
            T: Optional[torch.Tensor] = _T,
            K: Optional[torch.Tensor] = None,
            P: Optional[torch.Tensor] = None,
            device: Device = "cpu"
    ):
        """

        Args:
            image_size: (height, width) of image size.
                A tensor of shape (B, N, 2) or a list/tuple.
            focal_length: Focal length of the camera in world units.
                A tensor of shape (B, N, 1) or (B, N, 2) for
                square and non-square pixels respectively.
            R: Rotation matrix of shape (B, N, 3, 3)
            T: Translation matrix of shape (B, N, 3)
            K: (optional) camera intrinsic matrix of shape (B, N, 3, 3)
                If provided, don't need focal_length
            P: (optional) camera projection matrix of shape (B, N, 3, 4)
                If provided, don't need Rotation and Translation matrices

            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.

        if K is None:
            K = torch.eye(3, device=device)[None]
            K[:, 0, 0] = focal_length
            K[:, 1, 1] = focal_length
        else:
            K = K.to(device=device)
        if P is None:
            G = torch.hstack([R.T, -R.T @ T.T]).to(device=device)
            P = K @ G
        else:
            if not isinstance(P, torch.Tensor):
                P = torch.tensor(P)
            P = P.to(device=device)
            if len(P.shape) == 2:
                P= P[None, None]
            elif len(P.shape) == 3:
                P= P[None]
            R = P[...,:3]
            T = torch.squeeze(P[..., 3:],dim=-1)
        super().__init__(
            device=device,
            focal_length=focal_length,
            R=R,
            T=T,
            K=K,
            P=P,
            image_size=image_size
        )
        self.camera_center = camera_center
        self.index = torch.arange(self._N)
        if (self.image_size < 1).any():  # pyre-ignore
            raise ValueError("Image_size provided has invalid values")


    def get_camera_center(self) -> torch.Tensor:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        # the camera center is the translation component (the last column) of the transform matrix P (3x4 RT matrix)
        return self.camera_center


    def _transform_points_old(self, points, eps: Optional[float] = None) -> List[torch.Tensor]:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: List of Tensor of shape (Pi, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, num_cams, P, 2) or (num_cams,P, 2) depending
            on the dimensions of the transform
        """
        points_batch = []
        for p in points:
            ones = torch.ones(p.shape[0], 1, dtype=p.dtype, device=p.device)
            points_batch.append(torch.cat([p, ones], dim=1))

        composed_matrices = self.P
        if composed_matrices.dim() == 3:
            composed_matrices = composed_matrices[None]
        composed_matrices = composed_matrices.transpose(-2, -1)
        points_out = [_broadcast_bmm(points, composed_matrix) for points, composed_matrix in zip(points_batch, composed_matrices)]
        denom = [out[..., 2:] for out in points_out]  # denominator
        if eps is not None:
            for i, d in enumerate(denom):
                d_sign = d.sign() + (d == 0.0).type_as(d)
                denom[i] = d_sign * torch.clamp(d.abs(), eps)
        points_out = [out[..., :2] / d for out, d in zip(points_out, denom)]

        return points_out


    def _transform_points(self, points, eps: Optional[float] = None) -> List[torch.Tensor]:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: List of Tensor of shape (Pi, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, num_cams, P, 2) or (num_cams,P, 2) depending
            on the dimensions of the transform
        """
        points_batch = []
        for p in points:
            ones = torch.ones(p.shape[0], 1, dtype=p.dtype, device=p.device)
            points_batch.append(torch.cat([p, ones], dim=1))

        composed_matrices = self.P
        if composed_matrices.dim() == 3:
            composed_matrices = composed_matrices[None]
        composed_matrices = composed_matrices.transpose(-2, -1)
        points_out = [_broadcast_bmm(points, composed_matrix) for points, composed_matrix in zip(points_batch, composed_matrices)]
        denom = [out[..., 2:] for out in points_out]  # denominator
        if eps is not None:
            for i, d in enumerate(denom):
                d_sign = d.sign() + (d == 0.0).type_as(d)
                denom[i] = d_sign * torch.clamp(d.abs(), eps)
        points_out = [out[..., [1,0]] / d for out, d in zip(points_out, denom)]

        return points_out



    def project_points(
        self, points, eps: Optional[float] = None, screen: bool = False
    ) -> torch.Tensor:
        """
        Transforms points from world space to screen space.
        Input points follow the SHDOM coordinate system conventions: +X[km] (North), +Y [km] (East), +Z[km] (Up).
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: list of torch tensors, each of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            screen: if True returns the projected points in pixels
                    else in ndc space: X, Y in [-1 1]. For non square
                    images, we scale the points such that largest side
                    has range [-1, 1] and the smallest side has range
                    [-u, u], with u < 1.

        Returns
            new_points: transformed points with the same shape as the input
            except the last axis size is 2.
        """
        points_out = self._transform_points(points, eps=eps)
        if screen:
            points_out = self._to_screen_transform(points_out)
        self.index = torch.arange(self._N)
        return points_out

    def _to_screen_transform(self, points):
        image_size = torch.unsqueeze(self.image_size, 2)
        for i, p in enumerate(points):
            if p.shape[-1]==3:
                p = p[...,:2]
            points[i] = p * (image_size[i] / 2) + (image_size[i] / 2)
        return points

    def clone(self):
        """
        Returns a copy of `self`.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)


class AirMSPICameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    def __init__(
            self,
            mapping: torch.tensor,
            centers: torch.tensor=None,
            device: Device = "cpu"
    ):

        super().__init__(
            device=device,
        )
        self.mapping = mapping.to(device=self.device)
        self.centers = centers.to(device=self.device)

    def project_points(
        self, points, eps: Optional[float] = None, screen: bool = False
    ) -> torch.Tensor:
        """
        Transforms points from world space to screen space.
        Input points follow the SHDOM coordinate system conventions: +X[km] (North), +Y [km] (East), +Z[km] (Up).
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: list of torch tensors, each of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            screen: if True returns the projected points in pixels
                    else in ndc space: X, Y in [-1 1]. For non square
                    images, we scale the points such that largest side
                    has range [-1, 1] and the smallest side has range
                    [-u, u], with u < 1.

        Returns
            new_points: transformed points with the same shape as the input
            except the last axis size is 2.
        """
        if points is not None:
            points_out = [map[:,p,:] for map, p in zip([self.mapping], points)]
        else:
            points_out = [self.mapping]

        return points_out

    def project_pointsv2(
        self, points, eps: Optional[float] = None, screen: bool = False
    ) -> (torch.Tensor,torch.Tensor):
        """
        Transforms points from world space to screen space.
        Input points follow the SHDOM coordinate system conventions: +X[km] (North), +Y [km] (East), +Z[km] (Up).
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: list of torch tensors, each of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            screen: if True returns the projected points in pixels
                    else in ndc space: X, Y in [-1 1]. For non square
                    images, we scale the points such that largest side
                    has range [-1, 1] and the smallest side has range
                    [-u, u], with u < 1.

        Returns
            new_points: transformed points with the same shape as the input
            except the last axis size is 2.
        """
        if points is not None:
            assert len(points)==1
            points_out = self.mapping[:,:,points[0],:].to(device=self.device)
            pixel_centers = self.centers[:,:,points[0],:].to(device=self.device)

        else:
            points_out = self.mapping.to(device=self.device)
            pixel_centers = self.centers.to(device=self.device)

        return points_out, pixel_centers

    def clone(self):
        """
        Returns a copy of `self`.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)


def _broadcast_bmm(a, b):
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multiplied. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if b.dim() == 2:
        b = b[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    from ProbCT.scene.volumes import Volumes


    #test camera projection
    def sample_features(latents, uv):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param latent (B, C, H, W) images features
        :param uv (B, N, 2) image points (x,y)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :return (B, C, N) L is latent size
        """
        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = torch.empty(0, device=uv.device)
        for latent in latents:
            samples = torch.cat((samples, torch.squeeze(F.grid_sample(
                latent,
                uv,
                align_corners=True,
                mode='bilinear',
                padding_mode='zeros',
            ))), dim=1)
        return samples  # (Cams,cum_channels, N)

    if False:
        with open('/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/train/cloud_results_0.pkl', 'rb') as outfile:
            x = pickle.load(outfile)
        image_sizes = np.array([image.shape for image in x['images']])
        cameras = PerspectiveCameras(image_sizes, P=x['cameras_P'])
        camera_position_list = x['cameras_pos']
        layers = 4
        images = [torch.arange(int(128/(i+1))**2).reshape(1,1,int(128/(i+1)),-1).double().repeat(image_sizes.shape[0],1,1,1) for i in range(layers)]
        indices = torch.topk(torch.tensor(x['ext']).reshape(-1), 10).indices
        print(torch.tensor(x['ext']).reshape(-1)[indices])
        grid = x['grid']
        volume = Volumes(torch.tensor(x['ext'])[None, None].double(), grid)
        projected_to_world_points = volume.get_coord_grid()[0][indices].double()
        projected_to_screen = cameras.project_points(projected_to_world_points, screen=True)
        projected_to_ndc = cameras.project_points(projected_to_world_points, screen=False)
        for im, screen_points in zip(x['images'], projected_to_screen):
            plt.imshow(im / np.max(im))
            plt.scatter(screen_points[:, 1].cpu().numpy(), screen_points[:, 0].cpu().numpy(), s=1, c='red',
                        marker='x')
            plt.show()
        A = sample_features(images, projected_to_ndc.double())
        print(sample_features(images, projected_to_ndc.double()))
        print()
    else:
        from .cameras import AirMSPICameras
        DEFAULT_DATA_ROOT = '/home/roironen/Data'
        data_root = '/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras/train'
        image_root = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/pushbroom/ROI/AIRMSPI_IMAGES_LWC_LOW_SC/'
        mapping_path = '/wdata/roironen/Data/voxel_pixel_list32x32x32_BOMEX_img350x350.pkl'
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        images_mapping_list = []
        for _, map in mapping.items():
            voxels_list = []
            v = map.values()
            voxels = np.array(list(v), dtype=object)
            ctr = 0
            for i, voxel in enumerate(voxels):
                if len(voxel) > 0:
                    pixels = np.unravel_index(voxel, np.array([350,350]))
                    mean_px = np.mean(pixels, 1)
                    voxels_list.append(mean_px)
                else:
                    ctr += 1
                    voxels_list.append([-100000, -100000])
            images_mapping_list.append(voxels_list)
        with open(image_root, 'rb') as f:
            images = pickle.load(f)['images']
        image_sizes = np.array([[350,350]]*9)
        device = 'cuda'
        cameras = AirMSPICameras(image_size=torch.tensor(image_sizes),mapping=torch.tensor(mapping, device=device).float(),
                                         device=device)
        with open(data_root, 'rb') as f:
            x = pickle.load(f)
        layers = 4
        indices = torch.topk(torch.tensor(x['ext']).reshape(-1), 10).indices
        print(torch.tensor(x['ext']).reshape(-1)[indices])
        grid = x['grid']
        volume = Volumes(torch.tensor(x['ext'])[None, None].double(), grid)
        projected_to_screen = cameras.project_points(indices, screen=True)
        for im, screen_points in zip(x['images'], projected_to_screen):
            plt.imshow(im / np.max(im))
            plt.scatter(screen_points[:, 1].cpu().numpy(), screen_points[:, 0].cpu().numpy(), s=1, c='red',
                        marker='x')
            plt.show()
        A = sample_features(images, projected_to_screen.double())
        print(sample_features(images, projected_to_screen.double()))
        print()


