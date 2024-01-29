# This file contains legacy code of VIP-CT framework and not in use.
# It is based on PyTorch3D source code ('https://github.com/facebookresearch/pytorch3d') by FAIR
# Copyright (c) Facebook, Inc. and its affiliates.
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
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import torch
from .harmonic_embedding import HarmonicEmbedding


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        self.embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        self.embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

    def forward(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function embeds the 3D points and rays normalized directions
        sampled along projection rays in camera coordinate system.


        Args:
            points: A tensor of shape `(minibatch, ..., 3)` denoting the
                    3D points of the sampled rays in camera coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the normalized direction vectors of the sampled rays in camera coords.

        Returns:
            embeds_xyz: A tensor of shape `(minibatch x ... x self.n_harmonic_functions_dir*6 + 3)`
                represents the 3D points embedding.
            embeds_dir: A tensor of shape `(minibatch x ... x self.n_harmonic_functions_xyz*6 + 3)`
                represents the normalized directions embedding.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        # rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]
        directions = self.harmonic_embedding_dir(directions)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions_dir*6 + 3]
        # For each 3D world coordinate, we obtain its harmonic embedding.
        points = self.harmonic_embedding_xyz(points)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions_xyz*6 + 3]

        return points, directions

