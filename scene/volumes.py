# This file contains the code for cloud volume object for VIP-CT and ProbCT.
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


import copy
import torch
from ProbCT.util.types import Device, make_device
from typing import List, Sequence, Tuple, Union

_TensorBatch = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]

class Volumes:
    """
    This class provides functions for working with batches of volumetric grids
    of possibly varying spatial sizes.

    VOLUME DENSITIES

    The Volumes class can be either constructed from a 5D tensor of
    `extinctions` of size `batch x property_dim x depth x height x width` or
    from a list of differently-sized 4D tensors `[D_1, ..., D_batch]`,
    where each `D_i` is of size `[property_dim x depth_i x height_i x width_i]`.

    In case the `Volumes` object is initialized from the list of `extinctions`,
    the list of tensors is internally converted to a single 5D tensor by
    zero-padding the relevant dimensions. Both list and padded representations can be
    accessed with the `Volumes.extinctions()` or `Volumes.extinctions_list()` getters.
    The sizes of the individual volumes in the structure can be retrieved
    with the `Volumes.get_grid_sizes()` getter.

    The `Volumes` class is immutable. I.e. after generating a `Volumes` object,
    one cannot change its properties, such as `self._extinctions` or `self._features`
    anymore.


    
    VOLUME COORDINATES

    Additionally, the `Volumes` class keeps track of the locations of the
    centers of the volume cells in the world coordinates.

    Coordinate tensors that denote the locations of each of the volume cells in
    world coordinates (with shape `(depth x height x width x 3)`)
    can be retrieved by calling the `Volumes.get_coord_grid()`.

    
    """

    def __init__(
        self,
        extinctions: _TensorBatch,
        grid,
        ext_thr = 0
    ) -> None:
        """
        Args:
            **extinctions**: Batch of input feature volume occupancies of shape
                `(minibatch, extinction_dim, Nx, Ny, Nz)`, or a list
                of 4D tensors `[D_1, ..., D_minibatch]` where each `D_i` has
                shape `(extinction_dim, Nx_i, Ny_i, Nz_i)`.
                Each voxel contains a non-negative number
                corresponding to its extinction.
            **grid**: pySHDOM Grid or List of Grids of the objects.
        """

        # handle extinctions
        extinctions_, grid_sizes = self._convert_extinctions_features_to_tensor(
            extinctions,  "extinctions"
        )

        # take device from extinctions
        self.device = extinctions_.device
        # if isinstance(grid, np.ndarray):
        #     assert grid.shape[0] == 3
        #     grid = [Grid(x=grid[0],y=grid[1],z=grid[2])]
        if not isinstance(grid,list):
            grid = [grid]
        # assign to the internal buffers
        self._extinctions = extinctions_
        self._grid = grid
        self._ext_thr = ext_thr

    def _convert_extinctions_features_to_tensor(
            self, x: _TensorBatch, var_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Handle the `densities` or `features` arguments to the constructor.
        """
        if isinstance(x, (list, tuple)):
            x_tensor = list_to_padded(x)
            if any(x_.ndim != 4 for x_ in x):
                raise ValueError(
                    f"`{var_name}` has to be a list of 4-dim tensors of shape: "
                    f"({var_name}_dim, height, width, depth)"
                )
            if any(x_.shape[0] != x[0].shape[0] for x_ in x):
                raise ValueError(
                    f"Each entry in the list of `{var_name}` has to have the "
                    "same number of channels (first dimension in the tensor)."
                )
            x_shapes = torch.stack(
                [
                    torch.tensor(
                        list(x_.shape[1:]), dtype=torch.long, device=x_tensor.device
                    )
                    for x_ in x
                ],
                dim=0,
            )
        elif torch.is_tensor(x):
            if x.ndim != 5:
                raise ValueError(
                    f"`{var_name}` has to be a 5-dim tensor of shape: "
                    f"(minibatch, {var_name}_dim, height, width, depth)"
                )
            x_tensor = x
            x_shapes = torch.tensor(
                list(x.shape[2:]), dtype=torch.long, device=x.device
            )[None].repeat(x.shape[0], 1)
        else:
            raise ValueError(
                f"{var_name} must be either a list or a tensor with "
                f"shape (batch_size, {var_name}_dim, H, W, D)."
            )
        return x_tensor, x_shapes

    def get_coord_grid(self) -> torch.Tensor:
        """
        Return the 3D coordinate grid of the volumetric grid
        world coordinates.


        Returns:
            **coordinate_grid**: The grid of coordinates of shape
                `(minibatch, Nx_i*Ny_i*Nz_i, 3)`.
        """
        # TODO(dnovotny): Implement caching of the coordinate grid.
        grid_list = []
        for grid in self._grid:
            grid_x, grid_y, grid_z = torch.meshgrid(torch.tensor(grid[0], device=self.device), torch.tensor(grid[1], device=self.device), torch.tensor(grid[2], device=self.device))
            grid_list.append(torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T)
        return torch.stack(grid_list)

    def get_query_points(self, n_query, method='random', masks=None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

        query_points = self.get_coord_grid()
        ext = self.extinctions.reshape(self.extinctions.shape[0], -1)
        indices = [torch.arange(ext.shape[1], device=self.extinctions.device)]

        if masks is not None:
            ext = [e[m.reshape(-1)] if m is not None else e for e, m in zip(ext, masks)]
            query_points = [points[m.reshape(-1),:] if m is not None else points for points, m in zip(query_points, masks)]
            indices = [points[m.reshape(-1)] if m is not None else points for points, m in zip(indices, masks)]
        if method == 'topk':
            indices = [torch.topk(e, n_query if n_query < e.shape[0] else e.shape[0]).indices for e in ext]
            ##TODO support for clouds with less than n_quary unmasked points
            ext = [vol[index] for vol, index in zip(ext, indices)]
            query_points = [points[index, :] for points, index in zip(query_points, indices)]
        elif method == 'random':
            indices = [torch.randperm(e.shape[0])[:n_query if n_query < e.shape[0] else e.shape[0]] for e in ext]
            ext = [vol[index] for vol, index in zip(ext, indices)]
            query_points = [points[index, :] for points, index in zip(query_points, indices)]
        elif method == 'gt_sc_random':
            NotImplementedError
            gt_masks = [e>0 for e in ext]
            masked_gt_ext = [e[m.reshape(-1)] if m is not None else e for e, m in zip(ext, gt_masks)]
            gt_n_query = int(n_query * 0.8)
            fp_n_query = n_query - gt_n_query
            masked_gt_indices = [torch.randperm(e.shape[0])[:gt_n_query if gt_n_query < e.shape[0] else e.shape[0]] for e in masked_gt_ext]
            masked_gt_ext = [vol[index] for vol, index in zip(masked_gt_ext, masked_gt_indices)]
            masked_query_points = [e[m.reshape(-1)] if m is not None else e for e, m in zip(query_points, gt_masks)]
            mask_gt_query_points = [points[index, :] for points, index in zip(masked_query_points, masked_gt_indices)]
        elif method == 'random_bins':
            ext_points = [torch.hstack((e[...,None],points)) for e, points in zip(ext,query_points)]
            ext_points = [e[torch.argsort(e[:,0])].chunk(5) for e in ext_points]
            n_query_split = int(n_query/5)
            for i, e in enumerate(ext_points):
                ind = [torch.randperm(e_split.shape[0])[:n_query_split if n_query_split < e_split.shape[0] else e_split.shape[0]] for e_split in e]
                e = torch.vstack([vol[index] for vol, index in zip(e, ind)])
                ext[i] = torch.squeeze(e[:,0])
                query_points[i] = torch.squeeze(e[:,1:])

        elif method == 'all':
            pass
        # ext = torch.stack(ext)
        # query_points = torch.stack(query_points)
        # ext = list(ext)
        # indices = None

        else:
            NotImplementedError()
        return ext, query_points, indices

    def get_query_points_seq(self, n_query, method='random', masks=None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        ext = self.extinctions.squeeze(1)
        query_points = self.get_coord_grid().reshape(*ext.shape,-1)
        indices = None
        masks_toa = [(torch.sum(m, -1) > 0) for m in masks]
        indexxy = []
        query_points2d = []
        for m, q in zip(masks_toa, query_points):
            x, y = torch.meshgrid(torch.arange(m.shape[0]), torch.arange(m.shape[1]))
            # grid_x.append(x)
            # grid_y.append(y)
            indexxy.append(torch.stack((x[m], y[m])).T)
            query_points2d.append(q[m])
        if method == 'toa_random':
            indices = [torch.randperm(g.shape[0])[:n_query if n_query < g.shape[0] else g.shape[0]] for g in indexxy]
            indexxy = [xy[ii] for xy,ii in zip(indexxy,indices)]
            # masked_gridxy = [xy[m] for xy,m in zip(indexxy,masks_toa)]
            # indices2D = [torch.arange(ext.shape[1], device=self.extinctions.device)]
            ext = [vol[index[:,0],index[:,1]].flatten() for vol, index in zip(ext, indexxy)]
            query_points2d = [q[ii].reshape(-1,3) for q,ii in zip(query_points2d,indices)]
            # xy_z = [coord.reshape((*m.shape, 3))[torch.sum(m, -1) > 0] for coord, m in zip(query_points, masks)]
        elif method == 'toa_all':
            ext = [vol[index[:, 0], index[:, 1]].flatten() for vol, index in zip(ext, indexxy)]
            query_points2d = [q.reshape(-1,3) for q in query_points2d]

        else:
            NotImplementedError()
        return ext, query_points2d, indexxy


    def get_query_points_microphysics(self, n_query, method='topk', masks=None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

        query_points = self.get_coord_grid()
        ext = self.extinctions.reshape(self.extinctions.shape[0],self.extinctions.shape[1], -1)
        # indices = None
        indices = [torch.arange(ext.shape[-1], device=self.extinctions.device)]

        if masks is not None:
            ext = [e[:,m.reshape(-1)] if m is not None else e for e, m in zip(ext, masks)]
            query_points = [points[m.reshape(-1),:] if m is not None else points for points, m in zip(query_points, masks)]
            indices = [points[m.reshape(-1)] if m is not None else points for points, m in zip(indices, masks)]


        if method == 'topk':
            indices = [torch.topk(e[0], n_query if n_query < e.shape[-1] else e.shape[-1]).indices for e in ext]
            ##TODO support for clouds with less than n_quary unmasked points
            ext = [vol[:,index] for vol, index in zip(ext, indices)]
            query_points = [points[index, :] for points, index in zip(query_points, indices)]
        elif method == 'random':
            indices = [torch.randperm(e.shape[-1])[:n_query if n_query < e.shape[-1] else e.shape[-1]] for e in ext]
            ext = [vol[:,index] for vol, index in zip(ext, indices)]
            query_points = [points[index, :] for points, index in zip(query_points, indices)]

        elif method == 'all':
            pass
            # ext = torch.stack(ext)
            # query_points = torch.stack(query_points)
            # ext = list(ext)
            # indices = None

        else:
            NotImplementedError()
        ext = [e.T for e in ext]
        return ext, query_points, indices

    def get_query_points_and_neighbours(self, n_query, method='topk', masks=None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

        query_points = self.get_coord_grid()
        ext = torch.nn.functional.pad(self.extinctions, (1,1,1,1,1,1), 'constant', 0)
        ext = ext.unfold(4, 3, 1).unfold(3, 3, 1).unfold(2, 3, 1)

        # torch.nn.functional.pad(self.extinctions, 1, 'constant', 0)
        # torch.functional.
        ext = ext.reshape(ext.shape[0], -1,3,3,3)
        indices = None
        if masks is not None:
            ext = [e[m.reshape(-1),...] if m is not None else e for e, m in zip(ext, masks)]
            query_points = [points[m.reshape(-1),:] if m is not None else points for points, m in zip(query_points, masks)]

        if method == 'topk':
            indices = [torch.topk(e[:,1,1,1], n_query if n_query < e.shape[0] else e.shape[0]).indices for e in ext]
            ext = [vol[index,...] for vol, index in zip(ext, indices)]
            query_points = [points[index, :] for points, index in zip(query_points, indices)]
        elif method == 'random':
            indices = [torch.randperm(e.shape[0])[:n_query if n_query < e.shape[0] else e.shape[0]] for e in ext]
            ext = [vol[index,...] for vol, index in zip(ext, indices)]
            query_points = [points[index, :] for points, index in zip(query_points, indices)]
        # elif method == 'random_bins':
        #     ext_points = [torch.hstack((e[...,None],points)) for e, points in zip(ext,query_points)]
        #     ext_points = [e[torch.argsort(e[:,0])].chunk(5) for e in ext_points]
        #     n_query_split = int(n_query/5)
        #     for i, e in enumerate(ext_points):
        #         ind = [torch.randperm(e_split.shape[0])[:n_query_split if n_query_split < e_split.shape[0] else e_split.shape[0]] for e_split in e]
        #         e = torch.vstack([vol[index] for vol, index in zip(e, ind)])
        #         ext[i] = torch.squeeze(e[:,0])
        #         query_points[i] = torch.squeeze(e[:,1:])

        elif method == 'all':
            # ext = torch.stack(ext)
            # query_points = torch.stack(query_points)
            # ext = list(ext)
            indices = None

        else:
            NotImplementedError()
        ext = [vol.reshape(-1,27) for vol in ext]
        return ext, query_points, indices

    def __len__(self) -> int:
        return self._extinctions.shape[0]

    @property
    def extinctions(self) -> torch.Tensor:
        """
        Returns the extinctions of the volume.

        Returns:
            **extinctions**: The tensor of volume extinctions.
        """
        return self._extinctions

    def clone(self) -> "Volumes":
        """
        Deep copy of Volumes object. All internal tensors are cloned
        individually.

        Returns:
            new Volumes object.
        """
        return copy.deepcopy(self)

    def to(self, device: Device, copy: bool = False) -> "Volumes":
        """
        Match the functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
            device: Device (as str or torch.device) for the new tensor.
            copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
            Volumes object.
        """
        device_ = make_device(device)
        if not copy and self.device == device_:
            return self

        other = self.clone()
        if self.device == device_:
            return other

        other.device = device_
        other._extinctions = self._extinctions.to(device_)
        other._grid = self._grid#.to(device_)
        return other

    def cpu(self) -> "Volumes":
        return self.to("cpu")

    def cuda(self) -> "Volumes":
        return self.to("cuda")



def list_to_padded(
    x: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    pad_size: Union[Sequence[int], None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""
    Transforms a list of N tensors each of shape (Si_0, Si_1, ... Si_D)
    into:
    - a single tensor of shape (N, pad_size(0), pad_size(1), ..., pad_size(D))
      if pad_size is provided
    - or a tensor of shape (N, max(Si_0), max(Si_1), ..., max(Si_D)) if pad_size is None.

    Args:
      x: list of Tensors
      pad_size: list(int) specifying the size of the padded tensor.
        If `None` (default), the largest size of each dimension
        is set as the `pad_size`.
      pad_value: float value to be used to fill the padded tensor
      equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)

    Returns:
      x_padded: tensor consisting of padded input tensors stored
        over the newly allocated memory.
    """
    if equisized:
        return torch.stack(x, 0)

    if not all(torch.is_tensor(y) for y in x):
        raise ValueError("All items have to be instances of a torch.Tensor.")

    # we set the common number of dimensions to the maximum
    # of the dimensionalities of the tensors in the list
    element_ndim = max(y.ndim for y in x)

    # replace empty 1D tensors with empty tensors with a correct number of dimensions
    x = [
        (y.new_zeros([0] * element_ndim) if (y.ndim == 1 and y.nelement() == 0) else y)
        for y in x
    ]

    if any(y.ndim != x[0].ndim for y in x):
        raise ValueError("All items have to have the same number of dimensions!")

    if pad_size is None:
        pad_dims = [
            max(y.shape[dim] for y in x if len(y) > 0) for dim in range(x[0].ndim)
        ]
    else:
        if any(len(pad_size) != y.ndim for y in x):
            raise ValueError("Pad size must contain target size for all dimensions.")
        pad_dims = pad_size

    N = len(x)
    x_padded = x[0].new_full((N, *pad_dims), pad_value)
    for i, y in enumerate(x):
        if len(y) > 0:
            slices = (i, *(slice(0, y.shape[dim]) for dim in range(y.ndim)))
            x_padded[slices] = y
    return x_padded


def padded_to_list(
    x: torch.Tensor,
    split_size: Union[Sequence[int], Sequence[Sequence[int]], None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, S_1, S_2, ..., S_D) into a list
    of N tensors of shape:
    - (Si_1, Si_2, ..., Si_D) where (Si_1, Si_2, ..., Si_D) is specified in split_size(i)
    - or (S_1, S_2, ..., S_D) if split_size is None
    - or (Si_1, S_2, ..., S_D) if split_size(i) is an integer.

    Args:
      x: tensor
      split_size: optional 1D or 2D list/tuple of ints defining the number of
        items for each tensor.

    Returns:
      x_list: a list of tensors sharing the memory with the input.
    """
    x_list = list(x.unbind(0))

    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][: split_size[i]]
        else:
            slices = tuple(slice(0, s) for s in split_size[i])  # pyre-ignore
            x_list[i] = x_list[i][slices]
    return x_list
