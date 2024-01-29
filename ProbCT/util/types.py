# This file contains util routines for VIP-CT.
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

from typing import Optional, Union

import torch


Device = Union[str, torch.device]


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:  # pyre-ignore[16]
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def get_device(x, device: Optional[Device] = None) -> torch.device:
    """
    Gets the device of the specified variable x if it is a tensor, or
    falls back to a default CPU device otherwise. Allows overriding by
    providing an explicit device.

    Args:
        x: a torch.Tensor to get the device from or another type
        device: Device (as str or torch.device) to fall back to

    Returns:
        A matching torch.device object
    """

    # User overrides device
    if device is not None:
        return make_device(device)

    # Set device based on input tensor
    if torch.is_tensor(x):
        return x.device

    # Default device is cpu
    return torch.device("cpu")

