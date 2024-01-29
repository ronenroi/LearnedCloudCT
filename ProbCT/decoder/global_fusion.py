import torch
from torch import nn

class Global_fusion(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            in_channels,
            kernel_size=5,
            latent_size=64,

    ):
        """
        :param type Decoder network type.
        """
        super().__init__()
        self.model = nn.Sequential(
            torch.nn.Conv3d(in_channels,latent_size, kernel_size, padding=int((kernel_size - 1) / 2)),
            nn.ReLU(),
            torch.nn.Conv3d(latent_size, latent_size, kernel_size, padding=int((kernel_size - 1) / 2)),
            nn.ReLU(),
            torch.nn.Conv3d(latent_size, latent_size, kernel_size, padding=int((kernel_size - 1) / 2)),
            nn.ReLU(),
            torch.nn.Conv3d(latent_size, latent_size, kernel_size, padding=int((kernel_size - 1) / 2)),
            nn.ReLU(),
            torch.nn.Conv3d(latent_size, 1, 1)
        )
        # self.in_conv =
        # self.convs = nn.ModuleList([ for _ in range(3)])
        # self.relu = torch.nn.ReLU(True)
        # self.out = torch.nn.Conv3d(latent_size, 1, 1)



    def forward(self, x):
        return self.model(x)

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            in_channels=cfg.global_fusion.in_channels,
            kernel_size=cfg.global_fusion.kernel_size,
            latent_size=cfg.global_fusion.latent_size,
        )
