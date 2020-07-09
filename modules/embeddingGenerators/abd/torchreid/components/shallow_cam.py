import torch.nn as nn

from .attention import CAM_Module

__all__ = ['ShallowCAM']


class ShallowCAM(nn.Module):

    def __init__(self, feature_dim: int):

        super().__init__()
        self.input_feature_dim = feature_dim

        use = True

        if use:
            self._cam_module = cam_module = CAM_Module(self.input_feature_dim)

            # if False:
            #    self._cam_module_abc = cam_module  # Forward Compatibility
        else:
            self._cam_module = None

    def forward(self, x):

        if self._cam_module is not None:
            x = self._cam_module(x)

        return x
