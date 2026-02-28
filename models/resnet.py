"""
ResNet built with mytorch (Conv2d, BatchNorm2d, ReLU).
Run from repo root: PYTHONPATH=. python -c "from models.resnet import ResBlock; ..."
"""
import sys
import os
import numpy as np

_reporoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _reporoot not in sys.path:
    sys.path.insert(0, _reporoot)

from mytorch.nn.Conv2d import Conv2d
from mytorch.nn.activation import ReLU
from mytorch.nn.batchnorm2d import BatchNorm2d


class Conv_BN(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.layers = [
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            BatchNorm2d(out_channels),
        ]

    def forward(self, A):
        self.A = A
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, grad):
        dZ = grad
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ


class ResBlock(object):
    def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
        self.ConvBlock = [
            Conv_BN(in_channels, out_channels, filter_size, stride, padding),
            ReLU(),
            Conv_BN(out_channels, out_channels, 1, 1, 0),
        ]
        self.final_activation = ReLU()

        if stride != 1 or in_channels != out_channels or filter_size != 1 or padding != 0:
            self.residual_connection = Conv_BN(
                in_channels, out_channels, filter_size, stride, padding
            )
        else:
            self.residual_connection = None

    def forward(self, A):
        Z = A
        for layer in self.ConvBlock:
            Z = layer.forward(Z)
        self.Z_main = Z

        if self.residual_connection is None:
            Z_res = A
        else:
            Z_res = self.residual_connection.forward(A)
        self.Z_res = Z_res

        out = self.Z_main + self.Z_res
        out = self.final_activation.forward(out)
        return out

    def backward(self, grad):
        dZ = self.final_activation.backward(grad)
        grad_main = dZ
        grad_res = dZ

        if self.residual_connection is None:
            residual_grad = grad_res
        else:
            residual_grad = self.residual_connection.backward(grad_res)

        convlayers_grad = grad_main
        for layer in reversed(self.ConvBlock):
            convlayers_grad = layer.backward(convlayers_grad)

        return convlayers_grad + residual_grad
