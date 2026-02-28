# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

_reporoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _reporoot not in sys.path:
    sys.path.insert(0, _reporoot)

from mytorch.flatten import Flatten
from mytorch.nn.Conv1d import Conv1d
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU, Identity, Sigmoid, Tanh, GELU, Swish, Softmax
from mytorch.nn.loss import SoftMaxCrossEntropy


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24,8,8,4)
        self.conv2 = Conv1d(8,16,1,1)
        self.conv3 = Conv1d(16, 4, 1, 1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten()] # TODO: Add the layers in the correct order

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        
        # TODO: For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)

        w1_transposed = w1.T
        w1_reshaped = np.reshape(w1_transposed, (8, 8, 24))
        w1_final = np.transpose(w1_reshaped, (0, 2, 1))
        self.conv1.conv1d_stride1.W = w1_final

        w2_transposed = w2.T  # (8, 16)
        w2_reshaped = np.reshape(w2_transposed, (16, 1, 8))
        w2_final = np.transpose(w2_reshaped, (0, 2, 1))
        self.conv2.conv1d_stride1.W = w2_final


        # 第三层权重转换

        w3_transposed = w3.T  # (16, 4)
        w3_reshaped = np.reshape(w3_transposed, (4, 1, 16))
        w3_final = np.transpose(w3_reshaped, (0, 2, 1))
        self.conv3.conv1d_stride1.W = w3_final


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method
        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA

class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten()] # TODO: Add the layers in the correct order

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        # TODO: For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)
        #   4 : Slice the weight matrix and reduce to only the shared weights
        #   (hint: be careful, steps 1-3 are similar, but not exactly like in the simple scanning MLP)

        def reshape_weights(w, in_channels, out_channels, kernel_size):
            w = w.transpose(1, 0)  # transpose
            w = w.reshape(out_channels, kernel_size, in_channels)  # (Cout, k, Cin)
            w = w.transpose(0, 2, 1)  # (Cout, Cin, k)
            return w
        w1 = reshape_weights(w1, 24, 8, 8)
        w2 = reshape_weights(w2, 2, 16, 4)
        w3 = reshape_weights(w3, 8, 4, 2)

        self.conv1.conv1d_stride1.W = w1[:2, :24, :2]
        self.conv2.conv1d_stride1.W = w2[:8, :2, :2]
        self.conv3.conv1d_stride1.W = w3[:4, :8, :2]

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA