import numpy as np
from .resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        Z=np.zeros((A.shape[0],self.out_channels,A.shape[2]-self.kernel_size+1,A.shape[3]-self.kernel_size+1))
        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                Z[:,:,i,j]=np.tensordot(A[:,:,i:i+self.kernel_size,j:j+self.kernel_size],self.W,axes=([1,2,3],[1,2,3]))


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        pad_dLdZ =np.pad(dLdZ,((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1),(self.kernel_size-1,self.kernel_size-1)))
        self.dLdW = np.zeros(self.W.shape)  # TODO
        for i in range(self.dLdW.shape[2]):
            for j in range(self.dLdW.shape[3]):
                self.dLdW[:,:,i,j]=np.tensordot(dLdZ[:,:,:,:],self.A[:,:,i:i+dLdZ.shape[2],j:j+dLdZ.shape[3]],axes=([0,2,3],[0,2,3]))
        self.dLdb = np.zeros(self.b.shape)  # TODO
        for c in range(self.out_channels):
            self.dLdb[c]=np.sum(dLdZ[:,c,:,:])

        dLdA = np.zeros(self.A.shape)  # TODO
        flip_W =np.flip(self.W,axis=(2,3))
        for i in range(dLdA.shape[2]):
            for j in range(dLdA.shape[3]):
                dLdA[:,:,i,j]=np.tensordot(pad_dLdZ[:,:,i:i+self.kernel_size,j:j+self.kernel_size],flip_W[:,:,:,:],axes=((1,2,3),(0,2,3)))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels,out_channels,kernel_size,weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input appropriately using np.pad() function
        if self.pad>0:
            A=np.pad(A,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))

        # Call Conv2d_stride1
        # TODO
        A_strid =self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(A_strid) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        # TODO
        dLdZ =self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)  # TODO

        # Unpad the gradient
        # TODO
        if self.pad>0:
            dLdA=dLdA[:,:,self.pad:-self.pad,self.pad:-self.pad]
        else:
            dLdA=dLdA

        return dLdA
