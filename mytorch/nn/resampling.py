import numpy as np



class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        input_width = A.shape[-1]
        output_width = int(self.upsampling_factor * (input_width-1)+1)


        Z = np.zeros((A.shape[0],A.shape[1],output_width))# TODO
        Z[0:A.shape[0], 0:A.shape[1],0:output_width:self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        output_width = dLdZ.shape[-1]
        input_width= int((output_width-1)/self.upsampling_factor+1)

        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],input_width))
        dLdA[:,:,:] = dLdZ[:,:,0:output_width:self.upsampling_factor]  # TODO
        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        input_width = A.shape[-1]
        self.win1=input_width
        output_width=int(((input_width - 1) / self.downsampling_factor + 1))
        Z = np.zeros((A.shape[0],A.shape[1],output_width))

        Z[:,:,:] = A[:,:,0:input_width:self.downsampling_factor]  # TODO
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],self.win1))# TODO
        dLdA[:,:,::self.downsampling_factor] = dLdZ
        return dLdA


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        input_height=A.shape[-2]
        output_height=int(self.upsampling_factor * (input_height- 1) + 1)
        input_width = A.shape[-1]
        output_width = int(self.upsampling_factor * (input_width - 1) + 1)
        Z = np.zeros((A.shape[0],A.shape[1],output_height,output_width)) # TODO
        Z[:,:,0:output_height:self.upsampling_factor,0:output_width:self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        output_height = dLdZ.shape[-2]
        input_height =int((output_height - 1) / self.upsampling_factor + 1)
        output_width = dLdZ.shape[-1]
        input_width=int((output_width-1)/self.upsampling_factor+1)

        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],input_height,input_width))
        dLdA[:,:,:,:]=dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]
        # TODO
        return dLdA



class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        input_height = A.shape[-2]
        self.win2h=input_height
        output_height = int((input_height - 1) / self.downsampling_factor + 1)
        input_width = A.shape[-1]
        self.win2w=input_width
        output_width = int((input_width - 1) / self.downsampling_factor + 1)
        Z = np.zeros((A.shape[0], A.shape[1], output_height,output_width))


        Z[:,:,:,:]=A[:,:,::self.downsampling_factor,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1],self.win2w, self.win2h))  # TODO
        dLdA[:, :, ::self.downsampling_factor,::self.downsampling_factor] = dLdZ
        return dLdA

