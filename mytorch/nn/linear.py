import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug
        self.Cin = in_features
        self.Cout = out_features
        self.W = np.zeros((out_features,in_features))  # TODO
        self.b = np.zeros((self.Cout,1)) # TODO

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.A = A   # TODO
        self.N = A.shape[0]  # TODO - store the batch size parameter of the input A

        # Think how can `self.ones` help in the calculations and uncomment below code snippet.
        # self.ones = np.ones((self.N, 1))
        self.ones=np.ones((self.N,1))
        W_transpose=np.transpose(self.W)
        b_transpose=np.transpose(self.b)

        self.Z = self.A @ W_transpose+self.ones @ b_transpose  # TODO

        return self.Z # TODO - What should be the return value?

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        self.dLdZ=dLdZ
        dLdZ_transpose=np.transpose(self.dLdZ)
        dLdA = dLdZ@self.W  # TODO
        self.dLdW = dLdZ_transpose@self.A  # TODO
        self.dLdb = dLdZ_transpose@self.ones  # TODO

        if self.debug:
            self.dLdA = dLdA


        return dLdA
