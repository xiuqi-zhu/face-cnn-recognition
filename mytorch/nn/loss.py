import numpy as np


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]  # TODO
        self.C = self.A.shape[1]  # TODO
        se = (A-Y)*(A-Y)  # TODO
        n1=np.ones((self.N,self.C))
        n1_transpose=np.transpose(n1)

        c1=np.ones((self.C,self.C))
        sse =n1_transpose@se@c1   # TODO
        mse = sse/(self.N*self.C) # TODO
        return mse  # TODO - What should be the return value?

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        dLdA = 2*((self.A-self.Y)/(self.N*self.C))
        return dLdA  # TODO - What should be the return value?


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        Hint: Read the writeup to determine the shapes of all the variables.
        Note: Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]  # TODO
        self.C = self.A.shape[1]  # TODO

        Ones_C = np.ones((self.C,self.C))  # TODO
        Ones_N = np.ones((self.N,self.C))  # TODO
        Ones_N_transpose=np.transpose(Ones_N)
        A_max = np.max(A, axis=1, keepdims=True)

        self.softmax = np.exp(self.A-A_max)/np.sum(np.exp(self.A-A_max),axis=1,keepdims=True)  # TODO - Can you reuse your own softmax here, if not rewrite the softmax forward logic?

        crossentropy = (-self.Y*(np.log(self.softmax)))@Ones_C  # TODO
        sum_crossentropy_loss = Ones_N_transpose@crossentropy  # TODO
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        return mean_crossentropy_loss  # TODO - What should be the return value?

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        """
        dLdA = (self.softmax-self.Y)/self.N  # TODO
        return dLdA  # TODO - What should be the return value?
