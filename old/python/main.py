"""
port octave code
"""
import numpy as np


class model():

    # state vector
    s1 = np.array([[0], [1]])
    s0 = np.array([[1], [0]])

    def __init__(self, n, num_layers, theta = None, theta_import = False):
        """
        n is the n x n matirx
        num layers is the number of cascading H Rz Cz gates
        theta should be a np array 
        """
        self._n = n
        self._num_layers = num_layers
        # self.theta = theta #.npz file for theta parameters
        self._H = self.H()
        self._CZ = self.CZ()

        self.theta_list = self.random_theta(self._n, self._num_layers)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, new_n):
        if new_n > 14:
            print("n cannot be greater than 14, n is set to 14")
            self._n = new_n
        elif new_n < 3:
            print("n cannot be smaller than 3, n is set to 3")
            self._n = 3

        else:
            self._n = new_n

    def get_num_layers(self):
        return self.num_layers

    def set_num_layers(self, new_num_layers):
        self.num_layers = new_num_layers

    def random_theta(self, n, num_layers):
        output = []
        for layer in range(num_layers):
            output.append(np.random.rand(n,n))

        return output


    def CZ(self):
        """
        n wire CZ, CZ acts on n wires only to modify
        last ancilla qubit
        matrix will be in the form
        1 0 0 0 ... 0
        0 1 0 0 ... 0
        0 0 1 0 ... 0
        .     .
        .       .
        .         .
                   1 0
        0 ...      0-1


        """
        I = np.identity(2**self._n)
        I[2**self._n - 1, 2**self._n -1] -= 2
        return I


    def H(self):
        """
        Returns a H gate
        """
        output = None
        H = (1/np.sqrt(2))*(np.matrix([[1,1],[1,-1]]))

        for x in range(self._n - 1):
            output = np.kron(H, H)

        return output

    def Rz(self, theta):
        """
        Returns R(theta) gate, or rotation matrix
        """
        return np.matrix([[exp(-1j * theta /2.0), 0],[0, exp(1j * theta/ 2.0)]])
        #TODO Kron the Rz theta gate with theta (n x n) input
        # Find a more efficient method


    def run_model(self, num_epoch, save_theta = True, location = "./theta.npz"):
        #for i in xrange(num_epoch):
        #    #print loss every 1000 epoch

        # RZ kron, if there are 4 lines, each layer should have 4 theta values.
        # I need to find some way of having some list of theta value and
        # converting to some Rz matrix and then kron those matrices 

        graph = CZ * Rz1 * H * CZ
    def predict(theta):
        pass

if __name__ == "__main__":
    A = model(2,3)

    print(A._H)
    print(A.theta_list)
