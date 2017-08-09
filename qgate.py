#import tensorflow as tf
import numpy as np

def H():
    """
    Returns a H gate
    """
    return (1/np.sqrt(2))*(np.matrix([[1,1],[1,-1]]))

def RZ(theta):
    """
    Returns R(theta) gate, or rotation matrix
    """
    return np.matrix([[exp(-1j * theta /2.0), 0],[0, exp(1j * theta/ 2.0)]])
def CZ():
    """
    Returns a CZ gate
    """
    return np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])

if __name__ == "__main__":
    # 2 qubit state
    # |01> state
    a = np.matrix([[0],[1]])
    b = np.matrix([[0],[1]])
    q = np.kron(np.transpose(a),np.transpose(b))
    print(q)
    output = np.dot(q,CZ())
    print(output)

    """
    # When multiplying the matrix use dot product
    state = np.array([[1],[0]])
    print(state.shape)
    print(state)
    output = np.dot(H(),state)
    print(output)
    print(output.shape)
    """
