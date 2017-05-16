import tensorflow as tf
import numpy as np

def H():
    """
    Returns a H gate
    """
    return (1/np.sqrt(2))*(np.matrix([[1,1],[1,-1]]))

def R(theta):
    """
    Returns R(theta) gate, or rotation matrix
    """
    return np.matrix([[1,0],[0,np.exp(2*np.pi*1j*theta)]])

if __name__ == "__main__":
    # When multiplying the matrix use dot product
    state = np.array([[1],[0]])
    print(state.shape)
    print(state)
    output = np.dot(H(),state)
    print(output)
    print(output.shape)
