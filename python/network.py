import numpy as np

class Network(object):
    def __init__(self, layers, input_size):
    """
    Indicate the number of layers the network has

    input size : if matrix is n x n, input_size is n
    """
    
    self.layers = layers
    self.input_size = input_size

    # create randomized thetas for each layer
    # Thetas act as our weights
    # matrix of theta values (layer * 2n) 
    self.theta_val = np.random.randn(layers, input_size*2)
    
        
    def feed_forward(self, inputs):
        """
        inputs are matrices
        """
        

    def back_propagate(self)
