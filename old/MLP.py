"""
Bunch of models to train for tensorflow
start with siamese MLP for testing
"""

#import tensorflow as tf
import numpy as np
import processing

data = processing.get_data()
# Reshape the string as depending on the node size in this case it's 9
M = np.fromstring(data[0][2]).reshape((9,9))

print(M)
print(np.shape(M))


"""
MODELS
"""
def model(X, Y, weight, bias):
    pass




if __name__ == "__main__":

