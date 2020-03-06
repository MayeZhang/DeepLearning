#%%
import sys, os 
sys.path.append(os.pardir)
import numpy as np 
from common.functions import *
from common.gradient import numerical_gradient

# %%
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.W1 = weight_init_std * np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = weight_init_std * np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros(output_size)
    
    def prect(self, x):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        z1 = np.dot(x, W1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        y = softmax(a2)

        return y

    

    