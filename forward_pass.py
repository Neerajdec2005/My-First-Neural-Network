import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# X = [[1.3, 3.2, 4.5, 7.2],[-2.1, 4.3, 5.1, 6.8],[3.2, -5.1, 6.2, -2.1]]

X, y= spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.1*np.random.randn( n_inputs, n_neurons)
        self.biases= np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output= np.dot(inputs, self.weights) + self.biases

class ReLU_Activation:
    def forward(self, inputs):
        self.output= np.maximum(0, inputs)

class Softmax_Activation:
    def forward(self, inputs):
        exp_values= np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalize= exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        self.output= normalize

class Loss:
    def calculate(self, output, y):
        sample_losses=np.mean(self.forward(output, y))
        return sample_losses

class Categorical_Cross_Entropy(Loss):
    def forward(self, pred, target):
        pred_clipped= np.clip(pred, 1e-7, 1-1e-7)
        if len(target.shape)==1:
            loss=pred_clipped[range(len(pred)), target]
        elif len(target.shape)==2:
            loss=np.sum(pred_clipped*target, axis=1)
        negative_log= -np.log(loss)
        return negative_log
    
class Accuracy:
    def calculate(self, output, y):
        prediction= np.argmax(output, axis=1)
        if len(y.shape)==2:
            y= np.argmax(y, axis=1)
        accuracy= np.mean(prediction==y)
        return accuracy


dense1= Layer_Dense(2, 5)
activation= ReLU_Activation()

dense2= Layer_Dense(5, 3)
activation2= Softmax_Activation()

dense1.forward(X)
activation.forward(dense1.output)

dense2.forward(activation.output)
activation2.forward(dense2.output)

loss= Categorical_Cross_Entropy()

accuracy= Accuracy()

print(accuracy.calculate(activation2.output, y))

