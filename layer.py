import numpy as np

np.random.seed(0)

X = [[1.3, 3.2, 4.5, 7.2],[-2.1, 4.3, 5.1, 6.8],[3.2, -5.1, 6.2, -2.1]]


# ReLU activation

input = [ 1.4, 2.3, -1.9, 0.5, -0.3, 0.8, 1.2, -0.7]

output=[max(i, 0) for i in input]
print(output)
print()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.1*np.random.randn( n_inputs, n_neurons)
        self.biases= np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output= np.dot(inputs, self.weights) + self.biases

layer1= Layer_Dense(4, 6)
layer2= Layer_Dense(6, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)

