import numpy as np

dvalues=np.array([[1., 1., 1.],
                  [2., 2., 2.],
                  [3., 3., 3.]])

x = np.array([[1., -2., 3.5],
              [-4., 5., -6.2],
              [7.1, -8., 9.]])
w = np.array([[2., 3., 4.],
              [0.5, 1., 1.5],
              [1., 2., 3.]]).T

b= np.array([[3., 4., 5.]])

layer_outputs= np.dot(x,w)+b

relu_outputs= np.maximum(0, layer_outputs)

drelu= relu_outputs.copy()
drelu[layer_outputs<=0]= 0

dinputs= np.dot(drelu, w.T)

dweights= np.dot(x.T, drelu)

dbiases= np.sum(drelu, axis=0, keepdims=True)

w+= -0.001 * dweights
b+= -0.001 * dbiases

print("updated weights:", w)
print("updated biases:", b)


