import numpy as np

_in= [[1.3, 3.2, 4.5, 7.2],[-2.1, 4.3, 5.1, 6.8],[3.2, -5.1, 6.2, -2.1]]

weights1=[[1.2 ,-5.7, 3.1, 2.4],
         [-1.7 ,4.5, -3.7, 1.9],
         [4.1 ,2.2, -5.1, -2.9]]

bias1=[4, 2.4, 3.4]

# batch size of 32 is pretty common.

weights2=[[-1.2 ,5.7, 3.1],
         [1.7 ,-4.5, 3.7],
         [4.1 ,-2.2, 5.1]]

bias2=[1.4, 2.1, 1]


layer1=np.dot(_in, np.transpose(weights1)) + bias1

layer2=np.dot(layer1, np.transpose(weights2)) + bias2

print(layer2)