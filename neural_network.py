import numpy as np

_in= [1.3, 3.2, 4.5, 7.2]

weights=[[1.2 ,5.7, 3.1, 2.4],
         [1.7 ,4.5, 3.7, 1.9],
         [4.1 ,2.2, 5.1, 2.9]]

bias=[4, 2.4, 3.4]

layer_outputs=[]

for i in range(len(weights)):
    neuron_output=np.dot(weights[i], _in)+bias[i]
    layer_outputs.append(neuron_output)

print(layer_outputs)