import math
import numpy as np

E= math.e

layer_outputs= [1.4, -2.15, -4, 3]

exp=np.exp(layer_outputs)

print(exp)
print()

norm_base= sum(exp)

softmax= exp/ norm_base
print(softmax)
print()
print(sum(softmax))

