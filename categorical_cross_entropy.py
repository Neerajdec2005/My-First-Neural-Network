import numpy as np

X= np.array([[1.3, 3.2, 4.5, 7.2],[2.1, 4.3, 5.1, 6.8],[3.2, 5.1, 6.2, -2.1]])

# Using one hot encoding for the labels
targets= [2, 0, 1]

print(np.mean(-np.log(X[[0, 1, 2], [targets]])))

# the only issue is log(0) is undefined, so we should clip the values.
