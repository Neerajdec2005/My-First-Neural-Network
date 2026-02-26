# Plot decision boundary
import matplotlib.pyplot as plt

h = 0.01
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
dense1.forward(grid)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation_softmax = Activation_Softmax()
activation_softmax.forward(dense2.output)
Z = np.argmax(activation_softmax.output, axis=1).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='brg')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', edgecolors='k', s=20)
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()