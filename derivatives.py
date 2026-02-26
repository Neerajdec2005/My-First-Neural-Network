import numpy as np
import matplotlib.pyplot as plt

# Rise over run

def f(x):
    return 2*x**2

x= np.arange(0,5,0.0001)
y= f(x)

print(x)
print(y)

# Good approximation using numerical differentiation

delta= 0.0001

x1= 3
x2= x1+delta

y1= f(x1)
y2= f(x2)

approximated_derivative= (y2-y1)/(x2-x1)

print("approximated_derivative:",approximated_derivative)

# y=mx+b

b= y2- approximated_derivative*x2

print("b:", b)

def tangent_line(x):
    return approximated_derivative*x + b
plot_x= [x1-0.9, x1, x1+0.9]

plt.plot(x,y, label="f(x)")
plt.plot(plot_x, [tangent_line(x) for x in plot_x], label="tangent line at x=3")
# plt.legend()
plt.show()
# this method is numerical derivative, also called universal derivative.