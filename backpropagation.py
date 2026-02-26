
x = [1.0, 2.0, 3.0]
w = [2.0, 4.0, 6.0]

b= 1.0

def RELU(x):
    return max(0,x)

def forward(x,w,b):
    z= w[0]*x[0]+ w[1]*x[1]+ w[2]*x[2]+ b
    a= RELU(z)
    return a

print("forward:", forward(x,w,b))

def backward(x,w,b):

    dvalue= 1.0 # derivative of the loss.

    # derivative of RELU and the chain rule
    drelu_dz= dvalue * (1. if forward(x,w,b)>0 else 0)

    # partial derivatives of multiplication and chain rule
    dsum_dxw0= 1
    dsum_dxw1= 1
    dsum_dxw2= 1
    dsum_db= 1

    drelu_dxw0= drelu_dz * dsum_dxw0
    drelu_dxw1= drelu_dz * dsum_dxw1
    drelu_dxw2= drelu_dz * dsum_dxw2
    drelu_db= drelu_dz * dsum_db

    print(" chain rule at sum(x*w,b) with respect to RELU:", drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

    # partial derivatives of multiplication and chain rule
    dmul_dw0= w[0]
    dmul_dw1= w[1]
    dmul_dw2= w[2]
    dmul_dx0= x[0]
    dmul_dx1= x[1]
    dmul_dx2= x[2]

    drelu_dw0= drelu_dxw0 * dmul_dw0
    drelu_dw1= drelu_dxw1 * dmul_dw1
    drelu_dw2= drelu_dxw2 * dmul_dw2
    drelu_dx0= drelu_dxw0 * dmul_dx0
    drelu_dx1= drelu_dxw1 * dmul_dx1
    drelu_dx2= drelu_dxw2 * dmul_dx2

    print(" chain rule at multiplication with respect to RELU:", drelu_dw0, drelu_dw1, drelu_dw2, drelu_dx0, drelu_dx1, drelu_dx2)

    # Hence this is how backpropagation works.

backward(x,w,b)