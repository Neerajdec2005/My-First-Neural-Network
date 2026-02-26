import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases= np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs= inputs
        self.output= np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights= np.dot(self.inputs.T, dvalues)
        self.dbiases= np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs= np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs= inputs
        self.output= np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs= dvalues.copy()
        self.dinputs[self.inputs<=0]= 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs= inputs
        exp_values= np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities= exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        self.output= probabilities

    def backward(self, dvalues):
        self.dinputs= dvalues.copy()
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output= single_output.reshape(-1,1)
            jacobian_matrix= np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index]= np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses= self.forward(output, y)
        data_loss= np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples= len(y_pred)
        y_pred_clipped= np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape)==1:
            correct_confidences= y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape)==2:
            correct_confidences= np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods= -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples= len(dvalues)
        labels= len(dvalues[0])

        if len(y_true.shape)==1:
            y_true= np.eye(labels)[y_true]

        self.dinputs= -y_true/dvalues
        self.dinputs= self.dinputs/samples

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation= Activation_Softmax()
        self.loss= Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output= self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples= len(dvalues)

        if len(y_true.shape)==2:
            y_true= np.argmax(y_true, axis=1)

        self.dinputs= dvalues.copy()
        self.dinputs[range(samples), y_true]-= 1
        self.dinputs= self.dinputs/samples

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, eps=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay= decay
        self.iterations= 0
        self.eps=eps
        self.beta1= beta1
        self.beta2= beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate= self.learning_rate * (1.0/(1.0 + self.decay*self.iterations))

    def update_params(self, layer):
        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums= np.zeros_like(layer.weights)
            layer.bias_momentums= np.zeros_like(layer.biases)
            layer.weight_cache= np.zeros_like(layer.weights)
            layer.bias_cache= np.zeros_like(layer.biases)
        layer.weight_momentums= self.beta1* layer.weight_momentums + (1-self.beta1)* layer.dweights
        layer.bias_momentums= self.beta1* layer.bias_momentums + (1-self.beta1)* layer.dbiases

        weight_momentums_corrected= layer.weight_momentums/(1-self.beta1**(self.iterations+1))
        bias_momentums_corrected= layer.bias_momentums/(1-self.beta1**(self.iterations+1))

        layer.weight_cache= self.beta2* layer.weight_cache + (1-self.beta2)* layer.dweights**2
        layer.bias_cache= self.beta2* layer.bias_cache + (1-self.beta2)* layer.dbiases**2

        weight_cache_corrected= layer.weight_cache/(1-self.beta2**(self.iterations+1))
        bias_cache_corrected= layer.bias_cache/(1-self.beta2**(self.iterations+1))

        layer.weights+= -self.current_learning_rate * weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+self.eps)
        layer.biases+= -self.current_learning_rate * bias_momentums_corrected/(np.sqrt(bias_cache_corrected)+self.eps)
    def post_update_params(self):
        self.iterations+= 1


X, y= spiral_data(samples=100, classes=3)

# Forward Pass
dense1= Layer_Dense(2,64)
activation1= Activation_ReLU()
dense2= Layer_Dense(64,3)
loss_activation= Activation_Softmax_Loss_CategoricalCrossentropy()
optimizers= Optimizer_Adam(learning_rate=0.02,decay=1e-5)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss= loss_activation.forward(dense2.output, y)

    predictions= np.argmax(loss_activation.output, axis=1)
    if len(y.shape)==2:
        y= np.argmax(y, axis=1)
    accuracy= np.mean(predictions==y)

    if epoch%1000==0:
        print(f"epoch: {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizers.pre_update_params()
    optimizers.update_params(dense1)
    optimizers.update_params(dense2)
    optimizers.post_update_params()
