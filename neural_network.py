import numpy as np

from utils import Layer
from initilisation_functions import xavier_init
from activation_functions import ReluLayer, SigmoidLayer


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        # Initialize the weights with Xavier initialization
        self._W = xavier_init((n_in, n_out))
        # Initialize the biases as a 1D array of zeros
        self._b = np.zeros(n_out)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        self._cache_current = x
        return x.dot(self._W) + self._b

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """

        x = self._cache_current
        # Calculate gradients for weights and biases
        self._grad_W_current = x.T.dot(grad_z)
        self._grad_b_current = np.sum(grad_z, axis=0)
        # Calculate gradient with respect to the input of the layer
        grad_x = grad_z.dot(self._W.T)
        return grad_x
    

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        # Compute the updates for weights and biases
        weight_update = learning_rate * self._grad_W_current
        bias_update = learning_rate * self._grad_b_current

        # Apply the updates to the weights and biases
        self._W = self._W - weight_update
        self._b = self._b - bias_update



class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer
                represented as a list. The length of the list determines the
                number of linear layers.
            - activations {list} -- List of the activation functions to apply
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations
        self._layers = []
        layer_input_dim = input_dim

        for i, (neuron_count, activation_func) in enumerate(zip(neurons, activations)):
            self._layers.append(LinearLayer(layer_input_dim, neuron_count))
            if activation_func == "relu":
                self._layers.append(ReluLayer())
            elif activation_func == "sigmoid":
                self._layers.append(SigmoidLayer())
            layer_input_dim = neuron_count

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        
        # Collect gradients for each layer in a list
        gradients = []
        for layer in reversed(self._layers):
            grad_z = layer.backward(grad_z)
            gradients.append(grad_z)
        # Reverse the list to match the order of the layers
        gradients.reverse()
        return gradients[0]  # Return the gradient with respect to the input

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        
        # Update parameters in each layer
        for layer in self._layers:
            if hasattr(layer, "update_params"):
                layer.update_params(learning_rate)

