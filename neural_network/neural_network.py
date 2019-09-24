"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/neural-network
"""

import numpy as np
from nn_components.layers import FCLayer, ActivationLayer, BatchNormLayer
from tqdm import tqdm

class NeuralNetwork:

    def __init__(self, epochs, batch_size, optimizer, nn_structure):
        """
        Deep neural network architecture.

        Parameters
        ----------
        epochs: (integer) number of epochs to train.
        batch_size: (integer) number of batch size.
        optimizer: (object) optimizer object uses to optimize the loss.
        nn_structure: A list of 2-element tuple (num_neuron, activation)
                 represents neural network architecture.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.layers = self._structure(nn_structure)

    def _structure(self, nn_structure):
        """
        Structure function that initializes neural network architecture.

        Parameters
        ----------
        nn_structure: (list) a list of dictionaries define Neural Network architecture.

        - Each dict element in the list should have following key-pair value:
            + num_neurons: (int) define number of neurons in the dense layer.
            + weight_init: (str) choose which kind to initialize the weight, either `he` `xavier` or `std`.
            + activation (optional): (str) apply activation to the output of the layer. LINEAR -> ACTIVATION.
            + batch_norm (optional): (any) apply batch norm to the output of the layer. LINEAR -> BATCH NORM -> ACTIVATION.
        
        """
        layers = []
        for struct in nn_structure:
            num_neurons = struct["num_neurons"]
            weight_init = struct["weight_init"]
            fc = FCLayer(num_neurons=num_neurons, weight_init=weight_init)
            fc.initialize_optimizer(self.optimizer)
            layers.append(fc)
            if "batch_norm" in struct:
                bn_layer = BatchNormLayer()
                bn_layer.initialize_optimizer(self.optimizer)
                layers.append(bn_layer)
            if "activation" in struct:
                activation = struct["activation"]
                act_layer = ActivationLayer(activation=activation)
                layers.append(act_layer)
        return layers

    def _loss(self, Y, Y_hat):
        """
        Compute cross-entropy loss.

        Parameters
        ----------
        Y: one-hot encoding label. shape=(num_dataset, num_classes)
        Y_hat: softmax probability distribution over each data point. 
            shape=(num_dataset, num_classes)

        Returns
        -------
        J: cross-entropy loss.
        """
        assert Y.shape == Y_hat.shape, "Unmatch shape."
        return -np.mean(np.sum(Y*np.log(Y_hat), axis=1), axis=0)

    def _forward(self, train_X, prediction=False):
        """
        NN forward propagation level.

        Parameters
        ----------
        train_X: training dataset X.
                shape = (N, D)
        prediction: whether this forward pass is prediction stage or training stage.

        Returns
        -------
        Probability distribution of softmax at the last layer.
            shape = (N, C)
        """
        inputs = train_X
        for layer in self.layers:
            if isinstance(layer, BatchNormLayer):
                inputs = layer.forward(inputs, prediction=prediction)
                continue
            inputs = layer.forward(inputs)
        output = inputs
        return output

    def _backward_last(self, Y, Y_hat):
        """
        Special formula of backpropagation for the last layer.
        """
        m = Y.shape[0]
        delta = (Y_hat - Y)/m # shape = (N, C)
        dW = self.layers[-3].output.T.dot(delta)
        self.layers[-2].update_params(dW)
        dA_prev = delta.dot(self.layers[-2].W.T)
        return dA_prev

    def _backward(self, Y, Y_hat, X):
        """
        NN backward propagation level. Update weights of the neural network.

        Parameters
        ----------
        Y: one-hot encoding label.
            shape = (N, C).
        Y_hat: output values of forward propagation NN.
            shape = (N, C).
        X: training dataset.
            shape = (N, D).
        """
        dA_prev = self._backward_last(Y, Y_hat)
        for i in range(len(self.layers)-3, 0, -1):
            if isinstance(self.layers[i], ActivationLayer):
                dA_prev = self.layers[i].backward(dA_prev, None)
                continue
            dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])
        _ = self.layers[i-1].backward(dA_prev, X)

    def train(self, X_train, Y_train, *args):
        """
        Training function.

        Parameters
        ----------
        X_train: training dataset X.
        Y_train: one-hot encoding label.
        """
        for e in range(self.epochs):
            batch_loss = 0
            num_batches = 0
            pbar = tqdm(range(0, X_train.shape[0], self.batch_size), desc="Epoch " + str(e+1))
            for it in pbar:
                Y_hat = self._forward(X_train[it:it+self.batch_size])
                self._backward(Y_train[it:it+self.batch_size], Y_hat, X_train[it:it+self.batch_size])
                loss = self._loss(Y_train[it:it+self.batch_size], Y_hat)
                batch_loss += loss
                num_batches += 1
                pbar.set_description("Epoch " + str(e+1) + " - Loss: %.4f" % (batch_loss/num_batches))
            print("Loss at epoch %s: %f" % (e + 1 , batch_loss / num_batches))            

    def predict(self, test_X):
        """
        Predict function.
        """
        y_hat = self._forward(test_X, prediction=True)
        return np.argmax(y_hat, axis=1)

    def save(self, name):
        import pickle
        with open(name, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)