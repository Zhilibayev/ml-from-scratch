import sys
sys.path.append("..")
import os
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from neural_network import NeuralNetwork
from libs.utils import load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST


load_dataset_mnist("../libs")
mndata = MNIST('../libs/data_mnist')
weight_path = "nn_weights.pickle"
training_phase = weight_path not in os.listdir(".")
if training_phase:
    images, labels = mndata.load_training()
    images, labels = preprocess_data(images, labels)
    epochs = 10
    batch_size = 64
    learning_rate = 0.01

    optimizer = Adam(learning_rate)
    archs = [
        {"num_neurons": 100, "weight_init": "he_normal", "activation": "relu", "drop_out": 0.8},
        {"num_neurons": 125, "weight_init": "he_normal", "activation": "relu", "drop_out": 0.8},
        {"num_neurons": 50, "weight_init": "he_normal", "activation": "sigmoid", "batch_norm": None},
        {"num_neurons": labels.shape[1], "weight_init": "he_normal", "activation": "softmax"}]
    nn = NeuralNetwork(epochs, batch_size, optimizer, archs)
    nn.train(images, labels)
    nn.save(weight_path)
else:
    import pickle
    images_test, labels_test = mndata.load_testing()
    images_test, labels_test = preprocess_data(images_test, labels_test, test=True)
    with open(weight_path, "rb") as f:
        nn = pickle.load(f)
    pred = nn.predict(images_test)

    print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
    from sklearn.metrics.classification import confusion_matrix

    print("Confusion matrix: ")
    print(confusion_matrix(labels_test, pred))