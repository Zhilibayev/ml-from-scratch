import sys
sys.path.append("..")

from libs.utils import load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from convolutional_neural_network import CNN

def main():
    load_dataset_mnist("../libs")
    mndata = MNIST('../libs/data_mnist')
    lenet_arch = [{"type": "conv", "filter_size": (5, 5), "filters": 6, "padding": "SAME", "stride": 1, "activation": "sigmoid", "batch_norm": None},
                {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
                {"type": "conv", "filter_size": (5, 5), "filters": 16, "padding": "SAME", "stride": 1, "activation": "sigmoid", "batch_norm": None},
                {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
                "flatten",
                {"type": "fc", "num_neurons": 120, "weight_init": "std", "activation": "tanh"},
                {"type": "fc", "num_neurons": 84, "weight_init": "std", "activation": "tanh"},
                {"type": "fc", "num_neurons": 10, "weight_init": "std", "activation": "softmax"}
                ]
    epochs = 20
    batch_size = 32
    learning_rate = 0.1
    optimizer = SGD(alpha=learning_rate)
    cnn = CNN(epochs=epochs, batch_size=batch_size, optimizer=optimizer, cnn_structure=lenet_arch)
    training_phase = True
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels, nn=True)
        cnn.train(images[:10000], labels[:10000])

if __name__ == "__main__":
    main()