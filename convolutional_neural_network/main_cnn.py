import sys
sys.path.append("..")
import os
from libs.utils import load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from convolutional_neural_network import CNN


def main(use_keras=False):
    load_dataset_mnist("../libs")
    mndata = MNIST('../libs/data_mnist')
    arch = [{"type": "conv", "filter_size": (3, 3), "filters": 6, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he_normal"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            {"type": "conv", "filter_size": (3, 3), "filters": 16, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he_normal"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            {"type": "conv", "filter_size": (3, 3), "filters": 32, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he_normal"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            "flatten",
            {"type": "fc", "num_neurons": 128, "weight_init": "he_normal", "activation": "relu"}, # use "batch_norm": None
            {"type": "fc", "num_neurons": 64, "weight_init": "he_normal", "activation": "relu"},
            {"type": "fc", "num_neurons": 10, "weight_init": "he_normal", "activation": "softmax"}
            ]
    epochs = 5
    batch_size = 64
    learning_rate = 0.006
    if use_keras:
        print("Train MNIST dataset by CNN with Keras/Tensorflow.")
        from keras.optimizers import SGD as SGDKeras
        from cnn_keras import CNNKeras
        training_phase = True
        optimizer = SGDKeras(lr=learning_rate)
        cnn = CNNKeras(epochs=epochs, batch_size=batch_size, optimizer=optimizer, cnn_structure=arch)
    else:
        print("Train MNIST dataset by CNN with pure Python: Numpy.")
        optimizer = Adam(alpha=learning_rate)
        cnn = CNN(epochs=epochs, batch_size=batch_size, optimizer=optimizer, cnn_structure=arch)
        weight_path = "cnn_weights.pickle"
        training_phase = weight_path not in os.listdir(".")
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels, nn=True)
        
        if not use_keras:
            cnn.train(images, labels)
            cnn.save(weight_path)
        else:
            cnn.train(images, labels)
            training_phase = False
    if not training_phase:
        import pickle
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, nn=True, test=True)
        if not use_keras:
            with open(weight_path, "rb") as f:
                cnn = pickle.load(f)
        pred = cnn.predict(images_test)

        print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
        from sklearn.metrics.classification import confusion_matrix

        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, pred))

def test():
    import numpy as np
    from nn_components.layers import ConvLayer, PoolingLayer
    from optimizations_algorithms.optimizers import SGD
    import tensorflow as tf
    tf.enable_eager_execution()

    filter_size = (3, 3)
    filters = 16
    padding = "SAME"
    stride = 1

    optimizer = SGD()

    conv_layer = ConvLayer(filter_size=filter_size, filters=filters, padding=padding, stride=stride)
    conv_layer.initialize_optimizer(optimizer)
    conv_layer.debug = True

    pool_filter_size = (2, 2)
    pool_stride = 2
    pool_mode = "max"
    pool_layer = PoolingLayer(filter_size=pool_filter_size, stride=pool_stride, mode=pool_mode)

    X = np.random.normal(size=(16, 12, 12, 3))

    d_prev = np.random.normal(size=(16, 12, 12, 16))

    my_conv_forward = conv_layer.forward(X)
    my_dA, my_dW = conv_layer.backward(d_prev, X) 
    my_pool_forward = pool_layer.forward(X)

    with tf.device("/cpu:0"):
        tf_conv_forward = tf.nn.conv2d(X, conv_layer.W, strides=(stride, stride), padding=padding).numpy()
        tf_dW = tf.nn.conv2d_backprop_filter(X, filter_sizes=filter_size + (X.shape[-1], filters), out_backprop=d_prev,
                                            strides=(1, stride, stride, 1), padding=padding).numpy()
        tf_dA = tf.nn.conv2d_backprop_input(input_sizes=X.shape, filter=conv_layer.W, out_backprop=d_prev, 
                                            strides=(1, stride, stride, 1), padding=padding).numpy()

        tf_pool_forward = tf.nn.max_pool2d(X, ksize=pool_filter_size, strides=(pool_stride, pool_stride), padding="VALID")

    blank = "----------------------"
    print(blank + "TEST FORWARD CONVOLUTION" + blank)
    forward_result = np.allclose(my_conv_forward, tf_conv_forward)
    forward_out = "PASS" if forward_result else "FAIL"
    print("====> " + forward_out)
    
    print(blank + "TEST BACKWARD CONVOLUTION" + blank)
    dW_result = np.allclose(my_dW, tf_dW)
    dW_out = "PASS" if dW_result else "FAIL"
    print("====> dW case: " + dW_out)
    dA_result = np.allclose(my_dA, tf_dA)
    dA_out = "PASS" if dA_result else "FAIL"
    print("====> dA case: " + dA_out)

    print(blank + "TEST FORWARD POOLING" + blank)
    pool_result = np.allclose(my_pool_forward, tf_pool_forward)
    pool_out = "PASS" if pool_result else "FAIL"
    print("====> " + pool_out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A CNN program.")
    parser.add_argument("--keras", action="store_true", help="Whether use keras or not.")
    parser.add_argument("--test", action="store_true", help="Run the test cases.")
    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(use_keras=args.keras)