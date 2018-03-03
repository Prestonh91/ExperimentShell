import numpy as np
from KNN.file_reader import FileReader
from KNN.data_processor import DataProcessor
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from NeuralNets.neuralnode import Node
from NeuralNets.neuralnode import Layer
from NeuralNets.neuralnode import NeuralNet


def loadData():
    data = datasets.load_iris()
    return data.data, data.target


def processIris(data, targets):
    train, test, train_t, test_t = \
        train_test_split(data, targets, train_size=0.7, test_size=0.3)
    return train, test, train_t, test_t


def main():
    data, targets = loadData()
    norm_data = preprocessing.normalize(data)
    train, test, train_t, test_t = processIris(norm_data, targets)

    file_reader = FileReader()
    data_processor = DataProcessor()
    raw_data = file_reader.read_file("health.txt")
    h_data , h_data_norm, p_targets = data_processor.process_health(raw_data)
    p_train, p_test, p_train_t, p_test_t = processIris(h_data, p_targets)

    iris_network = NeuralNet()
    iris_network.create_layer(3)
    iris_network.train_network(train, train_t)
    iris_predictions = iris_network.predict(test)

    correct = 0
    for i in range(len(test_t)):
        if iris_predictions[i] == test_t[i]:
            correct += 1

    print("Iris with 3 nodes in hidden layer")
    print("Iris prediction correct = ", correct, "out of", len(test),
          "\nAccuracy = ", (correct / len(test_t) * 100))

    iris2_network = NeuralNet()
    iris2_network.create_layer(6)
    iris2_network.train_network(train, train_t)
    iris2_predictions = iris2_network.predict(test)

    correct = 0
    for i in range(len(test_t)):
        if iris2_predictions[i] == test_t[i]:
            correct += 1

    print("Iris with 6 nodes in hidden layer")
    print("Iris prediction correct = ", correct, "out of", len(test),
          "\nAccuracy = ", (correct / len(test_t) * 100))

    pima_network = NeuralNet()
    pima_network.create_layer(4)
    pima_network.train_network(p_train, p_train_t)
    pima_predictions = pima_network.predict(p_test)

    correct = 0
    for i in range(len(p_test_t)):
        if pima_predictions[i] == p_test_t[i]:
            correct += 1

    print("Pima with 4 nodes in hidden layer")
    print("Pima prediction correct = ", correct, "out of", len(p_test),
          "\nAccuracy = ", (correct / len(p_test_t) * 100))

    pima2_network = NeuralNet()
    pima2_network.create_layer(6)
    pima2_network.train_network(p_train, p_train_t)
    pima2_predictions = pima2_network.predict(p_test)

    correct = 0
    for i in range(len(p_test_t)):
        if pima2_predictions[i] == p_test_t[i]:
            correct += 1

    print("Pima with 6 nodes in hidden layer")
    print("Pima prediction correct = ", correct, "out of", len(p_test),
          "\nAccuracy = ", (correct / len(p_test_t) * 100))




if __name__ == "__main__":
    main()