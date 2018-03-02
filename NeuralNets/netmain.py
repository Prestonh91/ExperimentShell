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
    norm_data = preprocessing.normalize(data)
    train, test, train_t, test_t = \
        train_test_split(norm_data, targets, train_size=0.7, test_size=0.3)
    return train, test, train_t, test_t

def main():
    data, targets = loadData()
    train, test, train_t, test_t = processIris(data, targets)

    # file_reader = FileReader()
    # data_processor = DataProcessor()
    # raw_data = file_reader.read_file("health.txt")
    # h_data , h_data_norm, targets = data_processor.process_health(raw_data)

    iris_network = NeuralNet()
    iris_network.create_layer(4)
    iris_network.train_network(train[:1], train_t)
    # for layer in iris_network.layers:



    # layer = Layer(3)
    # layer.calc_layer_output(train)
    # print(layer.outputs)

    # h_layer = Layer(1)
    # h_layer.calc_layer_output(h_data_norm)
    # print(h_layer.outputs)


if __name__ == "__main__":
    main()