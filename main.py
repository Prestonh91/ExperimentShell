import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from file_reader import file_reader
from data_processor import data_processor
from k_neighbors import kNNClassifier

def calc_correct_percentage(test_t, guess_t):
    total = len(test_t)
    correct = (test_t == guess_t).sum()
    percent =  "%.2f" % (correct / total * 100.)
    print(correct, "/", total, " = ", percent, "%")


def load_iris_data():
    data = datasets.load_iris()
    return data.data, data.target


def get_classifier(neighbors):
    #classifier = GaussianNB()
    classifier = kNNClassifier(n_neighbors=neighbors)

    return classifier


def run_algorithm(data, targets, classifier):
    # Split the train and test data and targets and shuffle, keeping the
    # correct targets with the correct data
    train, test, train_t, test_t = \
        train_test_split(data, targets, train_size=0.7)

    # Use my own knn algorithm
    model = classifier.fit(data, targets)
    predictions_geo, predictions_man = model.predict(test)

    print("Number of neighbor = ", classifier.neighbors)
    print("My own algorithm(Euclidean Distance)")
    calc_correct_percentage(test_t, predictions_geo)

    print("My own algorithm(Manhattan Distance)")
    calc_correct_percentage(test_t, predictions_man)

    # User preloaded knn algorithm
    k_classifier = KNeighborsClassifier(n_neighbors=5)
    model = k_classifier.fit(data, targets)
    k_predictions = model.predict(test)

    print("Pre-loaded algorithm")
    calc_correct_percentage(test_t, k_predictions)
    print("\n\n")


def main():
    # Load Iris dataset and separate the Data and Targets from the dataset
    data, targets = load_iris_data()
    reader = file_reader()
    processor = data_processor()

    data = reader.read_file("car.txt")
    processor.process_cars1(data)



    # classifier = get_classifier(20)
    # run_algorithm(data, targets, classifier)
    #
    # classifier = get_classifier(10)
    # run_algorithm(data, targets, classifier)
    #
    # classifier = get_classifier(8)
    # run_algorithm(data, targets, classifier)
    #
    # classifier = get_classifier(6)
    # run_algorithm(data, targets, classifier)
    #
    # classifier = get_classifier(3)
    # run_algorithm(data, targets, classifier)


if __name__ == "__main__":
    main()


