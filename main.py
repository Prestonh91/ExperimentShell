import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from k_neighbors import kNNClassifier

def calc_correct_percentage(test_t, guess_t):
    total = len(test_t)
    correct = (test_t == guess_t).sum()
    percent =  "%.2f" % (correct / total * 100.)
    print(correct, "/", total, " = ", percent, "%")


def load_iris_data():
    data = datasets.load_iris()
    return data.data, data.target


def get_classifier():
    #classifier = GaussianNB()
    classifier = kNNClassifier(n_neighbors=3)

    return classifier


def run_algorithm(data, targets, classifier):
    # Split the train and test data and targets and shuffle, keeping the
    # correct targets with the correct data
    train, test, train_t, test_t = \
        train_test_split(data, targets, train_size=0.7)

    model = classifier.fit(data, targets)
    predictions = model.predict(test)
    calc_correct_percentage(test_t, predictions)



def main():
    # Load Iris dataset and separate the Data and Targets from the dataset
    data, targets = load_iris_data()


    classifier = get_classifier()
    predictions = run_algorithm(data, targets, classifier)

    # k_model = k_classifier.fit(train, train_target)
    # predictions = k_model.predict(test)

    # print (predictions)
    # print (test_target)


if __name__ == "__main__":
    main()


