from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, test_data):
        predictions = []

        for data in test_data:
            predictions.append(0)

        return predictions


class HardCodeClassifier:
    def __init__(self):
        pass

    def fit(self, data, targets):
        return HardCodedModel()


def calc_correct_percentage(test_targets, guess_targets):
    total = len(test_targets)
    correct = 0
    for i in range (len(test_targets)):
        if guess_targets[i] == test_targets[i]:
            correct += 1
    return correct / total * 100.


def load_iris_data():
    iris = datasets.load_iris()
    return iris


if __name__ == "__main__":
    # Load Iris dataset and separate the Data and Targets from the dataset
    iris = load_iris_data()
    iris_data = iris.data
    iris_target = iris.target

    # Initialize a pre-built algorithm
    gaussian_classifier = GaussianNB()

    #Initialize our hardcoded algorithm
    hard_code_classifier = HardCodeClassifier()

    # Split the dataset into the Train set and the Test set along with the
    # targets. Splits the data set into 70% training and 30% test
    train, test, train_target, test_target = \
        train_test_split(iris_data,
                         iris_target,
                         train_size=0.7)

    #Use the gaussian implementation to train a model and predict the targets
    gaussian_model = gaussian_classifier.fit(train, train_target)
    gaussian_targets = gaussian_model.predict(test)

    gaussian_percentage = calc_correct_percentage(test_target,
                                                 gaussian_targets)
    print ("Gaussian Algorithm: {:.2f}%".format(gaussian_percentage))

    #User our hardcoded implementation to train a model and predict the targets
    hard_code_model = hard_code_classifier.fit(train, train_target)
    hard_code_targets = hard_code_model.predict(test)

    hard_code_percent = calc_correct_percentage(test_target,
                                                hard_code_targets)
    print ("Hard Coded Algorithm: {:.2f}%".format(hard_code_percent))


