from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn import svm
from KNN.file_reader import FileReader
from KNN.data_processor import DataProcessor
from KNN.k_neighbors import kNNClassifier

def calc_correct_percentage(test_t, guess_t):
    total = len(test_t)
    correct = 0
    for i in range(total):
        if (test_t[i] == guess_t[i]):
            correct += 1
    percent = "%.2f" % (correct / total * 100.)
    print(correct, "/", total, " = ", percent, "%")


def load_iris_data():
    data = datasets.load_iris()
    return data.data, data.target


def get_classifier(neighbors):
    #classifier = GaussianNB()
    classifier = kNNClassifier(n_neighbors=neighbors)

    return classifier


def run_algorithm(data, targets, classifier, is_regression):
    # Split the train and test data and targets and shuffle, keeping the
    # correct targets with the correct data
    train, test, train_t, test_t = \
        train_test_split(data, targets, train_size=0.7)

    # Use my own knn algorithm
    model = classifier.fit(data, targets.ravel())
    predictions_geo, predictions_man = model.predict(test)

    # print("Number of neighbor = ", classifier.neighbors)
    print("My own algorithm(Euclidean Distance)")
    calc_correct_percentage(test_t, predictions_geo)

    print("My own algorithm(Manhattan Distance)")
    calc_correct_percentage(test_t, predictions_man)

    # User preloaded knn algorithm
    if not is_regression:
        k_classifier = KNeighborsClassifier(n_neighbors=5)
        model = k_classifier.fit(data, targets.ravel())
        cfl = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(cfl, train, train_t.ravel(), cv=5)
        print("Cross fold Validation",scores, "\n")
    elif is_regression:
        k_classifier = KNeighborsRegressor()
        model = k_classifier.fit(data,targets)
    k_predictions = model.predict(test)

    print("Pre-loaded algorithm")
    calc_correct_percentage(test_t, k_predictions)
    print("\n")


def main():
    # Load Iris dataset and separate the Data and Targets from the dataset
    # data, targets = load_iris_data()
    reader = FileReader()
    processor = DataProcessor()

    # Read data for car acceptability, process, and run
    raw_car_data = reader.read_file("car.txt")
    car_data, car_targets = processor.process_cars1(raw_car_data)


    classifier = get_classifier(5)
    run_algorithm(car_data, car_targets, classifier, False)

    # Read data for MPG car, process, and run
    raw_health_data = reader.read_file("health.txt")
    std_diabetes, norm_diabetes, tar_diabetes = \
        processor.process_health(raw_health_data)

    classifier = get_classifier(5)
    run_algorithm(std_diabetes, tar_diabetes, classifier, False)
    run_algorithm(norm_diabetes, tar_diabetes, classifier, False)

    raw_mpg_data = reader.read_file("mpg.txt")
    mpg_std_data, mpg_norm_data, mpg_targets = \
        processor.process_mpg_cars(raw_mpg_data)
    classifier = get_classifier(5)

    run_algorithm(mpg_std_data, mpg_targets, classifier, True)
    run_algorithm(mpg_norm_data, mpg_targets, classifier, True)

if __name__ == "__main__":
    main()
