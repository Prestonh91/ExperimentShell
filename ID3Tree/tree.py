from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
import pandas as pd
from ID3Tree.id3tree import ID3Tree
from ID3Tree.id3tree import TreeClassifier


def load_toy_set():
    data = pd.read_csv("lenses.txt", sep="\s+", header=None)
    data.drop([0], axis=1,inplace=True)
    data = data.rename(columns={1: 'Age', 2:'Glass_Prescription',
                                3:'Astigmatic', 4:'Tear_Rate',
                                5:'Contact'})
    clean_up = {'Age': {1: 'Young', 2: 'Pre-presbyopic',
                        3 : 'presbyopic'},
                'Glass_Prescription' : {1 : 'Myope', 2 : 'Hypermetrope'},
                'Astigmatic' : {1:'No', 2:'Yes'},
                'Tear_Rate': {1:'Reduced', 2:'Normal'},
                'Contact': {1:'Hard', 2:'Soft', 3:'None'}}
    data.replace(to_replace=clean_up, inplace=True)
    targets = pd.DataFrame(data.Contact)
    data.drop(labels=['Contact'],axis=1,inplace=True)
    return data, targets


def main():
    data, targets = load_toy_set()
    features = list(data.columns.values)
    train, test, train_t, test_t = train_test_split(data,
                                                    targets,
                                                    train_size=0.7)

    unique_vals = []
    unique_tar = pd.unique(targets.iloc[:,0])
    for col in data:
        values = pd.unique(data[col])
        for val in values:
            unique_vals.append(val)

    le_data = preprocessing.LabelEncoder()
    le_targets = preprocessing.LabelEncoder()
    le_data.fit(unique_vals)
    le_targets.fit(unique_tar)
    for col in data:
        data[col] = le_data.transform(data[col])
    targets.iloc[:, 0] = le_targets.transform(targets.iloc[:, 0])

    sk_train, sk_traint, sk_test, sk_testt = train_test_split(data,
                                                              targets,
                                                              train_size=0.7)

    classifier = TreeClassifier()
    sk_tree = tree.DecisionTreeClassifier()

    my_tree = classifier.fit(train, train_t, features)
    sk_train = sk_train.values.T.tolist()
    sk_traint = sk_traint.values.T.tolist()
    sk_tree = sk_tree.fit(sk_train, sk_traint)

    predictions = my_tree.predict(test, test_t)
    sk_test = sk_test.values.T.tolist()
    sk_predictions = sk_tree.predict(sk_test)

    sk_predictions = list(sk_predictions[0].astype(dtype=int))
    sk_predictions = le_targets.inverse_transform(sk_predictions)

    count = 0
    sk_count = 0
    for i in range(len(predictions)):
        if predictions[i] == test_t.iloc[i, 0]:
            count += 1
        if sk_predictions[i] == test_t.iloc[i, 0]:
            sk_count += 1


    print("My algorith got ", count, "correct out of ", len(predictions))
    print("Sklearns algorithm got ", sk_count, "correct out of ",
          len(sk_predictions))


if __name__ == "__main__":
    main()
