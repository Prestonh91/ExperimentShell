from sklearn import datasets
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
    targets = data.Contact
    data.drop(labels=['Contact'],axis=1,inplace=True)
    return data, targets


def main():
    data, targets = load_toy_set()
    print(data)
    print(targets)


if __name__ == "__main__":
    main()
