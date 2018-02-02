import numpy as np

class FeatureNode:
    def __init__(self):
        self.feature = None
        self.target = None
        self.children = []

class ClassNode:
    def __init__(self):
        self.target_class = None


class ID3Tree:
    def __init__(self):
        self.start = FeatureNode()
        self.data = None
        self.targets = None

    def calc_entropy(self, probabilities):
        if probabilities != 0:
            return -probabilities * np.log2(probabilities)
        else:
            return 0

    def make_tree(self):
        # If all examples have the same label

        # If there are no more features left to test
        print("Constructing a decision tree, please wait....\n")

    def display_tree(self):
        print("This is my tree that I am working on!!")


class TreeClassifier:
    def __init__(self):
        self.tree = ID3Tree()

    def fit(self, data, targets):
        self.tree.data = data
        self.tree.targets = targets
        return self.tree

    def make_tree(self):
        pass

    def __calc_info_gain(self):
        pass