import pandas as pd
import numpy as np
import copy

class Node:
    def __init__(self, label=None, path=None):
        self.path = path
        self.label = label

class ID3Tree:
    def __init__(self, data, targets, features):
        # The start of the tree
        self.start = self.make_tree(data, targets, features)

    def calc_entropy(self, probability):
        if probability != 0:
            return -probability * np.log2(probability)
        else:
            return 0

    def make_tree(self, data, targets, features):
        """
        If all examples have the same label
            return a leaf with the label
        Else if there are no features left to test
            return a lead with the most common label
        Else
            Consider each available feature
            Choose the one that maximizes information gain
            Create a new node for that feature

            For each possible value of the feature
                Create a branch for this value
                Create a subset of the examples for each branch
                Recursively call the function to create a node at the branch
        """
        # For ease combine data and classes to build the tree


        # If all examples have the same label(Class Value)
        uniques = pd.DataFrame(pd.unique(targets.values))

        if len(uniques) == 1:
            return Node(label=targets.iloc[0, 0])
        elif len(features) == 0:  # If no features left get most seen target
            leaf = pd.DataFrame(pd.value_counts(targets.iloc[:, 0])).index[0]
            return Node(label=leaf)
        else:
            # print("Need to subset the data and makes" +
            #       "branches based off features\n")

            # Prepare the data to calculate info gain
            feature_subset = data.filter(items=features)

            # Add the target list onto the data for convenience later
            target_name = targets.columns.values[0]

            entropies = []
            # Loop through each feature and calculate the info gain
            for feat in features:
                entropies.append(self.calc_info_gain(feature_subset.copy(True),
                                                     targets, feat))

            feature_choice = features[np.argmin(entropies)]
            feature_values = list(pd.unique(feature_subset[feature_choice]))

            subsets = []
            for feat_val in feature_values:
                subsets.append(feature_subset[feature_subset[feature_choice] ==
                                              feat_val])

            feature_dictionary = {}
            new_features = copy.deepcopy(features)
            del new_features[np.argmin(entropies)]
            for i in range(len(subsets)):
                target_subset = targets.ix[subsets[i].index.tolist()]
                feature_dictionary.update({feature_values[i]:
                                               self.make_tree(subsets[i],
                                                              target_subset,
                                                              new_features)})

            path = {feature_choice: feature_dictionary}
            # print('Looped through the recursive function\n')
            # print(path, '\n')
            return Node(path=path)

    def calc_info_gain(self, data, classes, feature):
        # Find the overall counts of data & classes
        class_name = classes.columns.values[0]
        n_data = len(data)

        # Find the features unique values count & names
        info_gain = pd.DataFrame(pd.value_counts(data[feature]))

        # Store the name of unique feature values
        feature_choices = list(info_gain.index)
        # Store the name of unique class values
        class_choices = \
            list(pd.DataFrame(pd.value_counts(classes.iloc[:, 0])).index)
        # Not 100% sure if this in necessary yet
        modifier = pd.DataFrame(data=None,
                                index=feature_choices,
                                columns=class_choices)
        data[class_name] = classes
        entropy_total = 0.0
        # Loop through the different unique feature values
        for row in range(len(feature_choices)):
            entropy = 0.0

            # Loop through the unique class values for the current class value
            for column in range(len(class_choices)):
                # Prepare a data frame that has only the rows with the current
                # feature value we are looking at
                feature_choice = \
                    pd.DataFrame(data[data[feature] == feature_choices[row]])
                # Find the count of the feature value with the current class
                # value
                count = len(feature_choice[feature_choice[class_name] ==
                                          class_choices[column]])
                modifier.iloc[row, column] = count
                # Add to the entropy for the current feature value
                entropy += self.calc_entropy(count / len(feature_choice))

            # Add to the total entropy the current feature value using a
            # weighted average
            entropy_total += (info_gain.loc[feature_choices[row], feature] /
                              n_data) * entropy

        return entropy_total

    def display_tree(self):
        self.display_nodes(self.start)

    def predict(self, data, targets):
        predictions = []
        for index, row in data.iterrows():
            (self.find_leaf(row, self.start, predictions))

        return predictions



    def display_nodes(self, node):
        if node.path != None:
            path = node.path
            print("Inside ", list(path.keys())[0], "node with")
            path = path[list(path.keys())[0]]
            print(list(path.keys()), " values")
            for key, value in path.items():
                print("Going into ", key, "Node")
                self.display_nodes(value)
        else:
            print(node.label, "\n")

    def find_leaf(self, data, node, predictions):
        if node.path is not None:
            feature = list(node.path.keys())[0]
            data_value = data[feature]
            for feat_val, new_path in node.path[feature].items():
                if data_value == feat_val:
                    self.find_leaf(data, new_path, predictions)
        else:
            predictions.append(node.label)
            return

class TreeClassifier:
    def __init__(self):
        pass

    def fit(self, data, targets, features):
        tree = ID3Tree(data, targets, features)
        return tree