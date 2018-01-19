import numpy as np

class kNNModel:
    def __init__(self, n_neighbors, data, data_class):
        self.data = data
        self.data_class = data_class
        self.neighbors = n_neighbors

    def predict(self, inputs):
        # Gather the number of total inputs to predict
        n_inputs = np.shape(inputs)[0]

        # Create an array to hold the predictions with the correct size
        # of inputs
        predictions = np.zeros(n_inputs)

        # Loop through each input to find the closest nieghbors and make the
        # predictions for each input
        for n in range(n_inputs):
            # We are find the Euclidean distance between all the points of data
            # compared to the one input we are focusing one in the loop
            distances = np.sum((self.data - inputs[n, :]) ** 2, axis=1)

            # np.argsort returns an array of the indices of what the array would
            # be if sorted, we use this to get the k nearest neighbors
            indices = np.argsort(distances, axis=0)

            # Retrieve an array of the unique classes in the k-neighbors
            classes = np.unique(self.data_class[indices[:self.neighbors]])

            # If there is only 1 unique class make that class the prediction
            # If there is more than 1 unique class add the amount of times the
            # different class show up, the class that shows up the most will be
            # the prediction
            if len(classes) == 1:
                predictions[n] = np.unique(classes)
            else:
                # To make it easier to retrieve the correct class put the counts
                # of each class inside the index its class corresponds to
                counts = np.zeros(max(classes) + 1)

                for i in range(self.neighbors):
                    # Add to counts the indice of the class of the neighbor
                    counts[self.data_class[indices[i]]] += 1
                predictions[n] = np.argmax(counts)

        return predictions


class kNNClassifier:
    def __init__(self, n_neighbors):
        self.neighbors = n_neighbors

    def fit(self, data, targets):
        return kNNModel(self.neighbors, data, targets)