import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self):
        self.num_of_layers = 0
        self.layers = []
        self.training_rate = 0.1
        self.epochs = 400
        self.accuracy_array = []
        pass

    def create_layer(self, num_of_nodes, output_layer=False):
        new_layer = Layer(num_of_nodes, output_layer)
        self.layers.append(new_layer)
        self.num_of_layers += 1

    def predict(self, test):
        predictions = []

        for row in test:
            self.predict_layer_outputs(row, predictions)

        return predictions

    def train_network(self, inputs, targets):
        total = (len(inputs))
        print("Total inputs = ", total)
        # This is used to determine where number of weights come from when
        # initializing them as well were to get the inputs for each layer when
        # calculating the layers output
        first_run = True
        class_num = len(np.unique(targets))

        # Once the user decides its time to run add the output layer
        self.create_layer(class_num, True)

        # Lets initialize the weights before running through
        for i in range(len(self.layers)):
            if not first_run:
                # Get the number of inputs from the node layer before if its not
                # the first layer
                self.layers[i].init_node_weights(self.layers[i-1].num_of_nodes)
            else:
                # Get the number of inputs from the row on first layer)
                self.layers[i].init_node_weights(len(inputs[0]))
                first_run = False

        for epoch in range(self.epochs):
            correct = 0
            # Loop through each input row to being training
            for i, row in list(enumerate(inputs)):

                # file.write("\nOutput layer at epoch " + str(epoch))
                self.calculate_layer_outputs(row)

                # Loop through the layers backwards to calculate error at each
                # layer
                for k, layer in reversed(list(enumerate(self.layers))):
                    unique_classes = len(np.unique(targets))
                    target = self.get_target_array(unique_classes, targets[i])
                    if not layer.output_layer:
                        self.layers[k].calc_hidden_error(self.layers[k - 1])
                    else:
                        guess = self.layers[k].calc_output_error(target)
                        if guess:
                            correct += 1


                # Loop through forward and update the weights. Set first_run to
                # true to make sure we take the inputs from the correct spot
                # again
                for f, layer in list(enumerate(self.layers)):
                    self.layers[f].update_weights(self.training_rate)

                    # if layer.output_layer:
                    #     file.write("\nNodes after weight update\n" +
                    #                self.layers[f].return_node())

                    self.layers[f].clear_nodes()

            self.accuracy_array.append(correct)

            if epoch % 50 == 0:
                print("At epoch ", epoch, "keep waiting")

        x = np.linspace(0, self.epochs, num=self.epochs, endpoint=True)
        plt.plot(x, self.accuracy_array, linewidth=1)
        plt.show()

    def get_target_array(self, num_of_classes, target):
        new_target = np.zeros(num_of_classes, dtype=int)
        if num_of_classes == 1:
            if target == 1:
                new_target[0] = 1
            else:
                new_target[0] = 0
        else:
            new_target[target] = 1
        return new_target

    def conv_array_to_target(self, tar_array):
        if len(tar_array) == 1:
            return tar_array[0]
        else:
            for i in range(len(tar_array)):
                if tar_array[i] == 1:
                    return i

    def predict_layer_outputs(self, row, guesses):
        file = open("tar_array.txt", "a")
        tar_array = []

        # Set to true so the layer can pull the inputs from the
        # correct place (e.i. The input row or the layer previous
        first_run = True
        # Calculate the output at each layer
        for j, layer in list(enumerate(self.layers)):
            if not first_run:
                self.layers[j].calc_layer_output(self.layers[j - 1].outputs)
            else:
                self.layers[j].calc_layer_output(row)
                first_run = False

            if layer.output_layer:
                for node in layer.nodes:
                    if node.a >= 0.5:
                        tar_array.append(1)
                    else:
                        tar_array.append(0)

                file.write(str(tar_array) + '\n')
                guesses.append(self.conv_array_to_target(tar_array))
                layer.clear_nodes()

        file.close()

    def calculate_layer_outputs(self, row):
        # Set to true so the layer can pull the inputs from the
        # correct place (e.i. The input row or the layer previous
        first_run = True
        # Calculate the output at each layer
        for j, layer in list(enumerate(self.layers)):
            if not first_run:
                self.layers[j].calc_layer_output(self.layers[j - 1].outputs)
            else:
                self.layers[j].calc_layer_output(row)
                first_run = False


class Layer:
    def __init__(self, num_of_nodes=None, output_layer=False):
        self.output_layer = output_layer
        self.num_of_nodes = num_of_nodes
        self.inputs = []
        self.nodes = []
        self.outputs = []

    def calc_layer_output(self, row):
        # Account here for the bias "input" by inserting -1 onto the front
        # of the row
        row = np.insert(row, 0, -1)

        # Save these for later calculations
        self.inputs = row
        # print ("Layers inputs = ", self.inputs)

        # This is a one dimensional list to hold the outputs for the nodes
        # in this row
        row_output = []

        # Loop through the nodes to calc the activation for the row
        for node in self.nodes:
            # Append the nodes activation to the one dimensional list
            row_output.append(node.calc_output(row))

        # Append the rows activations to store them in the layer
        self.outputs = row_output
        # print("Layer outputs = ", self.outputs)


    def init_node_weights(self, num_of_inputs):
        # For the number of nodes create a new node
        for i in range(self.num_of_nodes):
            # Account for bias "input" here by sending the number of inputs
            # plus one. This will create a weight for the bias "input"
            node = Node(num_of_inputs + 1)
            self.nodes.append(node)

    def display_weights(self):
        for node in self.nodes:
            print(node.weights,'\n')

    def display_activations(self):
        for node in self.nodes:
            print(node.a)

    def return_node(self):
        str_node = "Inputs = " + str(self.inputs) + "\n"
        str_node += "Outputs = " + str(self.outputs) + "\n"
        str_node += "A, H, Error, Weights:\n"
        for node in self.nodes:
            str_node += str(node.a) + " "
            str_node += str(node.h) + " "
            str_node += str(node.error) + "\n"
            str_node += str(node.weights)
            str_node += '\n'
        return str_node

    def calc_hidden_error(self, previous_layer):
        # aj(1-aj)sum[i-n](wjk*ek)
        for i, node in list(enumerate(self.nodes)):
            node.error = node.a * (1 - node.a)

            # Loop through the previous array and get the weighted error
            weighted_error = 0
            for j, inner_node in list(enumerate(previous_layer.nodes)):
                weighted_error += (inner_node.weights[i + 1] * inner_node.error)

            # print("Hidden error = ", node.error, "*", sum)
            node.error *= weighted_error

    def calc_output_error(self, target):
        prediction = []
        # print("Targets in calc output = ", target)
        # aj(1 - aj)(aj -tj)
        for i, node in list(enumerate(self.nodes)):
            node.error = node.a * (1 - node.a) * (node.a - target[i])

            if node.a > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)

        guess = True
        for i in prediction:
            if prediction[i] != target[i]:
                guess = False

        return guess

    def update_weights(self, training_rate):
        # print(self.inputs)
        for i, node in list(enumerate(self.nodes)):
            for j, weight in list(enumerate(node.weights)):
                node.weights[j] = weight - \
                                  (training_rate * node.error * self.inputs[j])

    def clear_nodes(self):
        self.inputs = []
        self.outputs = []
        for node in self.nodes:
            node.clear_node()


class Node:
    def __init__(self, num_of_inputs=None):
        self.num_of_inputs = num_of_inputs
        self.weights = None
        self.error = 0
        self.h = 0
        self.a = 0
        # initializes the weights bias is added in the layer
        self.init_weights()

    def init_weights(self):
        # Initializes the weights to small random neg and pos numbers
        self.weights = np.random.uniform(-0.5, 0.5, self.num_of_inputs)
        # print(self.weights, len(self.weights))

    def calcH(self, inputs):
        sum = ""
        # runs through the inputs and weights to calc the h, bias input is
        # added in the layer class
        for i in range(len(inputs)):
            self.h += inputs[i] * self.weights[i]
            sum = sum + "(" + str(inputs[i]) + " * " +\
                  str(self.weights[i]) + ") + \n"

        # print ("Node H calc = ", sum)

    def calcA(self):
        # calc a according to the nodes current h value
        self.a = 1 / (1 + np.exp(-self.h))
        # print("Self.a = ", self.a, '\n')

    def calc_output(self, inputs):
        # Calls functions necessary to calc activation according to given
        # inputs. Bias input should be added in Layer class
        self.calcH(inputs)
        self.calcA()
        return self.a

    def clear_node(self):
        # Clears the nodes 'a' and 'h' value to prepare for next set of inputs
        self.error = 0
        self.h = 0
        self.a = 0
