import numpy as np


class NeuralNet:
    def __init__(self):
        self.num_of_layers = 0
        self.layers = []
        self.training_rate = 0.2
        pass

    def create_layer(self, num_of_nodes, output_layer=False):
        new_layer = Layer(num_of_nodes, output_layer)
        self.layers.append(new_layer)
        self.num_of_layers += 1

    def train_network(self, inputs, targets):
        # This is used to determine where number of weights come from when
        # initializing them as well were to get the inputs for each layer when
        # calculating the layers output
        first_run = True
        class_num = len(np.unique(targets))

        # Once the user decides its time to run add the output layer
        self.create_layer(class_num, True)

        print("The weights before the first update")
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

            print("\n")
            self.layers[i].display_weights()

        for epochs in range(5000):
            # Loop through each input row to being training
            for i, row in list(enumerate(inputs)):
                # Reset to true so the layer can pull the inputs from the
                # correct place (e.i. The input row or the layer previous
                first_run = True

                # Calculate the output at each layer
                for j, layer in list(enumerate(self.layers)):
                    if not first_run:
                        layer.calc_layer_output(self.layers[j - 1].outputs)
                        pass
                    else:
                        layer.calc_layer_output(row)
                        first_run = False

                # Loop through the layers backwards to calculate error at each layer
                for k, layer in reversed(list(enumerate(self.layers))):
                    tar_array = self.get_target_array(class_num, targets[i])
                    if not layer.output_layer:
                        layer.calc_hidden_error(self.layers[k + 1])
                    else:
                        layer.calc_output_error(tar_array)

                # Loop through forward and update the weights. Set first_run to true
                # to make sure we take the inputs from the correct spot again
                first_run = True
                for f, layer in list(enumerate(self.layers)):
                    if not first_run:
                        layer.update_weights(self.layers[f - 1].outputs[0],
                                             self.training_rate)
                    else:
                        layer.update_weights(row, self.training_rate)
                        first_run = False

                    if epochs == 4999:
                        print("Weights after", epochs, "updates")
                        layer.display_weights()

                for layer in self.layers:
                    layer.clear_nodes()

    def get_target_array(self, num_of_classes, target):
        new_target = np.zeros(num_of_classes, dtype=int)
        new_target[target] = 1
        return new_target


class Layer:
    def __init__(self, num_of_nodes=None, output_layer=False):
        self.output_layer = output_layer
        self.num_of_nodes = num_of_nodes
        self.nodes = []
        self.outputs = []

    def calc_layer_output(self, row):
        # Account here for the bias "input" by inserting -1 onto the front
        # of the row
        row = np.insert(row, 0, -1)

        # This is a one dimensional list to hold the outputs for the nodes
        # in this row
        row_output = []

        # Loop through the nodes to calc the activation for the row
        for node in self.nodes:
            # Append the nodes activation to the one dimensional list
            row_output.append(node.calc_output(row))

        # Append the rows activations to store them in the layer
        self.outputs.append(row_output)

    def init_node_weights(self, num_of_inputs):
        # For the number of nodes create a new node
        for i in range(self.num_of_nodes):
            # Account for bias "input" here by sending the number of inputs
            # plus one. This will create a weight for the bias "input"
            node = Node(num_of_inputs + 1)
            self.nodes.append(node)

    def display_weights(self):
        for node in self.nodes:
            print(node.weights)

    def calc_hidden_error(self, previous_layer):
        # aj(1-aj)sum[i-n](wjk*ek)
        for i, node in list(enumerate(self.nodes)):
            node.error = node.a * (1 - node.a)

            # Loop through the previous array and get the weighted error
            weighted_error = 0
            for j, inner_node in list(enumerate(previous_layer.nodes)):
                weighted_error += (inner_node.weights[i + 1] * inner_node.error)

            node.error *= weighted_error

    def calc_output_error(self, target):
        # aj(1 - aj)(aj -tj)
        for i, node in list(enumerate(self.nodes)):
            node.error = node.a * (1 - node.a) * (node.a - target[i])

    def update_weights(self, inputs, training_rate):
        inputs = np.insert(inputs, 0, -1)
        for i, node in list(enumerate(self.nodes)):
            for j, weight in list(enumerate(node.weights)):
                node.weights[j] = weight - \
                                  (training_rate * node.error * inputs[j])

    def clear_nodes(self):
        for node in self.nodes:
            self.outputs = []
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
        # runs through the inputs and weights to calc the h, bias input is
        # added in the layer class
        for i in range(len(inputs)):
            self.h += inputs[i] * self.weights[i]

        # print("Self.h = ", self.h)

    def calcA(self):
        # calc a according to the nodes current h value
        self.a = 1 / (1 + np.exp(-self.h))
        # print("Self.a = ", self.a)

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
