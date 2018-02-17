import numpy as np

class Layer():
    def __init__(self, num_of_nodes=None):
        self.num_of_nodes = num_of_nodes
        self.nodes = []
        self.outputs = []

    def calc_layer_output(self, inputs):
        self.init_node_weights(len(inputs[0]))

        # Loop through each row of inputs first
        for row in inputs:
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
                # Clear the nodes a and h in order to prepare it for next row
                node.clear_node()

            # Append the rows activations to store them in the layer
            self.outputs.append(row_output)

    def init_node_weights(self, num_of_inputs):
        # For the number of nodes create a new node
        for i in range(self.num_of_nodes):
            # Account for bias "input" here by sending the number of inputs
            # plus one. This will create a weight for the bias "input"
            node = Node(num_of_inputs + 1)
            self.nodes.append(node)


class Node():
    def __init__(self, num_of_inputs=None):
        self.num_of_inputs = num_of_inputs
        self.weights = None
        self.h = 0
        self.a = 0
        # initializes the weights bias is added in the layer
        self.calc_weights()

    def calc_weights(self):
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
        self.h = 0
        self.a = 0