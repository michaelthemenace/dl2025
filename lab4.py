import math


def rand_float(a, b):
    return a + (rand_below(1000000) / 1000000) * (b - a)


def rand_below(n):
    if n <= 0:
        raise ValueError("n must be greater than 0")
    k = n.bit_length()
    numbytes = (k + 7) // 8
    while True:
        r = int.from_bytes(random_bytes(numbytes), "big")
        r >>= numbytes * 8 - k
        if r < n:
            return r


def random_bytes(n):
    with open("/dev/urandom", "rb") as file:
        return file.read(n)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Layer:
    def __init__(self, layer_i, neuron_no, input):
        self.neurons = []
        self.output = []
        self.layer_i = layer_i
        for i in range(0, neuron_no):
            if i == 0:
                neuron = Neuron(1)
            else:
                neuron = Neuron(input[i - 1])
            self.neurons.append(neuron)

    def forward(self):
        for i in range(0, len(self.neurons)):
            self.output.append(self.neurons[i].forward())

        return self.output


class Neuron:
    def __init__(self, value=None):
        self.value = value

    def forward(self):
        weight = rand_float(0, 1)
        print(weight)
        return sigmoid(self.value * weight)


class Network:
    def __init__(self, layers_no, layers_sizes, input):
        self.layers = []
        next_input = input
        for i in range(0, layers_no):
            layer = Layer(i, layers_sizes[i], next_input)
            next_input = layer.forward()
            print(f"Layer {i} output: {next_input}")
            self.layers.append(layer)
        self.output = layer.output

    def print_results(self):
        print(f"Output: {self.output}")


if __name__ == "__main__":
    file_path = "./network_config.txt"
    with open(file_path, "r") as file:
        lines = file.readlines()

    layers_sizes = []
    for i in range(0, len(lines)):
        if i == 0:
            layers_no = int(lines[i])
        else:
            layers_sizes.append(int(lines[i]))

    network = Network(layers_no, layers_sizes, [0, 0])
    network.print_results()
