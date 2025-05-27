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


def sigmoid_derivative(x):
    return x * (1 - x)


def cal_BSE_single_loss(yi, yi_pred, epsilon=1e-15):
    sig = sigmoid(yi_pred)
    sig = min(max(sig, epsilon), 1 - epsilon)
    return -(yi * math.log(sig) + (1 - yi) * math.log(1 - sig))


def is_1d_list(lst):
    return all(not isinstance(i, list) for i in lst)


class Neuron:
    def __init__(self, next_layer_neurons_no=0):
        self.next_layer_neurons_no = next_layer_neurons_no

    def __cal_value(self, prev_layer_ouput):
        if isinstance(prev_layer_ouput, list):
            self.z = 0
            for i in range(len(prev_layer_ouput)):
                self.z += prev_layer_ouput[i]
            self.a = sigmoid(self.z)
        else:
            self.a = prev_layer_ouput

    def __generate_weight(self, next_layer_neurons_no):
        if next_layer_neurons_no == 0:
            return
        self.weights = []
        for _ in range(0, next_layer_neurons_no):
            self.weights.append(rand_float(0, 1))

    def __produce_output(self):
        self.output = []
        for i in range(len(self.weights)):
            self.output.append(self.a * self.weights[i])

    def forward(self, prev_layer_ouput):
        self.__cal_value(prev_layer_ouput)
        self.__generate_weight(self.next_layer_neurons_no)
        self.__produce_output()
        return self.output

    def compute_error(self, target=None, next_layer_errors=None):
        if self.weights is None:
            self.error = (target - self.output) * self.output * (1 - self.output)
            return
        self.error = 0
        for i in range(len(self.weights)):
            self.error += self.weights[i] * next_layer_errors[i]
        self.error = sigmoid_derivative(self.a) * self.error

    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.error * self.a


class Layer:
    def __init__(self, neuron_no, next_layer_neurons_no=0):
        self.neurons = []
        self.output = []
        for _ in range(0, neuron_no):
            neuron = Neuron(next_layer_neurons_no)
            self.neurons.append(neuron)

    def forward(self, input=None):
        self.output = []
        for i in range(0, len(self.neurons)):
            if is_1d_list(input):
                self.output.append(self.neurons[i].forward(1))
            elif input is not None:
                matching_input = []
                for j in range(len(input)):
                    matching_input.append(input[j][i - 1])
                self.output.append(self.neurons[i].forward(matching_input))
            else:
                break
        return self.output

    def compute_errors(self, target=None, next_layer=None):
        for i, neuron in enumerate(self.neurons):
            if target is not None:
                neuron.compute_error(target=target)
            else:
                next_layer_errors = [neuron.error for neuron in next_layer.neurons]
                neuron.compute_error(next_layer_errors=next_layer_errors)

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)


class Network:
    def __init__(self, layers_no, layers_sizes):
        self.layers = []
        for i in range(0, layers_no):
            if i < len(layers_sizes) - 1:
                layer = Layer(layers_sizes[i], layers_sizes[i + 1])
            else:
                layer = Layer(layers_sizes[i])
            self.layers.append(layer)

    def forward(self, input):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(input)
            else:
                self.layers[i].forward(self.layers[i - 1].output)
        return self.layers[len(self.layers) - 1].forward()

    def back_propagate(self, target, learning_rate):
        self.layers[-1].compute_errors(target=target)

        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].compute_errors(next_layer=self.layers[i + 1])

        for layer in self.layers:
            layer.update_weights(learning_rate)

    def train(self, train_data, labels, learning_rate, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for input, target in zip(train_data, labels):
                output = self.forward(input)
                loss = cal_BSE_single_loss(target, output[0])
                total_loss += loss
                self.back_propagate(target=target, learning_rate=learning_rate)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data)}")


if __name__ == "__main__":
    network_path = "./network_config.txt"
    with open(network_path, "r") as file:
        lines = file.readlines()

    layers_sizes = []
    for i in range(0, len(lines)):
        if i == 0:
            layers_no = int(lines[i])
        else:
            layers_sizes.append(int(lines[i]))

    data_path = "./xor_data.txt"
    with open(data_path, "r") as file:
        raw_data = file.readlines()

    train_data = []
    labels = []
    for line in raw_data:
        row = list(map(float, line.split(",")))
        train_data.append(row[:-1])
        labels.append(row[-1:][0])

    network = Network(layers_no, layers_sizes, train_data[0])
    network.train(train_data, labels, learning_rate=0.1, epochs=100)

    new_input = [0, 0]
    predicted_output = network.forward(new_input)
    print(f"Prediction for input {new_input}: {predicted_output}")
