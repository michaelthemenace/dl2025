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


def cal_BSE_single_loss(yi, yi_pred, epsilon=1e-15):
    sig = sigmoid(yi_pred)
    sig = min(max(sig, epsilon), 1 - epsilon)
    return -(yi * math.log(sig) + (1 - yi) * math.log(1 - sig))


class Neuron:
    def __init__(self, value=None):
        self.value = value
        self.weight = rand_float(0, 1)
        self.output = None
        self.error = None

    def forward(self):
        self.output = sigmoid(self.value * self.weight + self.bias)
        return self.output

    def compute_error(self, target=None, next_layer=None):
        if target is not None:
            self.error = (target - self.output) * self.output * (1 - self.output)
        else:
            self.error = (
                self.output * (1 - self.output) * (self.weight * next_layer[0].error)
            )

    def update_weights(self, learning_rate):
        self.weight -= learning_rate * self.error * self.value


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
        self.output = [neuron.forward() for neuron in self.neurons]
        return self.output

    def compute_errors(self, target=None, next_layer=None):
        for i, neuron in enumerate(self.neurons):
            if target is not None:
                neuron.compute_error(target=target)
            else:
                neuron.compute_error(next_layer=next_layer.neurons)

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)


class Network:
    def __init__(self, layers_no, layers_sizes, input):
        self.layers = []
        next_input = input
        for i in range(0, layers_no):
            layer = Layer(i, layers_sizes[i], next_input)
            next_input = layer.forward()
            self.layers.append(layer)

    def forward(self, input):
        next_input = input
        for layer in self.layers:
            next_input = layer.forward()
        return next_input

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
