import math
import matplotlib.pyplot as plt


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1 if x > 0 else 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def convolve2d(image, kernel, stride=1):
    output_size = ((len(image) - len(kernel)) // stride) + 1
    output = [[0] * output_size for _ in range(output_size)]
    for i in range(0, output_size):
        for j in range(0, output_size):
            val = 0
            for m in range(len(kernel)):
                for n in range(len(kernel[0])):
                    val += image[i + m][j + n] * kernel[m][n]
            output[i][j] = val
    return output


def flatten(matrix):
    return [val for row in matrix for val in row]


class CNN:
    def __init__(self, kernel, fc_input_len):
        self.kernel = kernel
        self.fc_weights = [rand_float(-1, 1) for _ in range(fc_input_len)]
        self.bias = rand_float(-1, 1)

    def forward(self, image):
        self.conv_output = convolve2d(image, self.kernel)
        self.relu_output = [[relu(v) for v in row] for row in self.conv_output]
        self.flattened = flatten(self.relu_output)
        self.z = (
            sum([w * x for w, x in zip(self.fc_weights, self.flattened)]) + self.bias
        )
        self.output = sigmoid(self.z)
        return self.output

    def backpropagate(self, target, learning_rate):
        error = self.output - target
        d_output = error * sigmoid_derivative(self.output)
        for i in range(len(self.fc_weights)):
            self.fc_weights[i] -= learning_rate * d_output * self.flattened[i]
        self.bias -= learning_rate * d_output


def rand_float(a, b):
    return a + (rand_below(1000000) / 1000000) * (b - a)


def rand_below(n):
    if not hasattr(rand_below, "seed"):
        rand_below.seed = 123456789
    a = 1103515245
    c = 12345
    m = 2**31
    rand_below.seed = (a * rand_below.seed + c) % m
    return rand_below.seed % n


def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        images = []
        for _ in range(num_images):
            img = f.read(rows * cols)
            img = [b / 255.0 for b in img]
            img2d = [img[i * cols : (i + 1) * cols] for i in range(rows)]
            images.append(img2d)
        return images


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")
        labels = list(f.read(num_labels))
        return labels


if __name__ == "__main__":
    data_dir = "data"
    images = load_mnist_images(f"{data_dir}/train-images.idx3-ubyte")
    labels = load_mnist_labels(f"{data_dir}/train-labels.idx1-ubyte")

    batch_inputs = []
    batch_labels = []
    for img, lbl in zip(images, labels):
        if lbl in (0, 1):
            batch_inputs.append(img)
            batch_labels.append(lbl)
        if len(batch_inputs) >= 100:
            break

    kernel = [[1, 0], [0, -1]]
    input_size = len(batch_inputs[0])
    kernel_size = len(kernel)
    conv_out_size = input_size - kernel_size + 1
    fc_input_len = conv_out_size**2

    model = CNN(kernel, fc_input_len=fc_input_len)

    losses = []
    accuracies = []

    for epoch in range(3):
        total_loss = 0
        correct = 0
        for input_img, label in zip(batch_inputs, batch_labels):
            output = model.forward(input_img)
            loss = -(
                label * math.log(output + 1e-8)
                + (1 - label) * math.log(1 - output + 1e-8)
            )
            model.backpropagate(label, 0.1)
            total_loss += loss
            prediction = 1 if output >= 0.5 else 0
            correct += int(prediction == label)
        avg_loss = total_loss / len(batch_inputs)
        accuracy = correct / len(batch_inputs)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

    plt.figure(figsize=(12, 10))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.title("Loss Curve", fontsize=28)
    plt.xlabel("Epoch", fontsize=28)
    plt.ylabel("Loss", fontsize=28)
    plt.tick_params(axis="both", labelsize=28)
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.title("Accuracy Curve", fontsize=28)
    plt.xlabel("Epoch", fontsize=28)
    plt.ylabel("Accuracy", fontsize=28)
    plt.tick_params(axis="both", labelsize=28)
    plt.show()
