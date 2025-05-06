def Li(w0, w1, xi, yi):
    prediction = w1 * xi + w0
    loss = 0.5 * (prediction - yi) ** 2
    return loss


def compute_gradients(data, w0, w1):
    dL_dw0 = 0
    dL_dw1 = 0
    total_loss = 0

    for xi, yi in data:
        prediction = w1 * xi + w0
        error = prediction - yi
        dL_dw0 += error
        dL_dw1 += error * xi
        total_loss += 0.5 * error**2

    n = len(data)
    return dL_dw0 / n, dL_dw1 / n, total_loss / n


def linear_regression(data, w0, w1, r, epsilon):
    while True:
        dL_dw0, dL_dw1, loss = compute_gradients(data, w0, w1)
        if loss < epsilon:
            break
        w0 -= r * dL_dw0
        w1 -= r * dL_dw1
        print(f"Current w0: {w0}, Current w1: {w1}, Loss: {loss}")
    return w0, w1, loss


if __name__ == "__main__":
    file_path = "/home/mike/Documents/USTH/Deep-Learning/dl2024/lr.csv"

    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []

    for line in lines:
        xi, yi = map(float, line.strip().split(","))
        data.append((xi, yi))

    r = float(input("Enter learning rate r: "))
    epsilon = float(input("Enter convergence threshold epsilon: "))
    w0 = float(input("Enter initial w0: "))
    w1 = float(input("Enter initial w1: "))

    w0, w1, final_loss = linear_regression(data, w0, w1, r, epsilon)
    print(f"Final w0: {w0}, Final w1: {w1}, Final loss: {final_loss}")
