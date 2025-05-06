import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(x1i, x2i, w1, w2, w0):
    return w1 * x1i + w2 * x2i + w0


def cal_BSE_single_loss(yi, yi_pred, epsilon=1e-15):
    sig = sigmoid(yi_pred)
    sig = min(max(sig, epsilon), 1 - epsilon)
    return -(yi * math.log(sig) + (1 - yi) * math.log(1 - sig))


def cal_focal_loss_single(yi, yi_pred, gamma=2, alpha=0.25, epsilon=1e-15):
    p = sigmoid(yi_pred)
    p = min(max(p, epsilon), 1 - epsilon)

    if yi == 1:
        pt = p
        alpha_t = alpha
    else:
        pt = 1 - p
        alpha_t = 1 - alpha

    focal_loss = -alpha_t * (1 - pt) ** gamma * math.log(pt)
    return focal_loss


def cal_new_w0(w0, r, yi, yi_pred):
    return w0 - r * (1 - yi - sigmoid(-yi_pred))


def cal_new_w1(w1, r, x1i, yi, yi_pred):
    return w1 - r * (-yi * x1i + x1i * (1 - sigmoid(-yi_pred)))


def cal_new_w2(w2, r, x2i, yi, yi_pred):
    return w2 - r * (-yi * x2i + x2i * (1 - sigmoid(-yi_pred)))


def logistic_regression(file_path, r=0.5, w0=0, w1=0, w2=0):

    with open(file_path, "r") as file:
        lines = file.readlines()

    n = len(lines) - 1
    total_loss = 0

    for i in range(1, n):
        x1i, x2i, yi = map(float, lines[i].strip().split(","))
        yi_pred = predict(x1i, x2i, w1, w2, w0)
        w0 = cal_new_w0(w0, r, yi, yi_pred)
        w1 = cal_new_w1(w1, r, x1i, yi, yi_pred)
        w2 = cal_new_w2(w2, r, x2i, yi, yi_pred)
        Li = cal_BSE_single_loss(yi, yi_pred)
        print(f"{i}th iteration:")
        print(f"w0: {w0}, w1: {w1}, w2: {w2}, yi_pred: {sigmoid(yi_pred)} loss: {Li}")
        total_loss += Li

    final_loss = total_loss / n
    print(f"Final w0: {w0}, Final w1: {w1}, Final w2: {w2}, Final loss: {final_loss}")

    return final_loss


if __name__ == "__main__":
    file_path = "/home/mike/Documents/USTH/Deep-Learning/dl2024/loan.csv"
    logistic_regression(file_path)
