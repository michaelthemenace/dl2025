def gradient_descent(f, f_prime, x, r, epsilon):
    fx = f(x)
    while fx > epsilon:
        x = x - r * f_prime(x)
        fx = f(x)
        print(f"Current x: {x}, f(x): {fx}")
    return [x, fx]


if __name__ == "__main__":

    def f(x):
        return x**4

    def f_prime(x):
        return 4 * x**3

    x = float(input("Enter initial x: "))
    r = float(input("Enter learning rate r: "))
    epsilon = float(input("Enter convergence threshold epsilon: "))
    result = gradient_descent(f, f_prime, x, r, epsilon)
    print(f"Final x: {result[0]}, Final f(x): {result[1]}")
