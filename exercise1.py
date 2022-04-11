import math
import matplotlib.pyplot as plt
import numpy as np


def task1():
    x = np.random.rand(100)
    epsilon = np.random.uniform(low=-0.3, high=0.3, size=100)
    y = np.sin(2*math.pi*x)+epsilon
    data_points = np.dstack((x, y))
    data_points = np.squeeze(data_points)
    return data_points


def stochastic_gradient_descent(iterations, dim, alpha):
    error = []
    data_points = task1()
    theta = np.random.uniform(low=-0.5, high=0.5, size=dim)
    for z in range(0, iterations):
        for i in range(0, 100):
            for j in range(0, dim):
                theta[j] = theta[j] + alpha*(data_points[i][1] - h(theta, data_points[i][0]))*math.pow(data_points[i][0], j)
        error.append(math.sqrt(2*E(data_points, theta, 100)/100))
    return theta, error


def E(data_points, theta, m):
    sum = 0
    for i in range(0, m):
        sum += math.pow(h(theta, data_points[i][0]) - data_points[i][1], 2)
    return 0.5*sum


def h(theta, x):
    sum = 0
    for z in range(theta.size):
        sum = sum + theta[z] * math.pow(x, z)
    return sum


def create_polynomial(theta, precision=100, dim=2):
    x = np.linspace(0, 1, precision)
    y = np.zeros_like(x)
    for i in range(0, y.shape[0]):
        sum = 0
        for j in range(0, dim):
            sum = sum + math.pow(x[i], j)*theta[j]
        y[i] = sum
    return x, y


if __name__ == "__main__":
    ALPHA = 0.1
    DIM = 4
    ITERATIONS = 3000
    data_distr = task1()
    result, error = stochastic_gradient_descent(ITERATIONS, DIM, ALPHA)
    print(result)
    plt.scatter(data_distr[:,0], data_distr[:,1])
    poly_x, poly_y = create_polynomial(theta=result, dim=DIM)
    print(poly_y)
    plt.plot(poly_x, poly_y)
    plt.title(f"Alpha {ALPHA}, DIM {DIM}, ITERATIONS {ITERATIONS}")
    plt.show()

    plt.title("Error over time")
    plt.plot(error, color='red')
    plt.show()
