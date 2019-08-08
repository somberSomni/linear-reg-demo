import numpy as np
from numpy.linalg import linalg
import math

x = np.arange(0, 100).reshape((100, 1))
allOnes = np.ones((100, 1))
y = np.array([3*n + 10 for n in range(0,100)])
X = np.concatenate((allOnes, x), axis=1)

def hyp(theta, X):
    return np.matmul(X, theta)


def computeCost(theta, X, y):
    return np.sum((hyp(theta, X) - y) ** 2) / (2 * len(y))

def gradientDescent(theta, X, y, learning_rate):
    gradient = np.sum(np.multiply(X, np.sum(hyp(theta, X) - y)), axis=0) / len(y)
    print(learning_rate * gradient)
    return theta - learning_rate * gradient

def train(X,y, iterations):
    #initialize theta
    theta = np.array([10,0])
    epsilon = 0.001
    for i in range(iterations):
        cost = computeCost(theta, X, y)
        theta = gradientDescent(theta,X, y, 0.000001)
        print(cost, theta, i)
        if cost < epsilon:
            break

train(X,y,200)