import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class SGD:
    def __init__(self, x, y, weights, LR, num_iters, momentum):
        self.x = x
        self.y = y
        self.weights = weights
        self.LR = LR
        self.num_iters = num_iters
        self.momentum = momentum

    def loss(self):
        loss = np.sum((self.x @ self.weights - self.y) ** 2) / (self.y.size)
        return loss

    def gradientDescent(self, type):
        m = self.y.size
        update = 0
        for i in range(self.num_iters):
            if type == 3:
                y_est = np.dot(self.x, self.weights - self.momentum * update)
            else:
                y_est = np.dot(self.x, self.weights)
            grad = (1.0 / m) * np.dot(self.x.T, y_est - self.y)
            if type == 1:
                update = self.LR * grad
            else:
                update = self.LR * grad + self.momentum * update
            self.weights = self.weights - update
        return y_est

    def fit(self):
        loss_prev = 100
        for iter in range(self.num_iters):
            self.gradientDescent(type=3)
            loss = self.loss()
            if loss_prev - loss < 1e-20:
                print(f'Loss:{loss}, number of iterations:{iter}')
                print(self.weights)
                break
            loss_prev = loss
            if iter % 10 == 0:
                print(f'Loss:{loss}, number of iterations:{iter}')
                print(self.weights)


if __name__ == '__main__':
    data = load_iris()
    x = data.data
    y = data.target
    weights = np.random.rand(x.shape[1])
    sgd = SGD(x, y, weights, LR=0.01, num_iters=100, momentum=0.99)
    sgd.fit()
    print('Finish')
