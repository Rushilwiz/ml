from tarfile import GNUTYPE_SPARSE
from matplotlib import pyplot as plt

import time

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, layers=(784, 128, 64, 10,), learning_rate=0.1):
        self.W = []
        self.b = []
        self.layers = layers
        self.learning_rate = learning_rate
        
        for i in range(len(layers) - 2):
            w = np.random.randn(layers[i], layers[i + 1])
            self.W.append(w / np.sqrt(layers[i]))
        w = np.random.randn(layers[-2], layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
        self.b = [np.random.uniform(-1, 1, (1,layers[i+1])) for i in range(len(layers)-1)]
        
    def sigmoid(self, x_raw, derivative=False):
        if derivative:
            x = self.sigmoid(x_raw)
            return x * (1 - x)
        else:
            x = np.clip(x_raw, -500, 500)
            return 1.0 / (1 + np.exp(-x))
    
    def forward(self, x):
        a = [np.atleast_2d(x)]

        for layer in range(len(self.W)):
            a.append(self.sigmoid(np.dot(a[layer], self.W[layer]) + self.b[layer]))
        return a
        
    def backward(self, x, target, out):
        y = np.zeros((1, 10))
        y[0][int(target)] = 1
        error = y - out[-1]

        e = [error * self.sigmoid(out[-1], derivative=True)]

        for layer in range(len(self.W) - 1, 0, -1):
            e.append(np.dot(e[-1], self.W[layer].T) * self.sigmoid(out[layer], derivative=True))
        
        e.reverse()
        for layer in range(len(self.W)):
            self.W[layer] += self.learning_rate * np.dot(out[layer].T, e[layer])
            self.b[layer] += self.learning_rate * e[layer]
        
        return np.sum(np.square(error))


    def partial_fit(self, x, target):
        out = self.forward(x)
        loss = self.backward(x, target, out)
        return loss

    def accuracy(self, X, y):
        predictions = []

        for k in range(X.shape[0]):
            out = self.forward(X[k])
            pred = np.argmax(out[-1])
            predictions.append(pred == int(y[[k]]))
        return np.mean(predictions)

    def fit(self, X, y, X_test, y_test, epochs=1000):
        accuracy = []
        losses = []
        for epoch in range(epochs):
            start = time.time()
            loss_sum = 0
            for k in range(len(X)):
                if k % 10000 == 0:
                    print(f'{k} elements seen...')

                loss = self.partial_fit(X[k], y[k])
                loss_sum += loss
            
            losses.append(loss_sum / len(X))
                
            acc = self.accuracy(X_test, y_test)
            accuracy.append(acc)
            end = time.time()
            print("Epoch: {}, Accuracy: {}%".format(epoch, acc*100))
            print("Time: {}".format(end - start))
            print()
        
        return accuracy, losses

def main():
    epochs = 10
    
    test = pd.read_csv('mnist_test.csv')
    train = pd.read_csv('mnist_train.csv')
        
    print("loading data...")
    X_train = train.iloc[:,1:].to_numpy()
    y_train = train.iloc[:,0].to_numpy()

    X_test = test.iloc[:,1:].to_numpy()
    y_test = test.iloc[:,0].to_numpy()
    print("data loaded!")
    print()
    
    #nn = neural_network(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    nn = NeuralNetwork()
    accuracy, loss = nn.fit(X_train, y_train, X_test, y_test, epochs=epochs)

    plt.plot(list(range(1,epochs+1)),accuracy)
    plt.title("accuracy vs epochs")
    plt.show()

    plt.plot(list(range(1,epochs+1)),loss)
    plt.title("loss vs epochs")
    plt.show()

if __name__ == '__main__':
    main()
