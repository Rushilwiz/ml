from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    # NUMBER OF MAX EPOCHS
    epochs = 80
    
    # Load data from csv
    test = pd.read_csv('mnist_test.csv')
    train = pd.read_csv('mnist_train.csv')
    
    # Split data to test and train, X and y
    X_train = train.iloc[:,1:].to_numpy()
    y_train = train.iloc[:,0].to_numpy()

    X_test = test.iloc[:,1:].to_numpy()
    y_test = test.iloc[:,0].to_numpy()

    
    # Initialize Multi-Perceptron Classifier for partial fitting
    clf = MLPClassifier(hidden_layer_sizes=(128,64,), max_iter=epochs, learning_rate_init=0.001, verbose=True, random_state=1, early_stopping=False)
    
    # Initialize statistic lists
    X = list(range(1,epochs+1))
    y_loss, y_acc = [], []
    
    for k in range(epochs):
        print(f"Epoch {k}")
        
        # Partial fit dataset by doing forward pass and then backwards pass
        clf = clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
        
        # Add loss and accuracy values to statistics
        y_loss.append(clf.loss_)
        y_acc.append(clf.score(X_test, y_test))
    
    print(f"accuracy: {clf.score(X_test,y_test)*100}%")
    
    # Plot epochs vs loss
    plt.plot(X,y_loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    
    # Plot epochs vs accuracy
    plt.plot(X,y_acc)
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.show()

if __name__ == '__main__':
    main()