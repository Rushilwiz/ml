from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


import numpy as np
import matplotlib.pyplot as plt

def main():
    seed = 42
    # layer_sizes = (5,)
    # layer_sizes = tuple([5]*i)

    activation = "relu"

    X, y = datasets.make_circles(n_samples=1000, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    clf = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=500, random_state=seed, early_stopping=False)

    # Initialize statistic lists
    X = list(range(1,500+1))
    y_loss, y_acc, train_acc = [], [], []
    
    for k in range(500):
        print(f"Epoch {k}")
        
        # Partial fit dataset by doing forward pass and then backwards pass
        clf = clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
        
        # Add loss and accuracy values to statistics
        y_loss.append(clf.loss_)
        train_acc.append(clf.score(X_train, y_train))
        y_acc.append(clf.score(X_test, y_test))
    
    print(f"accuracy: {clf.score(X_test,y_test)*100}%")
    
    # Plot epochs vs loss
    plt.plot(X,y_loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    # Plot epochs vs accuracy
    plt.plot(X, train_acc, label="train")
    plt.plot(X, y_acc, label="test")
    plt.legend(loc='upper left')
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.title("accuracy vs epochs")
    plt.show()


if __name__ == "__main__":
    main()