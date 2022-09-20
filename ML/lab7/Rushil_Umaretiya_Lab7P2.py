import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():
    filename = 'iris.csv'
    class_attr = 'class'
    k = 10
    split = 0.2
    
    train, test = load_data(split)

    sklearn_knn(train, test, k, class_attr)

def sklearn_knn(test, train, k, class_attr):
    attributes = [i for i in train.columns.values if i != class_attr]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train[attributes], train[class_attr])
    acc = round(knn.score(test[attributes], test[class_attr])*100, 3)
    print(f'sklearn Accuracy: {acc}%')
    return acc

def load_data(split, filename='iris.csv'):
    df = pd.read_csv(filename)
    train, test = train_test_split(df, test_size=split)
    return train, test

if __name__ == '__main__':
    main()