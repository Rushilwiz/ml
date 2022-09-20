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

    knn(train, test, k, class_attr)
    sklearn_knn(train, test, k, class_attr)

def knn(train, test, k, class_attr):
    confusion, correct = {v:{c:0 for c in test[class_attr].unique()} for v in test[class_attr].unique()}, 0

    for index, row in test.iterrows():
        prediction = predict(train, row, k, class_attr)
        confusion[row[class_attr]][prediction] += 1
        if prediction == row[class_attr]: correct += 1

    print(f'KNN examined {len(test)} samples')
    print('---')
    print("Confusion Matrix")
    for (actual,guess) in confusion.items():
        print(guess)
    print()
    print(f'Accuracy: {round((correct/len(test))*100, 3)}%')


def predict(train, point, k, class_attr):
    attributes = [i for i in train.columns.values if i != class_attr]
    return max(list(zip(*sorted([(distance(point[attributes], row[attributes]), row[class_attr]) for index, row in train.iterrows()], key=lambda x: x[0])[:k]))[1])

def distance(test, train):
    return sum([(test[i] - train[i])**2 for i in range(len(test))])**0.5

def sklearn_knn(test, train, k, class_attr):
    attributes = [i for i in train.columns.values if i != class_attr]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train[attributes], train[class_attr])
    print(f'sklearn Accuracy: {round(knn.score(test[attributes], test[class_attr])*100, 3)}%')

def load_data(split):
    df = pd.read_csv('iris.csv')
    train, test = train_test_split(df, test_size=split)
    return train, test

if __name__ == '__main__':
    main()