import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from Rushil_Umaretiya_Lab7P1 import knn, load_data
from math import sqrt


def main():
    train, test = load_data(0.2, 'iris.csv')
    class_attr = 'class'
    x = [knn(train, test, k, class_attr, output=False) for k in range(1, round(sqrt(len(train))))]
    y = list(range(1, len(x) + 1))
    print(len(x) == len(y))


if __name__ == '__main__':
    main()