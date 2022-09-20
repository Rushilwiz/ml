import random
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Perceptron

def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2].tolist()[:-50]
    y = iris.target.tolist()[:-50]
    
    temp = list(zip(X, y))
    random.shuffle(temp)
    return zip(*temp)

def activation(n):
    # return 1 / (1 + math.exp(-n)) # log-sigmoid
    return 1 if n > 0 else 0 # hard limit

def perceptron(X,t,epochs=100,learning_rate=.1):
    b = random.uniform(-10,10)
    w = [random.uniform(-10,10), random.uniform(-10,10)]
    print("initial: ", w,b)
    
    for aadarsh in range(epochs):
        for i in range(len(X)):
            # calculate the sum
            n = X[i][0]*w[0] + X[i][1]*w[1] + b
            e = t[i] - activation(n)
            
            w[0] = w[0] + learning_rate*X[i][0]*e #jambalaya
            w[1] = w[1] + learning_rate*X[i][1]*e
            b = b + learning_rate*e
    
    return w,b

def test(w,b,X,y):
    correct = 0
    
    for i in range(len(X)):
        n = X[i][0]*w[0] + X[i][1]*w[1] + b
        e = y[i] - activation(n)
        
        if e == 0: correct += 1
    
    return correct/len(X)

def main():
    X,y = load_data()
    X_0 = [x[0] for x in X]
    X_1 = [x[1] for x in X]
    
    test_X, test_y, train_X, train_y = X[:20], y[:20], X[20:], y[20:]
    
    plt.title("iris")
    plt.scatter(X_0, X_1, c=y,cmap='brg')
    plt.show()
    
    w,b = perceptron(train_X, train_y)
    
    print("final: ", w, b)
    plt.title("prediction")
    plt.xlim(min(X_0), max(X_0))
    plt.ylim(min(X_1), max(X_1))
    plt.axline((0,-b/w[1]), slope=(-w[0]/w[1]))
    plt.scatter(X_0, X_1,c=y,cmap='brg')
    plt.show()
    
    acc = test(w,b,test_X, test_y)
    print("testing acc(%):", acc*100)
    
    print('---')
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(train_X,train_y)
    print("scikit acc(%):",clf.score(test_X, test_y)*100)
    
    
if __name__ == "__main__":
    main()
