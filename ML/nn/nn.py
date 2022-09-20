import random
import math
from matplotlib import pyplot as plt

points = 20
test_points = int(points/4)

def main():
    # generate test slope
    slope = random.uniform(0.5, 2)
    
    # generate data
    X,y,t = gen_data(slope)
    
    plt.title("dataset")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(X,y,c=t,cmap='brg')
    plt.axline((0,0), slope=slope)
    plt.show()
    
    # train perceptron
    w,b = perceptron(X,y,t)
    
    print("final: ", w, b)
    plt.title("prediction")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(X,y,c=t,cmap='brg')
    plt.axline((0,-b/w[1]), slope=(-w[0]/w[1]))
    
    print("acc(%):", test(w,b,slope)*100)

def gen_data(slope):
    X = [random.uniform(0 ,1) for x in range(points)]
    y = [random.uniform(0 ,1) for x in range(points)]
    t = [0 if (y[i]/X[i] < slope) else 1 for i in range (points)]
    
    return X,y,t

def activation(n):
    # return 1 / (1 + math.exp(-n)) # log-sigmoid
    return 1 if n > 0 else 0 # hard limit

def perceptron(X,y,t):
    b = random.uniform(0,1)
    w = [random.uniform(-1,1), random.uniform(-1,1)]
    print("initial: ", w,b)
    
    learning_rate = 1
    epochs = 10
    
    for i in range(epochs):
        for j in range(points):

            # calculate the sum
            n = X[j]*w[0] + y[j]*w[1] + b

            e = t[j] - activation(n)
            
            w[0] = w[0] + learning_rate*X[j]*e
            w[1] = w[1] + learning_rate*y[j]*e
            b = b + learning_rate*e
    
    return w,b

def test(w,b,slope):
    
    
    X = [random.uniform(0 ,1) for x in range(test_points)]
    y = [random.uniform(0 ,1) for x in range(test_points)]
    t = [0 if (y[i]/X[i] < slope) else 1 for i in range (test_points)]
    
    correct = 0
    
    for j in range(test_points):
        n = X[j]*w[0] + y[j]*w[1] + b
        e = t[j] - activation(n)
        
        if e == 0: correct += 1
    
    return correct/test_points
    
    
if __name__ == "__main__":
    main()