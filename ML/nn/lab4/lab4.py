import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def nn(data, hidden_layer_sizes=(100,), max_iter=200, learning_rate=0.1, return_model=False):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate, verbose=False, random_state=1)
    clf.fit(data[0], data[1])
    y_pred = clf.predict(data[2])
    
    accuracy = accuracy_score(data[3], y_pred)
    
    if return_model:
        return clf, accuracy
    else:
        return accuracy

def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (X_train, y_train, X_test, y_test)
    
def main():
    data = load_data()
    
    # print("acc(%):", nn(data, hidden_layer_sizes=(3,), max_iter=10))
    
    
    # a)
#     X = list(range(10))
#     y = [nn(data, hidden_layer_sizes=tuple([10]*x), max_iter=200)*100 for x in X]
    
#     plt.plot(X,y)
#     plt.xlabel("# of hidden layers")
#     plt.ylabel("accuracy")
#     plt.show()
    
    # b)
    X = list(range(1,201,10))
    y = [nn(data, hidden_layer_sizes=(100,), max_iter=x)*100 for x in X]
        
    plt.plot(X,y)
    plt.xlabel("MAX_EPOCH")
    plt.ylabel("accuracy")
    plt.show()

    # c)
#     clf, accuracy = nn(data, hidden_layer_sizes=(100,), max_iter=10, learning_rate=.01, return_model=True)
#     print(clf.coefs_[0][0][0])
    
#     y = [-0.029861371725764158,-0.02264642442306576,-0.026370648234539128,-0.03010906327805069,-0.0369403344661864,-0.04425032195146008,-0.04701612022155189,-0.0480744552027113,-0.04924013779012516,-0.04613149635077437]
#     X = list(range(10))
        
#     plt.plot(X,y)
#     plt.xlabel("epoch")
#     plt.ylabel("weight")
#     plt.show()
            
if __name__ == "__main__":
    main()