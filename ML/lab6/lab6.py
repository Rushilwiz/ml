import math
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

filename = 'iris.csv'
needs_discretized = True
class_attr = 'class'
split = .67
classifier = 2

def main():
    # Read CSV
    df = pd.read_csv(filename)
    
    # Randomize Order
    df = df.sample(frac=1)

    # Discretize
    if needs_discretized:
        for col in df:
            if col != class_attr:
                df[col] = pd.qcut(df[col], q=5)
    
    # Split Data
    if split != 1:
        testing = df.head(-math.floor(len(df)*split))
        data = df.head(math.floor(len(df)*split))
    else:
        testing = data = df
    
    # Choose Classifier
    if classifier == 1:
        r1(data, testing)
    elif classifier == 2:
        decision_tree(data, testing)
    else:
        naive_bayes(data, testing)
        
def r1(data, testing):
    # Set up big dictionary
    rules = dict()
    
    for attr in data:
        if attr != class_attr:
            rules[attr] = dict()

    # Loop thru data
    for attr in data:
        if attr != class_attr:
            freq = {v:{c:0 for c in data[class_attr].unique()} for v in data[attr].unique()}
            for i, sample in data.iterrows():
                freq[sample[attr]][sample[class_attr]] += 1
            
            attr_rule = dict()
            error = 0
            for (k,v) in freq.items():
                rule = max(v, key=v.get)
                for c in v:
                    if c != rule:
                        error += v[c]
                attr_rule[k] = rule
            error /= len(data)
            rules[attr] = (attr_rule, error)
    
    # Select best attr
    best_attr = min(rules, key=lambda x: rules[x][1])
    rule = rules[best_attr][0]
    print(f'R1 chose {best_attr}')
    print(print_tree(rule))
    print('---')
    
    confusion = {v:{c:0 for c in data[class_attr].unique()} for v in data[class_attr].unique()}
    
    correct = 0
    for i, row in testing.iterrows():
        confusion[row[class_attr]][rule[row[best_attr]]] += 1
        if row[class_attr] == rule[row[best_attr]]: correct += 1
        
    print("Confusion Matrix")
    
    for (actual,guess) in confusion.items():
        print(guess)
    print()
    print(f'Accuracy: {round((correct/len(testing))*100, 3)}%')


def decision_tree(data, testing):
    print(f'Decision Tree examined {len(data)} samples and built the following tree:', end='')
    rules = recur_tree(data)
    print_tree(rules)
    print('\n---')
    print("Confusion Matrix")
    confusion, correct = {v:{c:0 for c in data[class_attr].unique()} for v in data[class_attr].unique()}, 0
    
    for i, row in testing.iterrows():
        guess = test_tree(row, rules)
        confusion[row[class_attr]][guess] += 1
        if row[class_attr] == guess: correct += 1    
    
    for (actual,guess) in confusion.items():
        print(guess)

    print()
    print(f'Accuracy: {round((correct/len(testing))*100, 3)}%')
    
    # Test with sklearn tree
    dtc = tree.DecisionTreeClassifier()
    x,y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1-split), random_state=0)
    y_pred = dtc.fit(x_train, y_train).predict(x_test)
    print(f'skLearn accuracy: {sum(y_pred == y_test)*100/len(y_pred)}%')

def recur_tree(data):
    rules = {}
    
    # Find info gain per attrT
    info = calc_info(data)
    if info == 0:
        return data[class_attr].unique()[0]
    
    # gain = {attr:sum([info - calc_info(data[data[attr] == v]) for v in data[attr].unique()]) for attr in data if attr != class_attr}
    gain = {attr:0 for attr in data if attr != class_attr}
    for attr in gain:
        for v in data[attr].unique():
            gain[attr] += info - calc_info(data[data[attr] == v])
    
    # Choose highest info gain
    attr = max(gain, key=gain.get)
    if (gain[attr] == 0):    
        return data[class_attr].unique()[0]
    
    # Split data based on values of attr and recur
    rules[attr] = {}
    for v in data[attr].unique():
        rules[attr][v] = recur_tree(data[data[attr] == v])

    return rules
    
def calc_info(data):
    return abs(sum([(count/len(data))*math.log((count/len(data)), 2) for count in data[class_attr].value_counts()]))
    
def print_tree(rules, indent=0):
    if type(rules) != dict: return rules
    
    for key in rules.keys():
        print('\n'+' '*3*indent + f'* {key}', end='')
        s = print_tree(rules[key], indent + 1)
        if s: print(f' --> {s}', end='')
        
    return None

def test_tree(row, rules):
    if type(rules) != dict: return rules
    
    attr = list(rules.keys())[0]
    return test_tree(row, rules[attr][row[attr]])

def naive_bayes(data, testing):
    confusion, correct = {v:{c:0 for c in data[class_attr].unique()} for v in data[class_attr].unique()}, 0
    class_freq = {c:(len(data[data[class_attr] == c])) for c in data[class_attr].unique()}
    for i, row in testing.iterrows():
        probs = {c:(len(data[data[class_attr] == c]))/len(data) for c in data[class_attr].unique()}
    
        for attr in data:
            if attr != class_attr:
                same_value = data[data[attr] == row[attr]]
                for c in class_freq.keys():
                    probs[c] *= len(same_value[same_value[class_attr] == c])/class_freq[c]
        
        guess = max(probs, key=probs.get)
        confusion[row[class_attr]][guess] += 1
        if row[class_attr] == guess: correct += 1
            
    print(f'Naive Bayes examined {len(data)} samples')
    print('---')
    print("Confusion Matrix")
    for (actual,guess) in confusion.items():
        print(guess)
    print()
    print(f'Accuracy: {round((correct/len(testing))*100, 3)}%')
    
    # Test with sklearn GaussianNaiveBayes
    nb = GaussianNB()
    x,y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1-split), random_state=0)
    y_pred = nb.fit(x_train, y_train).predict(x_test)
    print(f'skLearn accuracy: {sum(y_pred == y_test)*100/len(y_pred)}%')
    
if __name__ == '__main__':
    main()
