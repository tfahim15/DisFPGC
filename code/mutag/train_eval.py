import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from Ensemble_Classifier import Ensemble
import math
freq = open("cover_freq.txt").readlines()
file_name = "mutag"
# loading labels of graphs from file
label = open("../../data/"+file_name+"/"+file_name+"_label.txt").readlines()
labels = []
for line in label:
    labels.append(int(line.split("\t")[0]))
labels = np.array(labels)

codes_count = 0
codes = dict()

for i in range(len(freq)):
    if freq[i][0] == "#":
        continue
    if freq[i] == "\n":
        codes_count += 1
    else:
        l = freq[i].replace("\n", "").split(" ")
        graph_id, iso_count = int(l[0]), int(l[1])
        if codes_count not in codes:
            codes[codes_count] = dict()
        codes[codes_count][graph_id] = iso_count


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def entropy(bucket):
    total = 0
    for i in bucket:
        total += bucket[i]
    res = 0
    for i in bucket:
        p = bucket[i]/total
        res -= p * math.log2(p)
    return res


def gain(subgraph, labels):
    positive_bucket = dict()
    negative_bucket = dict()
    positive, negative = 0, 0
    for i in range(len(labels)):
        if i in subgraph:
            positive += 1
            if labels[i] in positive_bucket:
                positive_bucket[labels[i]] += 1
            else:
                positive_bucket[labels[i]] = 1
        else:
            negative += 1
            if labels[i] in negative_bucket:
                negative_bucket[labels[i]] += 1
            else:
                negative_bucket[labels[i]] = 1
    try:
        split = -len(subgraph)/len(labels) * math.log2(len(subgraph)/len(labels))
    except ValueError:
        split = 9999999999
    try:
        split -= (len(labels) - len(subgraph)) / len(labels) * math.log2((len(labels) - len(subgraph)) / len(labels))
    except ValueError:
        split = 9999999999
    return ((positive/(positive+negative)) * entropy(positive_bucket) + (negative/(positive+negative)) * entropy(negative_bucket))/split


# filters codes by gain value
def filter(codes, labels, min_cov):
    code_list = []
    for c in codes:
        subgraph = []
        for i in range(len(train_index)):
            if train_index[i] in codes[c]:
                subgraph.append(i)
        info_ = gain(subgraph, labels)
        code_list.append((c, info_))
    code_list = sorted(code_list, key=lambda tup: tup[1])
    coverage = [0]*len(y)
    ret, r = dict(), 0
    for i in range(len(code_list)):
        c = code_list[i][0]
        picked = False
        for g in codes[c]:
            if coverage[g] < min_cov:
                coverage[g] += 1
                picked = True
        if picked:
            ret[r] = codes[c]
            r += 1
    return ret


dimension = 64
alpha = 1
min_cov = 10

total_graphs = len(labels)
n_splits = 10
total_accuracy = 0
kf = StratifiedKFold(n_splits=n_splits, random_state=27323, shuffle=True)
y = labels
fold = 0
for train_index, test_index in kf.split([i for i in range(len(labels))], y):
    fold += 1
    print("Fold:", fold)
    print("\tFiltering subgraphs...")
    t_codes = filter(codes, y[train_index], min_cov)
    total_codes = len(t_codes)
    W = np.random.normal(0.5, 0.1, (total_codes, dimension))
    W_ = np.random.normal(0.5, 0.1, (dimension, total_graphs))
    print("\tTraining Embedding...")
    # trains embedding
    for i in range(10):
        print("\t\tepoch:", i+1)
        loss = 0
        for j in range(total_graphs):
            x = np.zeros(total_codes)
            w = 0
            for code_id in t_codes:
                if j in codes[code_id]:
                    x[code_id] = 1.0
                    w += 1.0
            if w != 0:
                x /= w
            else:
                continue
            h = np.dot(np.transpose(W), x)
            u = np.dot(np.transpose(W_), h)
            y = softmax(u)
            t = np.zeros(total_graphs)
            t[j] = 1
            e = y-t
            for j in e:
                loss += j*j
            dW_ = np.outer(h, e)
            dW = np.outer(x, np.dot(W_, e))
            W -= alpha*dW
            W_ -= alpha*dW_

    W_ = np.transpose(W_)
    X = W_
    y = labels
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = Ensemble()
    # trains classifier
    print("\tTraining classifier...")
    clf.fit(X_train, y_train)
    # predicts classes
    predict_test = clf.predict(X_test)
    error_count = 0
    error_set = {}
    for i in range(y_test.__len__()):
        if predict_test[i] != y_test[i]:
            error_count = error_count + 1
        if y_test[i] not in error_set:
            error_set[y_test[i]] = 1
        else:
            error_set[y_test[i]] = error_set[y_test[i]] + 1
    total_accuracy += (1 - (error_count / y_test.__len__())) * 100
    print('\tAccuracy: ' + str((1 - (error_count / y_test.__len__())) * 100) + '%')
    print()

print("Average accuracy:", total_accuracy/n_splits)
