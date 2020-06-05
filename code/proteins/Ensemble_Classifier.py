from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Ensemble:

    def __init__(self):
        self.classifiers = [MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=5000), SVC(), DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier()]

    def fit(self, X_train, y_train):
        for clf in self.classifiers:
            clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = []
        for clf in self.classifiers:
            predictions.append(clf.predict(X_test))
        import copy
        prediction = copy.deepcopy(predictions[0])
        for i in range(len(prediction)):
            bucket = dict()
            for p in predictions:
                if p[i] in bucket:
                    bucket[p[i]] += 1
                else:
                    bucket[p[i]] = 1
            p , v = None, float("-inf")
            for b in bucket:
                if bucket[b] > v:
                    v = bucket[b]
                    p = b
            prediction[i] = p
        return prediction
