from numpy import array
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

def divide(data):  # will divide the data in order to be processed by the MultinomialNB function
    X = []
    y = []
    for line in data:
        X.append(line[:-1])
        y.append(line[-1])
    return (X, y)

def getAccuracy(predictions, real):
    return sum([1 if predictions[i] == real[i] else 0 for i in range(
            len(predictions))]) / len(predictions) 

def getMeanAccuracy(accuracies):
    return sum(accuracies) / len(accuracies)


data = [list(e) for e in arff.loadarff("kin8nm.arff")[0]]

clf = MLPRegressor(hidden_layer_sizes=[3,2],activation="relu", early_stopping=True, alpha = 1)

predictions = []
targets = []

kf = KFold(n_splits=5, random_state=0, shuffle=True)

for train_index, test_index in kf.split(data):
    trainingDataIn, trainingDataOut = divide([data[i] for i in train_index])
    testDataIn, testDataOut = divide([data[i] for i in test_index])

    clf.fit(trainingDataIn, trainingDataOut)

    predictions += list(clf.predict(testDataIn))
    targets += testDataOut



print(getAccuracy(predictions, targets))


