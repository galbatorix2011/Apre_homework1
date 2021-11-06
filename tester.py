from numpy import array
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
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

def sumSquaredErrors(predictions, targets):
    return sum([(predictions[i] - targets[i])**2 for i in range(len(targets))])

data = [list(e) for e in arff.loadarff("kin8nm.arff")[0]]
hiddenLayers = [8,6,8]
clfReg = MLPRegressor(hidden_layer_sizes= hiddenLayers,  activation="relu", alpha = 0.0031, learning_rate_init=0.004, early_stopping=True, random_state=0)
clfNoReg = MLPRegressor(hidden_layer_sizes=hiddenLayers,  activation="relu", alpha = 0, random_state=0)

predictionsReg = []
predictionsNoReg = []

targets = []

kf = KFold(n_splits=5, random_state=0, shuffle=True)

for train_index, test_index in kf.split(data):
    trainingDataIn, trainingDataOut = divide([data[i] for i in train_index])
    testDataIn, testDataOut = divide([data[i] for i in test_index])

    clfReg.fit(trainingDataIn, trainingDataOut)
    clfNoReg.fit(trainingDataIn, trainingDataOut)

    predictionsReg += list(clfReg.predict(testDataIn))
    predictionsNoReg += list(clfNoReg.predict(testDataIn))
    
    targets += testDataOut

residualsReg = [predictionsReg[i] - targets[i] for i in range(len(targets))]
residualsNoReg = [predictionsNoReg[i] - targets[i] for i in range(len(targets))]

print(sumSquaredErrors(predictionsReg, targets))
print(sumSquaredErrors(predictionsNoReg, targets))
fig = plt.figure(figsize =(10, 7))
 
plt.boxplot([residualsReg,residualsNoReg])

plt.xticks([1, 2], ["Residuals with Regularization", 'Residuals with no Regularization'])
# show plot
#plt.show()

