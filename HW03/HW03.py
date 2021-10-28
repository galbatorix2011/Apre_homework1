from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def getData(fileName):  # Converts the data in a .txt file to an array
    res = []
    # Reads the file
    with open(fileName) as f:
        lines = f.readlines()
    # Formats all lines
    for line in lines:
        tmp = line[:-1].split(",")
        if '?' in line:
            continue
        res.append([int(tmp[i]) if i < len(tmp) - 1 else tmp[i]
                    for i in range(len(tmp))])
    return res

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

data = getData("data.txt")  # Training Data Stored
kf = KFold(n_splits=5, random_state=0, shuffle=True)

accuracys010 = []
accuracys032 = []
accuracys1 = []
accuracys316 = []
accuracys10 = []



for train_index, test_index in kf.split(data):
    trainingDataIn, trainingDataOut = divide([data[i] for i in train_index])
    testDataIn, testDataOut = divide([data[i] for i in test_index])

    mlp010 = MLPClassifier(hidden_layer_sizes=[3,2],activation="relu", early_stopping=True, alpha = 0.10)
    mlp032 = MLPClassifier(hidden_layer_sizes=[3,2],activation="relu", early_stopping=True, alpha = 0.32)
    mlp1 = MLPClassifier(hidden_layer_sizes=[3,2],activation="relu", early_stopping=True, alpha = 1)
    mlp316 = MLPClassifier(hidden_layer_sizes=[3,2],activation="relu", early_stopping=True, alpha = 3.16)
    mlp10 = MLPClassifier(hidden_layer_sizes=[3,2],activation="relu", early_stopping=True, alpha = 10)

    mlp010.fit(trainingDataIn, trainingDataOut)
    mlp032.fit(trainingDataIn, trainingDataOut)
    mlp1.fit(trainingDataIn, trainingDataOut)
    mlp316.fit(trainingDataIn, trainingDataOut)
    mlp10.fit(trainingDataIn, trainingDataOut)

    prediction010 = mlp010.predict(testDataIn)
    prediction032 = mlp032.predict(testDataIn)
    prediction1 = mlp1.predict(testDataIn)
    prediction316 = mlp316.predict(testDataIn)
    prediction10 = mlp10.predict(testDataIn)

    accuracys010.append(getAccuracy(prediction010, testDataOut))
    accuracys032.append(getAccuracy(prediction032, testDataOut))
    accuracys1.append(getAccuracy(prediction1, testDataOut))
    accuracys316.append(getAccuracy(prediction316, testDataOut))
    accuracys10.append(getAccuracy(prediction10, testDataOut))




print("0.010 --> " + str(getMeanAccuracy(accuracys010)))
print("0.32 --> " + str(getMeanAccuracy(accuracys032)))
print("1 --> " + str(getMeanAccuracy(accuracys1)))
print("3.16 --> " + str(getMeanAccuracy(accuracys316)))
print("10 --> " + str(getMeanAccuracy(accuracys10)))
#print(matrix)




