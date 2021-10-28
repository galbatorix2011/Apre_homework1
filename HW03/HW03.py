from scipy.sparse.sputils import matrix
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


accuraciesEarlyStoping = []
accuraciesNoEarlyStoping = []

predictionsEarlyStoping = []
predictionsNoEarlyStoping = []

targets = []

for train_index, test_index in kf.split(data):
    trainingDataIn, trainingDataOut = divide([data[i] for i in train_index])
    testDataIn, testDataOut = divide([data[i] for i in test_index])
    
    mlpEarlyStoping = MLPClassifier(hidden_layer_sizes=[3,2],activation="relu", early_stopping=True, alpha = 1)
    mlpNoEarlyStoping = MLPClassifier(hidden_layer_sizes=[3,2],activation="relu", early_stopping=False, alpha = 1)

    mlpEarlyStoping.fit(trainingDataIn, trainingDataOut)
    mlpNoEarlyStoping.fit(trainingDataIn, trainingDataOut)

    predictionEarlyStop = mlpEarlyStoping.predict(testDataIn)
    predictionNoEarlyStop = mlpNoEarlyStoping.predict(testDataIn)

    for i in predictionEarlyStop:
        predictionsEarlyStoping.append(i)
    for j in mlpNoEarlyStoping.predict(testDataIn):
        predictionsNoEarlyStoping.append(j)
    
    for target in testDataOut:
        targets.append(target)

    accuraciesEarlyStoping.append(getAccuracy(predictionEarlyStop, testDataOut))
    accuraciesNoEarlyStoping.append(getAccuracy(predictionNoEarlyStop, testDataOut))

print("#-" * 30) 

print("NoEarlyStoping Accuracy----> " + str(getMeanAccuracy(accuraciesNoEarlyStoping)))
confusionNoEarlyStoping = confusion_matrix(targets, predictionsNoEarlyStoping, labels=["benign","malignant"])
print(confusionNoEarlyStoping)

print("")

print("EarlyStoping Accuracy----> " + str(getMeanAccuracy(accuraciesEarlyStoping)))
confusionEarlyStoping = confusion_matrix(targets, predictionsEarlyStoping, labels=["benign","malignant"])
print(confusionEarlyStoping)

print("#-" * 30) 





