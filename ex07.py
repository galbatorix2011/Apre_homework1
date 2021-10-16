# ---------------------------------Imports------------------------------
import math
from matplotlib.pyplot import clf
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
import random
# --------------------------------Functions-----------------------------

def getVectors(data,indice):
    res = []
    for line in data:
        res.append(line[indice])
    return res

def getData(fileName):
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

def divideData(data):
    benign = []
    malign = []

    for line in data:
        if line[-1] == "benign":
            benign.append(line)
        else:
            malign.append(line)
    
    return (benign, malign)

def getProbs(data, residualValue):
    res = [] # each line will have a list with the different probabilities of that feature
    for i in range(9): #Features
        tmp = []
        values = getVectors(data, i)
        for j in range(1,11):
            count = values.count(j)
            tmp.append((count / len(values)) if count != 0 else residualValue) 
        res.append(tmp)
    return res

def getClassification(point, benProbs, malProbs): #[2,5,1,7]
    benProbTemp = len(benProbs) / (len(benProbs) + len(malProbs))
    malProbTemp = len(malProbs) / (len(benProbs) + len(malProbs))
    for i in range(len(point) - 1):
        benProbTemp *= benProbs[i][point[i] - 1]
        malProbTemp *= malProbs[i][point[i] - 1]
    return "benign" if benProbTemp >= malProbTemp else "malignant"


residualValue = 0.0000001

data = getData("TrainingData.txt") #Training Data Stored

kf = KFold(n_splits=10, random_state=132, shuffle=True)
accuracies = []


for train_index, test_index in kf.split(data):
    # print(test_index)
    numClassifications = len(test_index)
    numRightClassifications = 0

    trainingData = [data[i] for i in train_index]
    ben, mal = divideData(trainingData)
    benProbs = getProbs(ben, 0)
    malProbs = getProbs(mal, 0)

    for index in test_index:
        point = data[index]
        pointClass = point[-1]
        if getClassification(point, benProbs, malProbs) == pointClass:
            numRightClassifications += 1
    accuracies.append( numRightClassifications / numClassifications)

print(accuracies)
smm = 0
for x in accuracies:
    smm += x
print(str(smm/len(accuracies)))