"""
###################
Group 132
###################

Afonso Almeida 92790
Duarte Elvas 98564

"""
# ---------------------------------Imports------------------------------
import math
from matplotlib.pyplot import clf
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
from scipy.stats import norm
import numpy as np
# -------------------------------- naive bayes Functions-----------------------------

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

def getClassificationNaiveBayes(point, benProbs, malProbs): #[2,5,1,7]
    benProbTemp = len(benProbs) / (len(benProbs) + len(malProbs))
    malProbTemp = len(malProbs) / (len(benProbs) + len(malProbs))
    for i in range(len(point) - 1):
        benProbTemp *= benProbs[i][point[i] - 1]
        malProbTemp *= malProbs[i][point[i] - 1]
    return "benign" if benProbTemp >= malProbTemp else "malignant"

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

# ------------------------------Global-Variables---------------------------


def getDistance(point1, point2):
    sum = 0
    for i in range(0, len(point1) - 1):
        sum += (point1[i] - point2[i])**2
    return math.sqrt(sum)


def getKnn(point1, indexes):
    nns = [[-1, -1] for i in range(k)]  # -1 when there is nothing there
    points = [data[i] for i in indexes]
    empty = k
    for point2 in points:
        dist = math.dist(point1[:-1], point2[:-1])
        if empty != 0:
            for i in range(len(nns)):
                if nns[i][0] == -1:
                    nns[i][0] = dist
                    nns[i][1] = point2
                    empty -= 1
                    break
        else:   
            maxDist = -1 
            index = -1
            for i in range(len(nns)):
                if dist < nns[i][0] and nns[i][0] > maxDist:
                    maxDist = nns[i][0]
                    index = i
            if maxDist != -1:
                nns[index][0] = dist
                nns[index][1] = point2
    return nns


def getClassificationKnn(point, indexes):
    nns = getKnn(point, indexes)
    neighbours = [element[1][-1] for element in nns]
    numMalignant = 0
    numBenign = 0
    for neighbour in neighbours:
        if neighbour == 'malignant':
            numMalignant += 1
        else:
            numBenign += 1
    return 'malignant' if numMalignant > numBenign else 'benign'

def getTotalAccuracy(accuracies):
    return sum(accuracies) / len(accuracies)

data = getData("TrainingData.txt")  # Training Data Stored
k = int(input('k: '))

kf = KFold(n_splits=10, random_state=132, shuffle=True)
accuracies = []
trainAccuracies = []

for train_index, test_index in kf.split(data):
    numClassifications = len(test_index)
    numRightClassifications = 0
    for index in test_index:
        point = data[index]
        pointClass = point[-1]
        if getClassificationKnn(point, train_index) == pointClass:
            numRightClassifications += 1
    accuracies.append(numRightClassifications / numClassifications)

    totalTrain = len(train_index)
    rightTrain = 0
    for index in train_index:
        point = data[index]
        pointClass = point[-1]
        if getClassificationKnn(point, train_index) == pointClass:
            rightTrain += 1
    trainAccuracies.append(rightTrain / totalTrain)

print("Accuracy de teste: " + str(getTotalAccuracy(accuracies)))
print("Accuracy de treino: " + str(getTotalAccuracy(trainAccuracies)))

kf = KFold(n_splits=10, random_state=132, shuffle=True)
NaiveBayesAccuracies = []


for train_index, test_index in kf.split(data):
    numClassifications = len(test_index)
    numRightClassifications = 0

    trainingData = [data[i] for i in train_index]
    ben, mal = divideData(trainingData)
    benProbs = getProbs(ben, 0)
    malProbs = getProbs(mal, 0)

    for index in test_index:
        point = data[index]
        pointClass = point[-1]
        if getClassificationNaiveBayes(point, benProbs, malProbs) == pointClass:
            numRightClassifications += 1
    NaiveBayesAccuracies.append( numRightClassifications / numClassifications)

print("Accuracy naive bayes: " + str(getTotalAccuracy(NaiveBayesAccuracies)))

pValue = stats.ttest_ind(np.array(accuracies), np.array(NaiveBayesAccuracies), alternative='greater')
print("Pvalue: "+ str(pValue[1]))