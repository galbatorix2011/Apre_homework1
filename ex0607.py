"""
###################
Group 132
###################

Afonso Almeida 92790
Duarte Elvas 98564

"""
# ---------------------------------Imports------------------------------
import math
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
import numpy as np
# --------------------------------Functions-----------------------------
def getVectors(data,indice): #Returns the vectors for a single value
    res = []
    for line in data:
        res.append(line[indice])
    return res

def getData(fileName): #Converts the data in a .txt file to an array
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

def naiveDivide(data): #will divide the data in order to be processed by the MultinomialNB function
    X = []
    y = []
    for line in data:
        X.append(line[:-1])
        y.append(line[-1])
    return (X, y)

def getNaivePrediction(trainData, testData): # Will fit and predict based on the Naive Bayes algorithm
    count = 0
    trainX, trainY = naiveDivide(trainData)
    testX, testY = naiveDivide(testData)
    clf = MultinomialNB()
    clf.fit(trainX, trainY) #train    
    predictions = clf.predict(testX)
    for i in range(len(predictions)):
        if (predictions[i] == testY[i]):
            count += 1
    return count/len(testData)

def getKnn(point1, indexes): # Will fit and predict based on the KNN algorith
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


def getClassificationKnn(point, indexes): #Will end the Knn algorithm
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

def getTotalAccuracy(accuracies): #Returns the accuracies
    return sum(accuracies) / len(accuracies)

#---------------------------------Main------------------------------------------------------------

data = getData("data.txt")  # Training Data Stored
k = int(input('k: ')) #K Input

kf = KFold(n_splits=10, random_state=132, shuffle=True) #Kfold  where the data is divided in Training and Testing
accuracies = [] #Stores the final accuracies
trainAccuracies = [] #Stores the train Accuracies
#----------------------------------------Knn---------------------------------------------------------
for train_index, test_index in kf.split(data): #Main for looop
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

print("Accuracy de teste knn: " + str(getTotalAccuracy(accuracies)))
print("Accuracy de treino knn: " + str(getTotalAccuracy(trainAccuracies)))
print("diff: " + str(getTotalAccuracy(trainAccuracies) - getTotalAccuracy(accuracies)))
#------------------------------------Naive-Bayes----------------------------------------------------
kf = KFold(n_splits=10, random_state=132, shuffle=True)
NaiveBayesAccuracies = []

for train_index, test_index in kf.split(data): #Main for Loop
    trainingData = [data[i] for i in train_index]
    testData = [data[i] for i in test_index]
    NaiveBayesAccuracies.append(getNaivePrediction(trainingData,testData))

print("Accuracy naive bayes: " + str(getTotalAccuracy(NaiveBayesAccuracies)))

pValue = stats.ttest_rel(np.array(NaiveBayesAccuracies), np.array(accuracies), alternative='greater')
print("Pvalue: "+ str(pValue))
