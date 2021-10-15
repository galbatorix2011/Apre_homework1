# ---------------------------------Imports------------------------------
import math
from matplotlib.pyplot import clf
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
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
    res = []
    for i in range(8): #Features
        tmp = []
        values = getVectors(data, i)
        for j in range(1,11):
            count = values.count(j)
            tmp.append( (count /len(values)) if count != 0 else residualValue  )
        res.append(tmp)
    return res
            
        

data = getData("TrainingData.txt") #Training Data Stored

ben, mal = divideData(data)


for i in getProbs(ben, 0.0000001):
    print(i)
    print("")

