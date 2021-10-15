# ---------------------------------Imports------------------------------
import math
from matplotlib.pyplot import clf
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
# --------------------------------Functions-----------------------------



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

def getProbs(data):
    probs = []
    count = 0
    for i in range(len(data) - 1):
        for j in range(11):
            
        

data = getData("TrainingData.txt") #Training Data Stored

ben, mal = divideData(data)

