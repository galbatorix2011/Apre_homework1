"""
###################
Group 132
###################

Afonso Almeida 92790
Duarte Elvas 98564

"""
# ---------------------------------Imports------------------------------
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# --------------------------------Functions-----------------------------

def getVectors(indice):
    res = []
    for line in data:
        res.append(line[indice])
    return res


def getData(fileName):
    res = []
    #Reads the file
    with open(fileName) as f: 
        lines = f.readlines()
    #Formats all lines
    for line in lines:
        tmp = line[:-1].split(",")
        if '?' in line:
            continue
        res.append([int(tmp[i]) if i < len(tmp) - 1 else tmp[i] for i in range(len(tmp))])
    return res

# ------------------------------Global-Variables---------------------------
 
def getDistance(point1, point2):
    sum = 0
    for i in range(0, len(point1) - 1):
        sum += (point1[i] - point2[i])**2
    return math.sqrt(sum)

def getKnn(point1, indexes):
    nns = [[-1, -1] for i in range(k)] # -1 when there is nothing there
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
            for i in range(len(nns)):
                if dist < nns[i][0]:
                    nns[i][0] = dist
                    nns[i][1] = point2
    return nns

def getClassification(point, indexes):
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

def main():
    data = getData("TrainingData.txt") #Training Data Stored
    size = int(input("testSize: "))
    k = int(input('k: '))

    kf = KFold(n_splits = 10, random_state = 132, shuffle = True)
    numRightClassifications = 0
    numClassifications = len(data) 


    for train_index , test_index in kf.split(data):
        print(test_index)
        for index in test_index:
            point = data[index];
            pointClass = point[-1]
            if getClassification(point, train_index) == pointClass:
                numRightClassifications += 1

    print('Accuracy = ' + str(numRightClassifications / numClassifications))
    

if __name__ == '__main__':
    main()
    