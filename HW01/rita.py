import numpy as np
from numpy.ma.core import count
from scipy.stats import norm
from scipy.stats import multivariate_normal
import math




def populate():
    res = []
    res.append([0.6, 0, 0.2, 0.4, 0]) #x1
    res.append([0.1, 1, -0.1, -0.4, 0]) #x2
    res.append([0.2, 0, -0.1, 0.2, 0]) #x3 
    res.append([0.1, 2, 0.8, 0.8, 0]) #x4
    res.append([0.3, 1, 0.1, 0.3, 1]) #x5
    res.append([-0.1, 2, 0.2, -0.2, 1]) #x6
    res.append([0.3, 2, -0.1, 0.2, 1]) #x7
    res.append([0.2, 1, 0.5, 0.6, 1]) #x8
    res.append([0.4, 0, -0.4, -0.7, 1]) #x9
    res.append([-0.2, 2, 0.4, 0.3, 1]) #x10

    return res

def getMean(index, data):
    count = 0
    for x in data:
        count += x[index]
    return count / 10

def getDesvio(index, data, mean):
    count = 0
    for x in data:
        count += (x[index] - mean)**2
    return math.sqrt((1/9) * count)

def mercyCovas(index1, index2, data, mean1, mean2):
    count = 0
    for x in data:
        count += (x[index1] - mean1) * (x[index2] - mean2)
    return (1 / 9) * count

def getProbsY1(y1, data):
    mean = getMean(0, data)
    print("media de y1: " + str(mean))
    deviation = getDesvio(0, data, mean)
    print("desvio padrao de y1: " + str(deviation))
    return norm.pdf(y1, loc=mean, scale=deviation)

def getProbsY2(y1):
    return 3 / 10 if y1 == 0 else 3 / 10 if y1 == 1 else 4 / 10

def getProbsY3Y4(y3, y4, data):
    meanY3 = getMean(2, data)
    meanY4 = getMean(3, data)
    means = [meanY3, meanY4]
    print("media de y4: " + str(means))
    e00 = mercyCovas(2, 2, data, meanY3, meanY3)
    e01 = mercyCovas(2, 3, data, meanY3, meanY4)
    e11 = mercyCovas(3, 3, data, meanY4, meanY4)
    covs = [[e00, e01], [e01, e11]]
    print("desvios padrao de y3: " + str(covs))
    return multivariate_normal(means, covs).pdf([y3, y4])

def getClassifcation(y1, y2, y3, y4, threshold, data):
    c = 1
    pC = 0.4 if c == 0 else 0.6
    pY1C = norm.pdf(y1, loc=0.25, scale=0.2380) if c == 0 else norm.pdf(y1, loc=0.05, scale=0.2881)
    if c == 0:
        pY2C = 0.5 if y2 == 0 else 0.25 if y2 == 1 else 0.25
    else:
        pY2C = 1/6 if y2 == 0 else 1/3 if y2 == 1 else 0.5
    var0 = multivariate_normal([0.20, 0.25], [ [0.18, 0.18], [0.18,0.25] ] )
    var1 = multivariate_normal([0.117,0.083], [[0.1097,0.1223],[0.1223,0.2137]])
    pY3Y4C = var0.pdf([y3,y4]) if c == 0 else var1.pdf([y3,y4])
    pY1 = getProbsY1(y1, data)
    pY2 = getProbsY2(y2)
    pY3Y4 = getProbsY3Y4(y3, y4, data)
    probscond = (pC*pY1C*pY2C*pY3Y4C)
    prob = probscond / (pY1*pY2*pY3Y4)
    return 0 if prob < threshold else 1

def getMatrix(data):
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    results = [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] for i in range(10)]
    for j in range(len(thresholds)):
        for i in range(len(results)):
            classifcation = getClassifcation(data[i][0], data[i][1], data[i][2], data[i][3], thresholds[j], data)
            results[i][j] = 'TP' if (classifcation == 1 and data[i][-1] == 1) else 'TN' if (classifcation == 0 and data[i][-1] == 0) else 'FN' if (classifcation == 0 and data[i][-1] == 1) else 'FP'
    return results

def getRates(matrix):
    rates = [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
    for i in range(10):
        countFP = 0
        countTP = 0
        for line in matrix:
            if line[i] == 'FP':
                countFP += 1
            elif line[i] == 'TP':
                countTP += 1 
        rates[0][i] = countTP / 6
        rates[1][i] = countFP / 4
    return matrix + rates

def main():
    dataArray = populate()
    
    c = eval(input("Classe: "))

    pC = 0.4 if c == 0 else 0.6
    y1 = eval(input("y1: "))
    y2 = eval(input("y2: "))
    y3 = eval(input("y3: "))
    y4 = eval(input("y4: "))
    
    pY1 = norm.pdf(y1, loc=0.25, scale=0.2380) if c == 0 else norm.pdf(y1, loc=0.05, scale=0.2881)
    if c == 0:
        pY2 = 0.5 if y2 == 0 else 0.25 if y2 == 1 else 0.25
    else:
        pY2 = 1/6 if y2 == 0 else 1/3 if y2 == 1 else 0.5
    var0 = multivariate_normal([0.20, 0.25], [ [0.18, 0.18], [0.18,0.25] ] )
    var1 = multivariate_normal([0.117,0.083], [[0.1097,0.1223],[0.1223,0.2137]])
    pY3Y4 = var0.pdf([y3,y4]) if c == 0 else var1.pdf([y3,y4])
    print(pC*pY1*pY2*pY3Y4)
    matrix = getMatrix(dataArray)
    matrix = getRates(matrix)
    for line in matrix:
        print(line)
    


if __name__ == "__main__":
    main()
     



