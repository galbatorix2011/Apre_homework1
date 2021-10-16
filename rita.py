import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

class data:
    def __init__(self, y1, y2, y3, y4, classer):
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.classer = classer


def populate():
    res = []
    res.append(data(0.6, 'A', 0.2, 0.4, 0)) #x1
    res.append(data(0.1, 'B', -0.1, -0.4, 0)) #x2
    res.append(data(0.2, 'A', -0.1, 0.2, 0)) #x3 
    res.append(data(0.1, 'C', 0.8, 0.8, 0)) #x4

    res.append(data(0.3, 'B', 0.1, 0.3, 1)) #x5
    res.append(data(-0.1, 'C', 0.2, -0.2, 1)) #x6
    res.append(data(0.3, 'C', -0.1, 0.2, 1)) #x7
    res.append(data(0.2, 'B', 0.5, 0.6, 1)) #x8
    res.append(data(0.4, 'A', -0.4, -0.7, 1)) #x9
    res.append(data(-0.2, 'C', 0.4, 0.3, 1)) #x10

    return res

def main():
    dataArray = populate()
    
    c = eval(input("Classe: "))

    pC = 0.4 if c == 0 else 0.6
    y1 = eval(input("y1: "))
    y2 = eval(input("y2: "))
    y3 = eval(input("y3: "))
    y4 = eval(input("y4: "))
    
    pY1 = norm.pdf(y1, loc=0.25, scale=0.2380) if c == 0 else norm.pdf(y1, loc=0.05, scale=0.243)
    if c == 0:
        pY2 = 0.5 if y2 == 0 else 0.25 if y2 == 1 else 0.25
    else:
        pY2 = 1/6 if y2 == 0 else 1/3 if y2 == 1 else 0.5
    var0 = multivariate_normal([0.20, 0.25], [ [0.18, 0.18], [0.18,0.25] ] )
    var1 = multivariate_normal([0.117,0.083], [[0.1097,0.1223],[0.1223,0.2137]])
    pY3Y4 = var0.pdf([y3,y4]) if c == 0 else var1.pdf([y3,y4])
    var3 = multivariate_normal([0.20, 0.25], [ [0.18, 0.18], [0.18,0.25] ] )
    print(var0.pdf([0.2,0.4]))


if __name__ == "__main__":
    main()
     



