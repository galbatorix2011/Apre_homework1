import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

def main():
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
    #var = multivariate_normal(mean = [80, 203.3333], cov = [[100, 50], [50, 233.3333]])
    #print(var.pdf([100, 225]))

     



