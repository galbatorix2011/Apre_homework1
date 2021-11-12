from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def getData(fileName):  # Converts the data in a .txt file to an array
    res = []
    # Reads the file
    with open(fileName) as f:
        lines = f.readlines()
    # Formats all lines
    for line in lines:
        tmp = line[:-1].split(",")
        if tmp[-1] == "malignan":
            tmp[-1] = "malignant"
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
    
def getECRScore(dataOut, labelK2, n): #Will calcuçate the ECR score with n dimentions
    c0Benign = 0
    c0Malignant = 0
    
    c1Benign = 0
    c1Malignant = 0
    
    c2Benign = 0
    c2Malignant = 0
    for i in range(len(labelK2)):
        if labelK2[i] == 0:
            if dataOut[i] == "malignant":
                c0Malignant +=1
            else:
                c0Benign +=1
        elif labelK2[i] == 1:
            if dataOut[i] == "malignant":
                c1Malignant +=1
            else:
                c1Benign +=1
        elif labelK2[i] == 2:
            if dataOut[i] == "malignant":
                c2Malignant +=1
            else:
                c2Benign +=1

    c0Total = c0Benign + c0Malignant
    c1Total = c1Benign + c1Malignant
    c2Total = c2Benign + c2Malignant
    return ((c0Total - max(c0Benign, c0Malignant)) + (c1Total - max(c1Benign, c1Malignant)) + (c2Total - max(c2Benign, c2Malignant))) / n
    
data = getData("data.txt")  # Training Data Stored
dataIn, dataOut = divide(data)

newData = SelectKBest(mutual_info_classif, k=2).fit_transform(dataIn,dataOut)
#NewData are the data points but only with the top-2 features

kmeansK2 = KMeans(n_clusters=2, random_state=0).fit(dataIn)
kmeansK3 = KMeans(n_clusters=3, random_state=0).fit(dataIn)
kmeansEx5 = KMeans(n_clusters=3, random_state=0).fit(newData)

labelK2 = list(kmeansK2.labels_)
labelK3 = list(kmeansK3.labels_)
labelEx5 = list(kmeansEx5.labels_)

print("ECR K2 ---> "+ str(getECRScore(dataOut, labelK2, 2)))
print("ECR K3 ---> "+ str(getECRScore(dataOut, labelK3, 3)))
print("")
print("Silhouette K2 ---> "+ str(silhouette_score(dataIn, labelK2)))
print("Silhouette K3 ---> "+ str(silhouette_score(dataIn, labelK3)))

print("#-" * 30)

print("ECR 5 ---> "+ str(getECRScore(dataOut, labelEx5, 3)))
print("Silhouette 5 ---> "+ str(silhouette_score(newData, labelEx5)))

cluster0 = []
cluster1 = []
cluster2 = []

for i in range(len(labelEx5)): # Will divide the data into clusters
    if labelEx5[i] == 0 :
        cluster0.append(newData[i])
    elif labelEx5[i] == 1:
        cluster1.append(newData[i])
    else:
        cluster2.append(newData[i])

plt.scatter([i[0] for i in cluster0] , [i[1] for i in cluster0], color = "blue", label = "cluster 0")
plt.scatter([i[0] for i in cluster1] , [i[1] for i in cluster1], color = "green", label = "cluster 1")
plt.scatter([i[0] for i in cluster2] , [i[1] for i in cluster2], color = "red", label = "cluster 2")

plt.legend()
plt.show()
    
