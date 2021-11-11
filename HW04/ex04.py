from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
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


#def outBinarization(dataOut):
#    return [lambda(i : 1 if i == "malignant" else 0) for i in dataOut]

def getAccuracy(predictions, real):
    return sum([1 if predictions[i] == real[i] else 0 for i in range(
            len(predictions))]) / len(predictions) 

def getMeanAccuracy(accuracies):
    return sum(accuracies) / len(accuracies)

def getECRScore(dataOut, prediction):
    c0Benign = 0
    c0Malignant = 0
    
    c1Benign = 0
    c1Malignant = 0
    for i in range(len(prediction)):
        if prediction[i] == 0:
            if dataOut[i] == "malignant":
                c0Malignant +=1
            else:
                c0Benign +=1
        else:
            if dataOut[i] == "malignant":
                c1Malignant +=1
            else:
                c1Benign +=1

    c0Total = c0Benign + c0Malignant
    c1Total = c1Benign + c1Malignant
    return ((c0Total - max(c0Benign, c0Malignant)) + (c1Total - max(c1Benign, c1Malignant))) / 2
    

data = getData("data.txt")  # Training Data Stored

dataIn, dataOut = divide(data)


newData = SelectKBest(mutual_info_classif, k=2).fit_transform(dataIn,dataOut)

#print(newData)

kmeans = KMeans(n_clusters=2, random_state=0).fit(dataIn)
kmeansK3 = KMeans(n_clusters=3, random_state=0, max_iter=20000).fit(newData)

prediction = list(kmeans.labels_)
prediction2 = list(kmeansK3.labels_)
#print(getECRScore(dataOut, prediction))
#print()
x = silhouette_score(dataIn, prediction)



cluster0 = []
cluster1 = []
cluster2 = []


for i in range(len(prediction2)):
    if prediction2[i] == 0 :
        cluster0.append(newData[i])
    elif prediction2[i] == 1:
        cluster1.append(newData[i])
    else:
        cluster2.append(newData[i])
print("cluster 0:")
print([list(e) for e in cluster0])
print("cluster 1:")
print([list(e) for e in cluster1])
print("cluster 2:")
print([list(e) for e in cluster2])

plt.scatter([i[0] for i in cluster0] , [i[1] for i in cluster0], color = "blue")

#plt.scatter([i[0] for i in cluster1] , [i[1] for i in cluster1], color = "green")

#plt.scatter([i[0] for i in cluster2] , [i[1] for i in cluster2], color = "red")

#plt.scatter([point[0] for point in newData], [point[1] for point in newData])


#plt.scatter([newData[i][0]  for i in range(len(prediction)) if prediction[i] == 1 ],[newData[i][1]  for i in range(len(prediction)) if prediction[i] == 1 ])
#plt.scatter([newData[i][0]  for i in range(len(prediction)) if prediction[i] == 0 ],[newData[i][1]  for i in range(len(prediction)) if prediction[i] == 0 ])
plt.show()
#print(x)
    
