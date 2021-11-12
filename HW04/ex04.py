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

def getECRScore(dataOut, prediction, n):
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
    print("c0: " + str(c0Total) )
    c1Total = c1Benign + c1Malignant
    return ((c0Total - max(c0Benign, c0Malignant)) + (c1Total - max(c1Benign, c1Malignant))) / n
    
def getECRScore(dataOut, prediction, n):
    c0Benign = 0
    c0Malignant = 0
    
    c1Benign = 0
    c1Malignant = 0
    
    c2Benign = 0
    c2Malignant = 0
    for i in range(len(prediction)):
        if prediction[i] == 0:
            if dataOut[i] == "malignant":
                c0Malignant +=1
            else:
                c0Benign +=1
        elif prediction[i] == 1:
            if dataOut[i] == "malignant":
                c1Malignant +=1
            else:
                c1Benign +=1
        elif prediction[i] == 2:
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


kmeans = KMeans(n_clusters=2, random_state=0).fit(dataIn)
kmeansK3s = KMeans(n_clusters=3, random_state=0).fit(dataIn)


kmeansK3 = KMeans(n_clusters=3, random_state=0).fit(newData)

prediction = list(kmeans.labels_)
predictionK3s = list(kmeansK3s.labels_)
prediction2 = list(kmeansK3.labels_)
print("ECR K2 ---> "+ str(getECRScore(dataOut, prediction, 2)))
print("ECR K3 ---> "+ str(getECRScore(dataOut, predictionK3s, 3)))
print()
print("Silhouette K2 ---> "+ str(silhouette_score(dataIn, prediction)))
print("Silhouette K3 ---> "+ str(silhouette_score(dataIn, predictionK3s)))

print("#-" * 30)

print("ECR 5 ---> "+ str(getECRScore(dataOut, prediction2, 3)))
print("Silhouette 5 ---> "+ str(silhouette_score(dataIn, prediction2)))

      



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

plt.scatter([i[0] for i in cluster0] , [i[1] for i in cluster0], color = "blue")

plt.scatter([i[0] for i in cluster1] , [i[1] for i in cluster1], color = "green")

plt.scatter([i[0] for i in cluster2] , [i[1] for i in cluster2], color = "red")

#plt.scatter([point[0] for point in newData], [point[1] for point in newData])


#plt.scatter([newData[i][0]  for i in range(len(prediction)) if prediction[i] == 1 ],[newData[i][1]  for i in range(len(prediction)) if prediction[i] == 1 ])
#plt.scatter([newData[i][0]  for i in range(len(prediction)) if prediction[i] == 0 ],[newData[i][1]  for i in range(len(prediction)) if prediction[i] == 0 ])
plt.show()
#print(x)
    
