from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np

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


data = getData("data.txt")  # Training Data Stored

kf = KFold(n_splits=10, random_state=132, shuffle=True)


def getMeanAccuracy(accuracies):
    return sum(accuracies) / len(accuracies)

def getAccuracy(predictions, real):
    return sum([1 if predictions[i] == real[i] else 0 for i in range(
            len(predictions))]) / len(predictions)    

accuracyDepth = []
accuracyFeatures = []
for i in [1, 3, 5, 9]:
    accuraciesFeatures = [] #each element will be a tuple with the test and training accuracy
    accuraciesDepth = [] #likewise
    for train_index, test_index in kf.split(data):  # Main for Loop
        trainingDataIn, trainingDataOut = divide([data[i] for i in train_index])
        testDataIn, testDataOut = divide([data[i] for i in test_index])

        #we select the best features with selectKbest using mutual_info_classif
        selection = SelectKBest(mutual_info_classif, k=i).fit(trainingDataIn, trainingDataOut)
        ftrainingDataIn = selection.transform(trainingDataIn)
        ftestDataIn = selection.transform(testDataIn)

        clfFeatures = tree.DecisionTreeClassifier(max_depth=None, max_features=None) # feature selection impact
        clfDepth = tree.DecisionTreeClassifier(max_depth=i, max_features=None) # max depth impact

        clfFeatures.fit(ftrainingDataIn, trainingDataOut)
        clfDepth.fit(trainingDataIn, trainingDataOut)

        predictionsDepthTest = clfDepth.predict(testDataIn)
        predictionsDepthTrain = clfDepth.predict(trainingDataIn)

        accuracyTest = getAccuracy(predictionsDepthTest, testDataOut)
        accuracyTrain = getAccuracy(predictionsDepthTrain, trainingDataOut)
        accuraciesDepth.append((accuracyTest, accuracyTrain))
        
        predictionsFeaturesTest = clfFeatures.predict(ftestDataIn)
        predictionsFeaturesTrain = clfFeatures.predict(ftrainingDataIn)

        accuracyTest = getAccuracy(predictionsFeaturesTest, testDataOut)
        accuracyTrain = getAccuracy(predictionsFeaturesTrain, trainingDataOut)
        accuraciesFeatures.append((accuracyTest, accuracyTrain))
    #Will append the final accuracys to the lists
    accuracyFeatures.append((getMeanAccuracy([accuraciesFeatures[i][0] for i in range(len(accuraciesFeatures))]),
                             getMeanAccuracy([accuraciesFeatures[i][1] for i in range(len(accuraciesFeatures))]))) 
    accuracyDepth.append((getMeanAccuracy([accuraciesDepth[i][0] for i in range(len(accuraciesDepth))]),
                          getMeanAccuracy([accuraciesDepth[i][1] for i in range(len(accuraciesDepth))])))

print("Coeficiente de Pearson: " + str(np.corrcoef([x[0] for x in accuraciesDepth],[x[0] for x in accuraciesFeatures])[0][1]))

x = [1, 3, 5, 9]
yDepthTest = [accuracy[0] for accuracy in accuracyDepth]
yDepthTrain = [accuracy[1] for accuracy in accuracyDepth]
yFeaturesTest = [accuracy[0] for accuracy in accuracyFeatures]
yFeaturesTrain = [accuracy[1] for accuracy in accuracyFeatures]
plt.figure('AP HW02 G132')  # Window Title
plt.plot(x, yDepthTest, "-o", label="Tree Depth Impact (Test)")
plt.plot(x, yDepthTrain, "-o", label="Tree Depth Impact (Train)")
plt.plot(x, yFeaturesTest, "-s", label="Feature Selection Impact (Test)")
plt.plot(x, yFeaturesTrain, "-s", label="Feature Selection Impact (Train)")
plt.ylabel("Accuracies")
plt.xlabel("Depth/Features")
plt.legend()
plt.show()
