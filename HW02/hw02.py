from sklearn import tree
from sklearn.model_selection import KFold

def getData(fileName): #Converts the data in a .txt file to an array
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

def divide(data): #will divide the data in order to be processed by the MultinomialNB function
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

accuracyDepth = []
accuracyFeatures = []
for i in [1,3,5,9]:
    accuraciesFeatures = []
    accuraciesDepth = []
    for train_index, test_index in kf.split(data): #Main for Loop
        trainingDataIn, trainingDataOut = divide([data[i] for i in train_index])
        testDataIn , testDataOut = divide([data[i] for i in test_index])
        clfFeatures = tree.DecisionTreeClassifier(max_depth=None, criterion="entropy", max_features= i) 
        clfDepth = tree.DecisionTreeClassifier(max_depth=i, criterion="entropy", max_features=None)
        
        clfFeatures.fit(trainingDataIn, trainingDataOut)
        clfDepth.fit(trainingDataIn, trainingDataOut)

        predictionsDepthTest = clfFeatures.predict(testDataIn)
        predictionsDepthTrain = clfFeatures.predict(trainingDataIn)
        accuracyTest = sum([1 if predictionsDepthTest[i] == testDataOut[i] else 0 for i in range(len(predictionsDepthTest))]) / len(predictionsDepthTest)
        accuracyTrain = sum([1 if predictionsDepthTrain[i] == trainingDataOut[i] else 0 for i in range(len(predictionsDepthTrain))]) / len(predictionsDepthTrain)
        accuraciesDepth.append((accuracyTest, accuracyTrain))

        predictionsFeaturesTest = clfDepth.predict(testDataIn)
        predictionsFeaturesTrain = clfDepth.predict(trainingDataIn)
        accuracyTest = sum([1 if predictionsFeaturesTest[i] == testDataOut[i] else 0 for i in range(len(predictionsFeaturesTest))]) / len(predictionsFeaturesTest)
        accuracyTrain = sum([1 if predictionsFeaturesTrain[i] == trainingDataOut[i] else 0 for i in range(len(predictionsDepthTrain))]) / len(predictionsDepthTrain)
        accuraciesFeatures.append((accuracyTest, accuracyTrain))
    accuracyFeatures.append((getMeanAccuracy([accuraciesFeatures[i][0] for i in range(len(accuraciesFeatures))]),
        getMeanAccuracy([accuraciesFeatures[i][1] for i in range(len(accuraciesFeatures))])))
    accuracyDepth.append((getMeanAccuracy([accuraciesDepth[i][0] for i in range(len(accuraciesDepth))]),
        getMeanAccuracy([accuraciesDepth[i][1] for i in range(len(accuraciesDepth))])))
    

print(accuracyDepth)
print(accuracyFeatures)


