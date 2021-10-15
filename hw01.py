"""
###################
Group 132
###################

Afonso Almeida 92790
Duarte Elvas 98564

"""
# ---------------------------------Imports------------------------------
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# --------------------------------Functions-----------------------------

def getVectors(data, indice):
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

def divideData(data):
    benign = []
    malign = []

    for line in data:
        if line[-1] == "benign":
            benign.append(line)
        else:
            malign.append(line)
    
    return (benign, malign)


# ------------------------------Global-Variables---------------------------

data = getData("TrainingData.txt") #Training Data Stored

ben, mal = divideData(data)

#Atrributes
titles = ["Clump Thickness", "Cell Size Uniformity", "Cell Shape Uniformity", "Marginal Adhesion", "Single Epi Cell Size", \
"Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"] 

#---------------------------------------------------------------------------

fig, _ = plt.subplots(nrows=3, ncols=3, figsize=(10,7)) # creating the plots
fig.tight_layout(pad=4.0) # spacing
ast=fig.axes # getting a list with the axes
fig.canvas.set_window_title('AP HW01 G132') # Window Title


#Setting the titles
for i in range(len(ast)):
    vectorsB = getVectors(ben,i)
    vectorsM = getVectors(mal,i)

    ast[i].title.set_text(titles[i])
    ast[i].set_ylabel("Count")
    ast[i].set_xlabel("Value")
    ast[i].legend(['benign', 'malignant'])
    ast[i].hist( [vectorsB, vectorsM], 4, density = False)


plt.show()
