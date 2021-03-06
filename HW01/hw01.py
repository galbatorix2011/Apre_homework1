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

# Returns the Colluns of the data list. (Returns all the values of a certain attribute)
def getVectors(data, indice):
    res = []
    for line in data:
        res.append(line[indice])
    return res

def getData(fileName):  # Fetches the data from a .txt to a list
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

def divideData(data):  # Will divide the data into Benign and Malignant
    benign = []
    malign = []

    for line in data:
        if line[-1] == "benign":
            benign.append(line)
        else:
            malign.append(line)

    return (benign, malign)

# ------------------------------Global-Variables---------------------------

data = getData("data.txt")  # Training Data Stored

ben, mal = divideData(data)  # ben = BenignData | mal = Malignant

# Atrributes
titles = ["Clump Thickness", "Cell Size Uniformity", "Cell Shape Uniformity", "Marginal Adhesion", "Single Epi Cell Size",
          "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]

# ---------------------------------------------------------------------------

fig, _ = plt.subplots(nrows=3, ncols=3, figsize=(10, 7))  # creating the plots
fig.tight_layout(pad=4.0)  # spacing
ast = fig.axes  # getting a list with the axes
fig.canvas.set_window_title('AP HW01 G132')  # Window Title

# Setting the Setting all the plots
for i in range(len(ast)):
    vectorsB = getVectors(ben, i)
    vectorsM = getVectors(mal, i)

    ast[i].title.set_text(titles[i]) #Set the plot title
    ast[i].set_ylabel("Count") #Set the Y label
    ast[i].set_xlabel("Value") # Set the X Label
    #Creating the Histogram
    ast[i].hist([vectorsB, vectorsM], 10, density=False, histtype='bar', color=['green', 'red'], label=["Benign", "Malignant"])
    ast[i].legend(prop={'size': 9}) #Setting the legend size

plt.show() #Shows the figure with all the plots
