"""
###################
Group 132
###################

Afonso Almeida 92790
Duarte Elvas 98564

"""
# ---------------------------------Imports------------------------------

import matplotlib.pyplot as plt

# --------------------------------Functions-----------------------------

def getVectors(indice):
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
        tmp = line[:-1].split(",") #removes the /n from the end of each line
        for i in range(len(tmp) - 1): #does not check the last element cause its not an int
            try: #Truns every str into int
                tmp[i] = int(tmp[i])
                res.append(tmp)
            except Exception as e:
                pass #ignores lines that have an '?'
    return res

# ------------------------------Global-Variables---------------------------

data = getData("TrainingData.txt") #Training Data Stored

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

    vectors = getVectors(i)
    ast[i].title.set_text(titles[i])
    ast[i].set_ylabel("Amount")
    ast[i].set_xlabel("Value")
    ast[i].hist(vectors, 4, density = True, edgecolor='red')


plt.show()

