import orbit 
import pandas as pd
import matplotlib.pyplot as plt
from orbit.diagnostics.plot import plot_predicted_data,plot_predicted_components
from model import runTheModel, getModelDict, trainTestestSplit
import pathlib
import os


## Instantiate Variables
dirPath = str(pathlib.Path().resolve())
dataPath = dirPath + "/data/"
rankPath = dataPath + 'ranking.csv'
processedDataPath = dataPath + 'procssed_data.csv'
imgPath = dirPath + "/img/"
date_col = 'date'
response_col = 'sales'

df = pd.read_csv(processedDataPath)
train, test = trainTestestSplit(df)

rankData = pd.read_csv(rankPath)
bestModel = rankData[rankData.Rank==1].Model[0]

modelDict = getModelDict(train)

if not os.path.exists(imgPath):
    os.mkdir(imgPath)

fullimgpath = imgPath + 'decomposition.jpg'
runTheModel(train, test, modelDict[bestModel], bestModel, date_col, response_col, decompose=True, imgPath=fullimgpath)



