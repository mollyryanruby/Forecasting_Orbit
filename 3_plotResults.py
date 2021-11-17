import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import os

dirPath = str(pathlib.Path().resolve())
dataPath = dirPath + "/data/"
modelPath = dataPath + 'model_output.csv'
rankPath = dataPath + 'ranking.csv'
rawPath = dataPath + 'procssed_data.csv'
date_col = 'date'
response_col = 'sales'
imgPath = dirPath + "/img/"

model_df = pd.read_csv(modelPath)
rankData = pd.read_csv(rankPath)
rawdata = pd.read_csv(rawPath)

def pivot(df):
    return pd.pivot_table(df, values='Prediction', index=['Date', 'Actual'], columns='Model').reset_index() 

pivot_df = pivot(model_df)
rawdata = rawdata.sort_values(by=date_col)
# Plot the time series
fig, ax = plt.subplots(figsize=(14,6))

ax.plot(rawdata[date_col], rawdata[response_col], label='Actual')
for model in model_df.Model.unique():  
    ax.plot(pivot_df.Date, pivot_df[model], label=model)

for ind, label in enumerate(ax.xaxis.get_ticklabels()):
    if ind % 13 == 0:  # every 5th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)

plt.legend()
fig.savefig(imgPath + "Forecast_Outputs.jpg")


fig, ax = plt.subplots(figsize=(16,6))
width = .75
color = '#000080'
ax.barh(rankData.Model, rankData.MAPE, width, color=color)

for i, v in enumerate(rankData.MAPE):
    ax.text(v + .25, i, str(round(v,2)), 
            color =color, fontweight = 'bold')

fig.savefig(imgPath + "MAPE_Comparison.jpg")