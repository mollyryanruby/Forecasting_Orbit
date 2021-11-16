import pandas as pd
import pathlib
import matplotlib.pyplot as plt

dirpath = str(pathlib.Path().resolve()) + "/data/"
modelPath = dirpath + 'model_output.csv'
rankPath = dirpath + 'ranking.csv'

model_df = pd.read_csv(modelPath)
rankPath = pd.read_csv(rankPath)

def pivot(df):
    return pd.pivot_table(df, values='Prediction', index=['Date', 'Actual'], columns='Model').reset_index() 

model_df2 = pivot(model_df)

for model in model_df.Model.unique():  
    plt.plot(model_df2.Date, model_df2[model], label=model)
plt.legend()
plt.show()