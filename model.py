import pandas as pd
import numpy as np
import subprocess
import sys
import pathlib
import time

start_time = time.time()

def checkForPackages(package):

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0].lower() for r in reqs.split()]

    if not package in installed_packages:
        print("Installing", package, "...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

modelpackages = {'orbit': 'orbit-ml', 'pmdarima': 'pmdarima'}

for package in modelpackages:
    try:
        import package
    except: 
        checkForPackages(modelpackages[package])

from orbit.models.dlt import ETSFull, DLTMAP, DLTFull
from orbit.models.lgt import LGTMAP, LGTFull, LGTAggregated
# from orbit.diagnostics.plot import plot_predicted_data
from pmdarima.arima import auto_arima


## Instantiate Variables
dataPath = str(pathlib.Path().resolve()) + "/data/"
rawDataPath = dataPath + 'train.csv.zip'
date_col = 'date'
response_col = 'sales'

## Generate Functions
def preprocessData(df, date_col, response_col):
    # Convert from daily data into monthly data
    df[date_col] = df[date_col].apply(lambda x: str(x)[:-3])
    df = df.groupby(date_col)[response_col].sum().reset_index()
    df[date_col] = pd.to_datetime(df[date_col], format='%Y/%m')
    return df.sort_values(by=date_col)

def trainTestestSplit(df, test_size=12):
    train = df[:-test_size]
    test = df[-test_size:]
    return train, test

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def runTheModel(df_train, df_test, model, modName, date_col, response_col):
    if not 'ARIMA' in modName:
        model.fit(df_train)
        pred = model.predict(df_test[[date_col]]).prediction.unique()
    else:
        pred = model.predict(df_test.shape[0])

    error = mape(df_test[response_col], pred)
    prediction_df = pd.DataFrame.from_dict({'Date': df_test[date_col], 'Actual': df_test[response_col], 'Prediction': pred, 'MAPE': error, 'Model': modName})
    
    # print(prediction_df.head())
    return prediction_df

## Load and Process Data
df = pd.read_csv(rawDataPath)
df = preprocessData(df, date_col, response_col)
train, test = trainTestestSplit(df)

# Instantiate model dictionary
models = {'ETSFull': ETSFull(
                    response_col=response_col,
                    date_col=date_col,
                    seasonality=12,
                    seed=8888
                    ),
            'DLTMAP_Linear': DLTMAP(
                    response_col=response_col,
                    date_col=date_col,
                    seasonality=12,
                    seed=8888
                    ),
            'DLTMAP_LogLin': DLTMAP(
                            response_col=response_col,
                            date_col=date_col,
                            seasonality=12,
                            seed=8888,
                            global_trend_option='loglinear'
                            ),
            'DLTMAP_Logistic': DLTMAP(
                                response_col=response_col,
                                date_col=date_col,
                                seasonality=12,
                                seed=8888,
                                global_trend_option='logistic'
                                ),
            'LGTMAP': LGTMAP(
                        response_col=response_col,
                        date_col=date_col,
                        seasonality=12,
                        seed=8888
                        ), # Commented out because the results were too poor
            'LGTFull': LGTFull(
                        response_col=response_col,
                        date_col=date_col,
                        seasonality=12,
                        seed=8888,
                    ),
            'LGTAggregation': LGTAggregated(
                                response_col=response_col,
                                date_col=date_col,
                                seasonality=12,
                                seed=8888,
                            ),
            'DLTFull': DLTFull(
                        response_col=response_col,
                        date_col=date_col,
                        seasonality=12,
                        seed=8888,
                        num_warmup=5000,
                    ),
           'ARIMA': auto_arima(train[[response_col]],
                                m=12,
                                seasonal=True
                                ) }
        

# Run the models
predictions = []

for mod in models:
    print('running', mod)
    predictions.append(runTheModel(train, test, models[mod], mod, date_col, response_col))

# Create dataframe of all results
full_df = pd.concat(predictions)
full_df['Rank'] = full_df.MAPE.rank(method='dense')

# Create dataframe of model rankings
ranking_df = full_df[['Model', 'Rank', 'MAPE']].drop_duplicates().sort_values(by='Rank')

# Write data
full_df.to_csv(dataPath + 'model_output.csv', index=False)
ranking_df.to_csv(dataPath + 'ranking.csv', index=False)
df.to_csv(dataPath + 'procssed_data.csv', index=False)
# train.to_csv(dirpath + "train.csv", index=False)
# test.to_csv(dirpath + 'test.csv', index=False)

print("Application took:", time.time() - start_time, "to run.")

