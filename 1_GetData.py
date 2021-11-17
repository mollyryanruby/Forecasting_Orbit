import subprocess
import sys
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import os

def checkForPackages(package):

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0].lower() for r in reqs.split()]

    if not package in installed_packages:
        print("Installing", package, "...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import kaggle
except:
    checkForPackages('kaggle')

# kaggle competitions download -c demand-forecasting-kernels-only -f test.csv.7z
from kaggle.api.kaggle_api_extended import KaggleApi

dirPath = str(pathlib.Path().resolve())
dataPath =  dirPath + "/data/"
api = KaggleApi()
api.authenticate()

api.competition_download_file('demand-forecasting-kernels-only','train.csv', path=dataPath)


# def getDataPerZip(zip):
#     countyData = ql.get_table("ZILLOW/REGIONS", region_type='zip', paginate=True)
#     regions = countyData[countyData.region.str.contains(zip)].region_id.unique()

#     if len(regions) == 1:
#         priceData = ql.get_table("ZILLOW/DATA", 
#                                 indicator_id='ZCON', 
#                                 region_id=regions[0], 
#                                 paginate=True)
#     else:
#         print("Multiple regions selected")  
#         priceData = ql.get_table("ZILLOW/DATA", 
#                                 indicator_id='ZCON',
#                                 paginate=True)
#         priceData = priceData.filter(priceData.region_id.str.isin(regions))
    
#     return priceData

# def filterDates(df):
#     df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
#     df2 = df[(df.date >= '2016-01-01') & (df.date < '2020-01-01')]
#     return df2

# ql.ApiConfig.api_key = "3shXhW8vy7tPPKoazXwb"

# df = getDataPerZip('10023')
# df = filterDates(df)

# dirpath = str(pathlib.Path().resolve()) + "/data/"

# if not os.path.exists(dirpath):
#     os.mkdir(dirpath)

# path = dirpath + 'raw.csv'

# df.to_csv(path)
