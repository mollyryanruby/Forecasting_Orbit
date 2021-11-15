import pandas as pd
import matplotlib.pyplot as plt
import orbit
from orbit.utils.dataset import load_iclaims
from orbit.models.dlt import ETSFull
from orbit.diagnostics.plot import plot_predicted_data

dirpath = str(pathlib.Path().resolve()) + "/data/"
path = dirpath + 'raw.csv'

df = pd.read_csv(path)

print(df.head())