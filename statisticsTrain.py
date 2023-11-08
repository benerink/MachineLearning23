import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('train.csv')

# Setting to display all columns
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# check all columns for missing entries without shortening the output
print(data.isnull().sum())

# print all entries in poolflaeche
print(data['Poolflaeche'])
