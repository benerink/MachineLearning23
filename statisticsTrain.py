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

# check all numerical columns via loop for missing entries and replace them with 0
for col in data.columns:
    if data[col].dtype != 'object':
        print(col)
        data[col] = data[col].fillna(0)

# check all non-numerical columns via loop for missing entries and replace them with 'NA'
for col in data.columns:
    if data[col].dtype == 'object':
        print(col)
        data[col] = data[col].fillna('NA')

# write the data to a new csv file in the current folder
data.to_csv('train_cleaned.csv', index=False)

# check all numerical columns via loop and print the description
newdata = pd.read_csv('train_cleaned.csv')
for col in newdata.columns:
    if newdata[col].dtype != 'object':
        print(col)
        print(newdata[col].describe())
    else:
        print(col)
        print(newdata[col].describe(include='all'))





