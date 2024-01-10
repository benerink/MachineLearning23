import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Read the cleaned transformed data
data = pd.read_excel('train_cleaned_transformed.xlsx')

# Setting to display all columns
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# create a random Forest model with sklearn
rf = RandomForestClassifier()
print(rf.get_params())

# fit the model to the data to get a predicition for a price of a house
rf.fit(data.drop('Verkaufspreis', axis=1), data['Verkaufspreis'])






