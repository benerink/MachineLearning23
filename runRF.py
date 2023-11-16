import pandas as pd
from sklearn.model_selection import train_test_split
from randomForest import RandomForest
import numpy as np
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

start_time = time.time()


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


data = pd.read_csv('train_cleaned.csv')

# Identify numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns

# Filter only numeric columns
data_numeric = data[numeric_columns]

X = data_numeric.drop('Verkaufspreis', axis=1)
y = data_numeric['Verkaufspreis']

print("Data loaded successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Random Forest-Modell erstellen und trainieren
clf = RandomForest(n_trees=10)
print("Random Forest model created.")
clf.fit(X_train.values, y_train.values)

# Vorhersagen machen
predictions = clf.predict(X_test.values)

# Genauigkeit berechnen
acc = accuracy(y_test.values, predictions)
print("Genauigkeit:", acc)

end_time = time.time()
execution_time = end_time - start_time
print(f"Laufzeit: {execution_time} Sekunden")
