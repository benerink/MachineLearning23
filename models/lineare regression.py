import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from utils.utils import get_excel_file_path

df = pd.read_excel(get_excel_file_path("train_cleaned_transformed.xlsx"))

# Choose relevant features for prediction
features = ['Baujahr', 'Umbaujahr', 'Wohnungsklasse', 'OberirdischeWohnflaeche']

# Extract features and target variable
X = df[features]
y = df['Verkaufspreis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.scatter(X_test['Baujahr'], y_test, color='black', label='Actual Prices')
plt.scatter(X_test['Baujahr'], y_pred, color='blue', label='Predicted Prices')
plt.xlabel('Baujahr (Year of Construction)')
plt.ylabel('Verkaufspreis (Selling Price)')
plt.legend()
plt.show()

new_data = pd.DataFrame([[2025, 2025, 30, 120]], columns=features)
predicted_price = model.predict(new_data)
print(f'Predicted Selling Price for 2025: {predicted_price[0]}')
