import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_and_predict_random_forest(df):

    # Choose relevant features for prediction
    #features = ['Baujahr', 'Umbaujahr', 'Wohnungsklasse', 'OberirdischeWohnflaeche']
    features = ['Hausqualitaet', 'OberirdischeWohnflaeche', 'KellerbereichgroesseGes', '1Stockflaeche']


    # Extract features and target variable
    X = df[features]
    y = df['Verkaufspreis']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Visualize the results
    plt.scatter(X_test.iloc[:, 0], y_test, color='black', label='Actual Prices')
    plt.scatter(X_test.iloc[:, 0], y_pred, color='green', label='Predicted Prices (Random Forest)')
    plt.xlabel('Baujahr (Year of Construction)')
    plt.ylabel('Verkaufspreis (Selling Price)')
    plt.legend()
    plt.show()

    # Predict the selling price for new data
    new_data = pd.DataFrame([[2025, 2025, 30, 120]], columns=features)
    predicted_price = model.predict(new_data)
    print(f'Predicted Selling Price for 2025: {predicted_price[0]}')
    print(predicted_price)

    return X_test, y_test, y_pred, predicted_price

df = pd.read_excel("C:\\Users\\Luca\\PycharmProjects\\MachineLearning23\\resources\\train_cleaned_transformed_scaled.xlsx")

train_and_predict_random_forest(df)