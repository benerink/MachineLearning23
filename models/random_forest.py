import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def train_and_predict_random_forest(df, top_features):
    # Ensure at least one feature is provided
    if not top_features:
        raise ValueError("At least one feature must be provided.")

    # Placeholder values for each feature
    value_for_feature1 = 10
    value_for_feature2 = 20
    value_for_feature3 = 30
    value_for_feature4 = 40
    value_for_feature5 = 50
    value_for_feature6 = 60
    value_for_feature7 = 70
    value_for_feature8 = 80
    value_for_feature9 = 90
    value_for_feature10 = 100

    # Define a dictionary with predefined values for each number of top features (1 to 10)
    predefined_values = {
        1: [value_for_feature1],
        2: [value_for_feature1, value_for_feature2],
        3: [value_for_feature1, value_for_feature2, value_for_feature3],
        4: [value_for_feature1, value_for_feature2, value_for_feature3, value_for_feature4],
        5: [value_for_feature1, value_for_feature2, value_for_feature3, value_for_feature4, value_for_feature5],
        6: [value_for_feature1, value_for_feature2, value_for_feature3, value_for_feature4, value_for_feature5,
            value_for_feature6],
        7: [value_for_feature1, value_for_feature2, value_for_feature3, value_for_feature4, value_for_feature5,
            value_for_feature6, value_for_feature7],
        8: [value_for_feature1, value_for_feature2, value_for_feature3, value_for_feature4, value_for_feature5,
            value_for_feature6, value_for_feature7, value_for_feature8],
        9: [value_for_feature1, value_for_feature2, value_for_feature3, value_for_feature4, value_for_feature5,
            value_for_feature6, value_for_feature7, value_for_feature8, value_for_feature9],
        10: [value_for_feature1, value_for_feature2, value_for_feature3, value_for_feature4, value_for_feature5,
             value_for_feature6, value_for_feature7, value_for_feature8, value_for_feature9, value_for_feature10]
    }

    # Check the number of top features
    num_top_features = len(top_features)

    # Check if the number of top features is within a valid range
    if not 1 <= num_top_features <= 10:
        raise ValueError("Number of top features should be between 1 and 10.")

    # Retrieve the predefined values for the specified number of top features
    values = predefined_values[num_top_features]

    # Choose relevant features for prediction
    features = top_features

    # Extract features and target variable
    X = df[features]
    y = df['Verkaufspreis']

    # Handle missing values (NaN) or zeros by imputing the mean value
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Generate new data for prediction
    new_data = pd.DataFrame([values], columns=top_features[:num_top_features])

    # Predict the selling price for new data
    predicted_price = model.predict(new_data)
    print(f'Random Forest Predicted Selling Price for New Data: {predicted_price[0]}')

    return X_test, y_test, y_pred, predicted_price
