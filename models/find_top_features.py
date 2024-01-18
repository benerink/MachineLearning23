from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score


def find_top_features(df, num_top_features):
    # Extract features and target variable
    X = df.drop(columns=['Verkaufspreis'])  # Exclude the target variable
    y = df['Verkaufspreis']

    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model to the entire dataset
    model.fit(X, y)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame to store feature importances
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort features by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Select the specified number of most important features
    top_features = importance_df.head(num_top_features)['Feature'].tolist()

    return top_features
