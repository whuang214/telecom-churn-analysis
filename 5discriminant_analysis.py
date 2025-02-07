import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# Encode categorical features
def encode_categorical_features(
    df: pd.DataFrame, categorical_columns: list
) -> pd.DataFrame:
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df.loc[:, col] = le.fit_transform(
            df[col]
        )  # Using .loc to avoid SettingWithCopyWarning
        label_encoders[col] = le
    return df, label_encoders


# Define feature selection based on F-Scores
BEST_FEATURES = [
    "satisfaction_score",
    "churn_score",
    "contract",
    "tenure",
    "number_of_referrals",
    "dependents",
    "total_long_distance_charges",
    "total_revenue",
    "internet_service",
    "number_of_dependents",
    "total_charges",
    "monthly_ charges",
    "paperless_billing",
    "unlimited_data",
    "premium_tech_support",
    "online_security",
    "offer",
    "referred_a_friend",
    "partner",
    "senior_citizen",
]


def preprocess_data(df: pd.DataFrame, target: str) -> tuple:
    # Select relevant features
    df = df[
        BEST_FEATURES + [target]
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Encode categorical features
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    df, label_encoders = encode_categorical_features(df, categorical_columns)

    # Split data
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoders


# Train Discriminant Analysis model
def train_lda(X_train: pd.DataFrame, y_train: pd.Series) -> LinearDiscriminantAnalysis:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda


# Evaluate model
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return accuracy


# Print Discriminant Function Coefficients and Intercepts
def print_discriminant_coefficients(
    model: LinearDiscriminantAnalysis, feature_names: list
):
    print("Intercepts:", model.intercept_)
    print("Coefficients:")
    for feature, coef in zip(feature_names, model.coef_[0]):
        print(f"{feature}: {coef}")


# Print Classification Scores and Propensities
def print_classification_scores(model, X_train, y_train):
    scores = model.decision_function(X_train)  # Get classification scores
    probabilities = model.predict_proba(X_train)  # Get propensities

    print("\nFirst 5 Customers - Classification Scores & Propensities:")
    for i in range(5):
        print(f"Customer {i+1}:")
        print(f" - Classification Score: {scores[i]}")
        print(f" - Churn Propensity: {probabilities[i][1]:.4f}")
        print("-----")


def main():
    # Load data
    file_path = "data/CustomerData_Composite-4.csv"
    df = load_data(file_path)

    # Define target variable
    target = "churn_value"

    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df, target)

    # Train model
    lda_model = train_lda(X_train, y_train)

    # Evaluate model
    evaluate_model(lda_model, X_test, y_test)

    # Print Discriminant Analysis Coefficients
    print_discriminant_coefficients(lda_model, X_train.columns.tolist())

    # Print Classification Scores and Propensities
    print_classification_scores(lda_model, X_train, y_train)


if __name__ == "__main__":
    main()
