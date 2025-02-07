#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by dropping non-useful columns and encoding categorical features.
    """
    # Drop columns that are not useful (adjust as needed)
    drop_columns = [
        "customer_id",
        "churn_label",
        "churn_category",
        "churn_reason",
        "customer_status",
    ]
    df = df.drop(columns=drop_columns, errors="ignore")

    # Encode categorical features using LabelEncoder
    categorical_features = df.select_dtypes(include=["object"]).columns
    for col in categorical_features:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


def select_features_and_target(df: pd.DataFrame):
    """
    Select the best features based on permutation importance results and set churn_value as the target.

    Best features chosen:
        - satisfaction_score
        - churn_score
        - contract
        - number_of_referrals
        - monthly_ charges
        - tenure
        - total_long_distance_charges
        - total_charges
        - total_revenue
    """
    features = [
        "satisfaction_score",
        "churn_score",
        "contract",
        "number_of_referrals",
        "monthly_ charges",
        "tenure",
        "total_long_distance_charges",
        "total_charges",
        "total_revenue",
    ]
    target = "churn_value"
    return df[features], df[target]


def compute_permutation_importance(
    rf_model,
    X_train: pd.DataFrame,
    y_train,
    scoring: str = "accuracy",
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute feature importance using permutation importance with the given Random Forest model.

    Parameters:
        rf_model: Trained RandomForestClassifier.
        X_train (pd.DataFrame): Training features.
        y_train: Training target.
        scoring (str): Scoring metric to evaluate performance drop (default 'accuracy').
        n_repeats (int): Number of times to permute a feature.
        random_state (int): Random seed for reproducibility.

    Returns:
        A DataFrame with features and their permutation importance scores sorted in descending order.
    """
    perm_result = permutation_importance(
        rf_model,
        X_train,
        y_train,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": perm_result.importances_mean}
    )

    return importance_df.sort_values(by="Importance", ascending=False)


def plot_feature_importance(importance_df: pd.DataFrame):
    """
    Plot the feature importance using a bar plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Permutation Feature Importance in Random Forest")
    plt.xlabel("Importance (Drop in Accuracy)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    # Set the path to your CSV file (adjust the path if necessary)
    file_path = "data/CustomerData_Composite-4.csv"

    # Load and preprocess the dataset
    df = load_data(file_path)
    df = preprocess_data(df)

    # Select the best features and target variable
    X, y = select_features_and_target(df)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train the Random Forest classifier using only the selected features
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = rf_model.predict(X_test)
    print("Accuracy on test set: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Compute and display confusion matrix
    print("Confusion Matrix:")
    plot_confusion_matrix(y_test, y_pred)  # Plot confusion matrix

    # Compute and display permutation feature importance
    importance_df = compute_permutation_importance(rf_model, X_train, y_train)
    print("\nPermutation Feature Importance:")
    print(importance_df)

    # Plot the feature importance
    plot_feature_importance(importance_df)


if __name__ == "__main__":
    main()
