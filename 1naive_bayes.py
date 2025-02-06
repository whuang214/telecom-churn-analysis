import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def drop_unnecessary_columns(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame:
    """Drop columns that are not useful for the analysis."""
    return df.drop(columns=drop_columns, errors="ignore")


def encode_categorical_features(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Encode categorical features using LabelEncoder.

    Returns:
        df: DataFrame with encoded categorical columns.
        label_encoders: Dictionary mapping each feature to its LabelEncoder.
    """
    label_encoders = {}
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


def select_features_and_target(
    df: pd.DataFrame, selected_features: list, target: str
) -> (pd.DataFrame, pd.Series):
    """
    Select the specified features and target variable.

    Returns:
        X: DataFrame containing the selected features.
        y: Series containing the target variable.
    """
    X = df[selected_features]
    y = df[target]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """
    Split data into training and testing sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_naive_bayes_model(X_train: pd.DataFrame, y_train: pd.Series) -> GaussianNB:
    """Train a Gaussian Naïve Bayes model."""
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model


def evaluate_model(model: GaussianNB, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the model's performance.

    Returns:
        conf_matrix: Confusion matrix as a numpy array.
        report: Classification report as a string.
        y_pred: Predicted class labels.
        y_pred_prob: Predicted probabilities for the positive class.
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return conf_matrix, report, y_pred, y_pred_prob


def plot_confusion_matrix(conf_matrix: np.ndarray):
    """Plot the confusion matrix using seaborn."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Naïve Bayes Confusion Matrix")
    plt.show()


def plot_roc_curve(y_test: pd.Series, y_pred_prob: np.ndarray):
    """Plot the ROC curve (Lift/Gains Chart) for the model predictions."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, marker=".", label="Naïve Bayes")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Lift (Gains) Chart")
    plt.legend()
    plt.show()


def main():
    # Define file path and columns to drop
    file_path = "data/CustomerData_Composite-4.csv"
    drop_columns = [
        "customer_id",
        "churn_label",
        "churn_category",
        "churn_reason",
        "customer_status",
    ]

    # Load and preprocess the dataset
    df = load_data(file_path)
    df = drop_unnecessary_columns(df, drop_columns)
    df, label_encoders = encode_categorical_features(df)

    # Define selected features and target variable
    selected_features = [
        "satisfaction_score",
        "churn_score",
        "contract",
        "number_of_referrals",
        "tenure",
        "payment_method",
        "total_long_distance_charges",
        "internet_type",
        "monthly_ charges",  # Note: Ensure this column name is correct
        "total_charges",
    ]
    target = "churn_value"
    X, y = select_features_and_target(df, selected_features, target)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the Naïve Bayes model
    nb_model = train_naive_bayes_model(X_train, y_train)

    # Evaluate model performance
    conf_matrix, report, y_pred, y_pred_prob = evaluate_model(nb_model, X_test, y_test)

    # Print evaluation results
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)

    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Plot the ROC curve (Lift/Gains Chart)
    plot_roc_curve(y_test, y_pred_prob)


if __name__ == "__main__":
    main()
