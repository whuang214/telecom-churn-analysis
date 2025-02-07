import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def drop_non_useful_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop columns that are not useful for modeling."""
    return df.drop(columns=columns, errors="ignore")


def encode_categorical_features(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Identify and encode categorical features using Label Encoding.

    Returns:
        df: DataFrame with encoded categorical columns.
        label_encoders: Dictionary of LabelEncoders for each encoded column.
    """
    label_encoders = {}
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


def split_data(
    df: pd.DataFrame,
    target_class: str,
    target_reg: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the data into training and testing sets for both classification and regression tasks.

    Returns:
        X_train_cls, X_test_cls, y_train_cls, y_test_cls,
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    """
    X = df.drop(columns=[target_class, target_reg], errors="ignore")
    y_classification = df[target_class]
    y_regression = df[target_reg]

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_classification, test_size=test_size, random_state=random_state
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=test_size, random_state=random_state
    )
    return (
        X_train_cls,
        X_test_cls,
        y_train_cls,
        y_test_cls,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    )


def compute_mutual_information(
    X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Compute Mutual Information Scores for feature selection (Naïve Bayes perspective)."""
    mi_scores = mutual_info_classif(X_train, y_train)
    mi_scores_df = pd.DataFrame({"Feature": X_train.columns, "MI Score": mi_scores})
    return mi_scores_df.sort_values(by="MI Score", ascending=False)


def compute_decision_tree_importance(
    X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Compute feature importance using a Decision Tree Classifier."""
    clf_tree = DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train, y_train)
    importances_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": clf_tree.feature_importances_}
    )
    return importances_df.sort_values(by="Importance", ascending=False)


def compute_decision_tree_regression_importance(
    X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """
    Compute feature importance using a Decision Tree Regressor for CLTV prediction.

    Improvements:
    - Handles missing values in the target (`y_train`).
    - Ensures feature importance scores are normalized (sum to 1).
    - Returns a sorted DataFrame with clearer importance interpretation.

    Returns:
        pd.DataFrame: Feature importance rankings sorted in descending order.
    """
    # Handle missing values in the target variable
    valid_indices = y_train.dropna().index
    X_train_clean = X_train.loc[valid_indices]
    y_train_clean = y_train.loc[valid_indices]

    # Initialize and train Decision Tree Regressor
    reg_tree = DecisionTreeRegressor(random_state=42)
    reg_tree.fit(X_train_clean, y_train_clean)

    # Get feature importance values
    feature_importance = reg_tree.feature_importances_

    # Create a DataFrame with feature importance scores
    importances_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": feature_importance}
    )

    # Normalize importance scores for better interpretability
    importances_df["Normalized Importance"] = (
        importances_df["Importance"] / importances_df["Importance"].sum()
    )

    # Sort by importance in descending order
    return importances_df.sort_values(by="Importance", ascending=False).reset_index(
        drop=True
    )


def compute_random_forest_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "accuracy",
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute feature importance using a Random Forest Classifier and permutation importance.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        scoring (str): Metric to evaluate performance drop (default 'accuracy').
        n_repeats (int): Number of times to permute a feature to get stable estimates.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with features and their permutation importance scores,
                      sorted in descending order.
    """
    # Fit the Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Compute permutation importance
    perm_importance = permutation_importance(
        rf_model,
        X_train,
        y_train,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    # Create DataFrame of permutation importance scores
    importances_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": perm_importance.importances_mean}
    )

    return importances_df.sort_values(by="Importance", ascending=False)


def compute_f_statistics(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Compute F-statistics and corresponding p-values for features using discriminant analysis.
    Useful for ranking features in a classification context.
    """
    f_scores, p_values = f_classif(X_train, y_train)
    f_stats_df = pd.DataFrame(
        {"Feature": X_train.columns, "F-Score": f_scores, "P-Value": p_values}
    )
    return f_stats_df.sort_values(by="F-Score", ascending=False)


def display_results(results: dict):
    """Print out the DataFrames containing the computed feature importance scores."""
    for title, df in results.items():
        print(f"\n{title}:\n", df)


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

    # Load and preprocess data
    df = load_data(file_path)
    df = drop_non_useful_columns(df, drop_columns)
    df, label_encoders = encode_categorical_features(df)

    # Define target variables for classification and regression
    target_class = "churn_value"
    target_reg = "cltv"

    # Split the data
    (
        X_train_cls,
        X_test_cls,
        y_train_cls,
        y_test_cls,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    ) = split_data(df, target_class, target_reg)

    # Compute various feature importance metrics
    results = {
        "Mutual Information Scores (Naïve Bayes)": compute_mutual_information(
            X_train_cls, y_train_cls
        ),
        "Decision Tree Feature Importance": compute_decision_tree_importance(
            X_train_cls, y_train_cls
        ),
        "Regression Tree Feature Importance": compute_decision_tree_regression_importance(
            X_train_reg, y_train_reg
        ),
        "Random Forest Feature Importance": compute_random_forest_importance(
            X_train_cls, y_train_cls
        ),
        "Discriminant Analysis F-Scores": compute_f_statistics(
            X_train_cls, y_train_cls
        ),
    }

    # Display the computed results
    display_results(results)


if __name__ == "__main__":
    main()
