import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame:
    """
    Drop unnecessary columns and encode categorical variables.
    Categorical variables are factorized.
    """
    df = df.drop(columns=drop_columns, errors="ignore")
    categorical_features = df.select_dtypes(include=["object"]).columns
    df[categorical_features] = df[categorical_features].apply(
        lambda col: pd.factorize(col)[0]
    )
    return df


def select_features_and_target(df: pd.DataFrame, selected_features: list, target: str):
    """
    Select the specified features and target variable.
    For this regression tree, the target is CLTV.
    """
    X = df[selected_features]
    y = df[target]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_regression_tree(
    X_train,
    y_train,
    max_depth: int = 7,
    min_samples_split: int = 20,
    random_state: int = 42,
) -> DecisionTreeRegressor:
    """Train a regression tree for predicting CLTV."""
    regressor = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_regression_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the regression model using common error measures:
    Mean Absolute Error (MAE), Mean Squared Error (MSE), Root MSE (RMSE), and R².
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = train_mse**0.5
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = test_mse**0.5
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        "Train MAE": train_mae,
        "Train MSE": train_mse,
        "Train RMSE": train_rmse,
        "Train R2": train_r2,
        "Test MAE": test_mae,
        "Test MSE": test_mse,
        "Test RMSE": test_rmse,
        "Test R2": test_r2,
    }


def plot_regression_tree(model, feature_names):
    """Plot a static visualization of the regression tree using matplotlib."""
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
    plt.title("Regression Tree for CLTV Prediction")
    plt.show()


def main():
    # Configuration
    file_path = "data/CustomerData_Composite-4.csv"  # Use the dataset provided in the assignment
    drop_columns = [
        "customer_id",
        "churn_label",
        "churn_category",
        "churn_reason",
        "customer_status",
    ]

    # Best features based on the provided Regression Tree Feature Importance table.
    best_features_reg = [
        "tenure",
        "churn_score",
        "total_population",
        "city",
        "longitude",
        "monthly_ charges",
        "age",
        "latitude",
        "avg_monthly_long_distance_charges",
        "zip_code",
        "total_charges",
        "total_long_distance_charges",
        "total_revenue",
        "avg_monthly_gb_download",
        "satisfaction_score",
    ]

    target = "cltv"  # Target variable for regression

    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_data(df, drop_columns)

    # Select features and target variable
    X, y = select_features_and_target(df, best_features_reg, target)

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the regression tree
    reg_tree = train_regression_tree(X_train, y_train)

    # Evaluate the regression tree on both training and validation data
    regression_stats = evaluate_regression_model(
        reg_tree, X_train, y_train, X_test, y_test
    )

    print("Regression Tree Model Evaluation Metrics:")
    for metric, value in regression_stats.items():
        print(f"{metric}: {value:.4f}")

    # Plot the regression tree for visualization
    plot_regression_tree(reg_tree, best_features_reg)


if __name__ == "__main__":
    main()
