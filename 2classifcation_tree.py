import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame:
    """
    Drop unnecessary columns and encode categorical variables.
    """
    df = df.drop(columns=drop_columns, errors="ignore")
    categorical_features = df.select_dtypes(include=["object"]).columns
    df[categorical_features] = df[categorical_features].apply(
        lambda col: pd.factorize(col)[0]
    )
    return df


def select_features(
    df: pd.DataFrame, selected_features: list, target: str
) -> (pd.DataFrame, pd.Series):
    """
    Select predefined best features for training.
    """
    X = df[selected_features]
    y = df[target]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_decision_tree(
    X_train,
    y_train,
    max_depth: int = 7,
    min_samples_split: int = 20,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier with given parameters."""
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_train, y_train, X_test, y_test):
    """
    Evaluate the classifier and return performance metrics.
    """
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    return train_accuracy, test_accuracy, conf_matrix


def plot_confusion_matrix(conf_matrix):
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def perform_grid_search(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Use GridSearchCV to fine-tune Decision Tree parameters.
    """
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "splitter": ["best", "random"],
    }
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)

    # Print the features used by GridSearchCV
    print("\nFeatures used in GridSearchCV:", X_train.columns.tolist())

    return grid_search.best_estimator_, grid_search


def main():
    # Configuration
    file_path = "data/CustomerData_Composite-4.csv"
    drop_columns = [
        "customer_id",
        "churn_label",
        "churn_category",
        "churn_reason",
        "customer_status",
    ]
    best_features = [
        "satisfaction_score",
        "churn_score",
        "contract",
        "online_security",
        "monthly_ charges",
        "internet_type",
        "total_revenue",
        "number_of_referrals",
    ]
    target = "churn_value"

    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_data(df, drop_columns)

    # Select features and target variable
    X, y = select_features(df, best_features, target)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train decision tree
    clf = train_decision_tree(X_train, y_train)
    train_acc, test_acc, conf_matrix = evaluate_model(
        clf, X_train, y_train, X_test, y_test
    )

    print("Initial Decision Tree Model:")
    print("Training Accuracy:", train_acc)
    print("Validation Accuracy:", test_acc)
    plot_confusion_matrix(conf_matrix)

    # Fine-Tune using GridSearchCV
    best_tree, grid_search = perform_grid_search(X_train, y_train)
    train_acc_best, test_acc_best, _ = evaluate_model(
        best_tree, X_train, y_train, X_test, y_test
    )

    print("\nFine-Tuned Decision Tree Model:")
    print("Best Parameters:", grid_search.best_params_)
    print("Training Accuracy with Best Parameters:", train_acc_best)
    print("Validation Accuracy with Best Parameters:", test_acc_best)


if __name__ == "__main__":
    main()
