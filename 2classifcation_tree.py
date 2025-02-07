import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame:
    """
    Drop unnecessary columns and encode categorical variables.

    Categorical variables are encoded using pandas.factorize.
    """
    df = df.drop(columns=drop_columns, errors="ignore")
    categorical_features = df.select_dtypes(include=["object"]).columns
    df[categorical_features] = df[categorical_features].apply(
        lambda col: pd.factorize(col)[0]
    )
    return df


def select_features(
    df: pd.DataFrame, selected_features: list, target: str
) -> (pd.DataFrame, pd.DataFrame, pd.Series):
    """
    Select features for an initial run and all features (excluding the target) for fine-tuning.

    Returns:
        X_selected: DataFrame with selected features.
        X_all: DataFrame with all features (except target) for tuning.
        y: Target variable.
    """
    X_selected = df[selected_features]
    X_all = df.drop(columns=[target], errors="ignore")
    y = df[target]
    return X_selected, X_all, y


def split_data(
    X_selected: pd.DataFrame,
    X_all: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataset into training and testing sets for both selected features and all features.

    Returns:
        X_train_selected, X_test_selected, X_train_all, X_test_all, y_train, y_test
    """
    X_train_selected, X_test_selected, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state
    )
    X_train_all, X_test_all, _, _ = train_test_split(
        X_all, y, test_size=test_size, random_state=random_state
    )
    return X_train_selected, X_test_selected, X_train_all, X_test_all, y_train, y_test


def train_initial_tree(
    X_train,
    y_train,
    max_depth: int = 5,
    min_samples_split: int = 10,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier with the given parameters using selected features."""
    clf = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_train, y_train, X_test, y_test):
    """
    Evaluate the given classifier on both training and testing sets.

    Returns:
        train_accuracy: Accuracy on the training set.
        test_accuracy: Accuracy on the testing set.
        conf_matrix: Confusion matrix for the test set.
    """
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    return train_accuracy, test_accuracy, conf_matrix


def plot_confusion_matrix(conf_matrix, title: str = "Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def display_feature_importances(features: list, importances: list, title: str):
    """Display feature importances as a sorted DataFrame."""
    importances_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importances_df = importances_df.sort_values(by="Importance", ascending=False)
    print(title)
    print(importances_df)


def perform_grid_search(
    X_train_all: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv: int = 5,
    random_state: int = 42,
):
    """
    Use GridSearchCV to fine-tune Decision Tree parameters with all available features.

    Returns:
        best_tree: Best estimator from grid search.
        grid_search: The GridSearchCV object.
    """
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid,
        cv=cv,
        scoring="accuracy",
    )
    grid_search.fit(X_train_all, y_train)
    return grid_search.best_estimator_, grid_search


def plot_decision_tree(
    clf: DecisionTreeClassifier,
    feature_names,
    class_names,
    figsize: tuple = (12, 8),
    title: str = "Decision Tree",
):
    """Plot the decision tree using sklearn's plot_tree function."""
    plt.figure(figsize=figsize)
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title(title)
    plt.show()


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
    selected_features = [
        "satisfaction_score",
        "churn_score",
        "contract",
        "online_security",
        "total_revenue",
        "number_of_referrals",
        "age",
        "city",
        "internet_type",
    ]
    target = "churn_value"

    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_data(df, drop_columns)

    # Select features and target variables
    X_selected, X_all, y = select_features(df, selected_features, target)

    # Split data for initial run and fine-tuning
    X_train_selected, X_test_selected, X_train_all, X_test_all, y_train, y_test = (
        split_data(X_selected, X_all, y)
    )

    # (i) Initial Run with Selected Features
    tree_clf = train_initial_tree(
        X_train_selected, y_train, max_depth=5, min_samples_split=10
    )
    train_acc, test_acc, conf_matrix_initial = evaluate_model(
        tree_clf, X_train_selected, y_train, X_test_selected, y_test
    )

    print("Initial Run with Selected Features:")
    print("Training Accuracy:", train_acc)
    print("Validation Accuracy:", test_acc)

    plot_confusion_matrix(conf_matrix_initial, title="Confusion Matrix (Initial Run)")
    display_feature_importances(
        selected_features,
        tree_clf.feature_importances_,
        "Feature Importances (Initial Run):",
    )

    # (ii) Fine-Tune Tree Parameters with All Features using GridSearchCV
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "splitter": ["best", "random"],
    }
    best_tree, grid_search = perform_grid_search(X_train_all, y_train, param_grid, cv=5)

    train_acc_best, test_acc_best, _ = evaluate_model(
        best_tree, X_train_all, y_train, X_test_all, y_test
    )

    print("\nFine-Tuned Run with All Features:")
    print("Best Parameters:", grid_search.best_params_)
    print("Training Accuracy with Best Parameters:", train_acc_best)
    print("Validation Accuracy with Best Parameters:", test_acc_best)

    # Display feature importances for the fine-tuned tree
    fine_tuned_importances = pd.DataFrame(
        {"Feature": X_all.columns, "Importance": best_tree.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    print("\nFeature Importances (Fine-Tuned Tree):")
    print(fine_tuned_importances)

    # Plot the fine-tuned decision tree
    plot_decision_tree(
        best_tree,
        feature_names=X_all.columns,
        class_names=["No Churn", "Churn"],
        title="Fine-Tuned Decision Tree (All Features)",
    )


if __name__ == "__main__":
    main()
