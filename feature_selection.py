import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "data/CustomerData_Composite-4.csv"
df = pd.read_csv(file_path)

# Drop non-useful columns
drop_columns = [
    "customer_id",
    "churn_label",
    "churn_category",
    "churn_reason",
    "customer_status",
]
df = df.drop(columns=drop_columns, errors="ignore")

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Apply label encoding to categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define the target variables
X = df.drop(columns=["churn_value", "cltv"], errors="ignore")
y_classification = df["churn_value"]
y_regression = df["cltv"]

# Split into training and testing sets (80-20 split)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

# 1. Naïve Bayes - Compute Mutual Information Scores
mi_scores = mutual_info_classif(X_train_cls, y_train_cls)
mi_scores_df = pd.DataFrame(
    {"Feature": X_train_cls.columns, "MI Score": mi_scores}
).sort_values(by="MI Score", ascending=False)

# 2. Classification Tree - Feature Importances
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train_cls, y_train_cls)
tree_importances_df = pd.DataFrame(
    {"Feature": X_train_cls.columns, "Importance": clf_tree.feature_importances_}
).sort_values(by="Importance", ascending=False)

# 3. Regression Tree - Feature Importances for CLTV Prediction
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train_reg, y_train_reg)
reg_tree_importances_df = pd.DataFrame(
    {"Feature": X_train_reg.columns, "Importance": reg_tree.feature_importances_}
).sort_values(by="Importance", ascending=False)

# 4. Random Forest - Feature Importances
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_cls, y_train_cls)
rf_importances_df = pd.DataFrame(
    {"Feature": X_train_cls.columns, "Importance": rf_model.feature_importances_}
).sort_values(by="Importance", ascending=False)

# 5. Discriminant Analysis - F-Statistics for Feature Importance
f_scores, p_values = f_classif(X_train_cls, y_train_cls)
lda_f_stats_df = pd.DataFrame(
    {"Feature": X_train_cls.columns, "F-Score": f_scores, "P-Value": p_values}
).sort_values(by="F-Score", ascending=False)

# Display results
print("Mutual Information Scores (Naïve Bayes):\n", mi_scores_df)
print("\nDecision Tree Feature Importance:\n", tree_importances_df)
print("\nRegression Tree Feature Importance:\n", reg_tree_importances_df)
print("\nRandom Forest Feature Importance:\n", rf_importances_df)
print("\nDiscriminant Analysis F-Scores:\n", lda_f_stats_df)
