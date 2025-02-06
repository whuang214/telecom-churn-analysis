import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/CustomerData_Composite-4.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
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

# Apply label encoding to categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select top features based on Mutual Information Scores
selected_features = [
    "satisfaction_score",
    "churn_score",
    "contract",
    "number_of_referrals",
    "tenure",
    "payment_method",
    "total_long_distance_charges",
    "internet_type",
    "monthly_ charges",
    "total_charges",
]

X = df[selected_features]
y = df["churn_value"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test)
y_pred_prob = nb_model.predict_proba(X_test)[:, 1]

# Evaluate model performance
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

# Plot Confusion Matrix
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

# Lift (Gains) Chart
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, marker=".", label="Naïve Bayes")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Lift (Gains) Chart")
plt.legend()
plt.show()
