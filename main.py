
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------
# 1. Create Project Folders
# --------------------------------

os.makedirs("dataset", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 55)
print(" TELECOM CUSTOMER CHURN PREDICTION ")
print("=" * 55)

# --------------------------------
# 2. Load Excel Dataset
# --------------------------------

data_path = "dataset/customer_churn_excel.xlsx"

if not os.path.exists(data_path):
    raise FileNotFoundError(
        "Excel file not found! Place it inside dataset folder."
    )

df = pd.read_excel(
    data_path,
    sheet_name="Customer Churn",
    engine="openpyxl"
)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

# --------------------------------
# 3. Data Preprocessing
# --------------------------------

print("\nPreprocessing Dataset...")

df.columns = df.columns.str.strip()

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Convert target column into numbers
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df = df.dropna(subset=["Churn"]).copy()
df["Churn"] = df["Churn"].astype(int)

# Remove customerID because it is an identifier
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("Preprocessing Completed!")
print("\nChurn Distribution:")
print(y.value_counts())

# --------------------------------
# 4. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# --------------------------------
# 5. Identify Column Types
# --------------------------------

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# --------------------------------
# 6. Create Preprocessing Function
# --------------------------------

def create_preprocessor():

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

# --------------------------------
# 7. Create Machine Learning Models
# --------------------------------

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

class_weight_ratio = negative / positive

random_forest = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

xgboost = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

models = {
    "Random Forest": random_forest,
    "XGBoost": xgboost
}

# --------------------------------
# 8. Cross Validation
# --------------------------------

print("\nPerforming 5-Fold Cross Validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )

    print(
        name,
        "CV ROC-AUC:",
        round(scores.mean(), 4)
    )

# --------------------------------
# 9. Train and Evaluate Models
# --------------------------------

results = {}
probabilities = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "auc": auc,
        "predictions": y_pred
    }

    probabilities[name] = y_prob

    print("Accuracy:", round(accuracy, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# --------------------------------
# 10. Generate ROC Curve
# --------------------------------

plt.figure(figsize=(8, 6))

for name, y_prob in probabilities.items():

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Customer Churn ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_curve.png")
plt.close()

# --------------------------------
# 11. Select Best Model
# --------------------------------

best_name = max(
    results,
    key=lambda name: results[name]["auc"]
)

best_model = results[best_name]["model"]
best_pred = results[best_name]["predictions"]

print("\nBest Model:", best_name)

# --------------------------------
# 12. Confusion Matrix
# --------------------------------

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# --------------------------------
# 13. Feature Importance
# --------------------------------

fitted_preprocessor = best_model.named_steps["preprocessor"]
classifier = best_model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
importance = classifier.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

top_features = feature_df.sort_values(
    "Importance",
    ascending=False
).head(10)

print("\nTop 10 Important Features:")
print(top_features.to_string(index=False))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# --------------------------------
# 14. Save Trained Models
# --------------------------------

for name, result in results.items():

    filename = name.lower().replace(" ", "_")

    joblib.dump(
        result["model"],
        f"models/{filename}.pkl"
    )

# --------------------------------
# 15. Save Model Comparison
# --------------------------------

summary = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": result["accuracy"],
        "ROC_AUC": result["auc"]
    }
    for name, result in results.items()
])

summary.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(summary.to_string(index=False))

print("\nBest Model:", best_name)
print("Best ROC-AUC:", round(results[best_name]["auc"], 4))

if results[best_name]["auc"] > 0.80:
    print("Target Achieved: ROC-AUC above 80%")
else:
    print("Target Not Achieved: Further tuning required.")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Check outputs and models folders.")
