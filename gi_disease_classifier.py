
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# STEP 1: Generate Simulated GI Dataset
np.random.seed(42)
n_samples = 1500

gi_data = {
    "Age": np.random.randint(25, 80, size=n_samples),
    "BMI": np.random.uniform(22, 40, size=n_samples),
    "Fecal_Calprotectin_ug_g": np.random.uniform(
        10, 600, size=n_samples
    ),
    "Microbiome_Index": np.random.uniform(
        0.1, 1.0, size=n_samples
    ),
    "H_Pylori_Status": np.random.choice(
        ["Negative", "Positive"], size=n_samples
    ),
    "Diet_Type": np.random.choice(
        ["Omnivore", "Vegetarian", "Low-FODMAP"],
        size=n_samples
    ),
    "Food_Intolerance": np.random.choice(
        ["None", "Lactose_Gluten"], size=n_samples
    ),
    "Stress_Level": np.random.choice(
        ["Low", "Moderate", "High"], size=n_samples
    ),
    "NSAID_Use": np.random.choice(
        ["No", "Yes"], size=n_samples
    ),
}

df = pd.DataFrame(gi_data)


# STEP 2: Generate Ground-Truth Outcome
risk_score = (
    (df["Fecal_Calprotectin_ug_g"] * 0.008)
    + (df["H_Pylori_Status"] == "Positive").astype(int) * 1.5
    + (df["Food_Intolerance"] == "Lactose_Gluten").astype(int) * 1.2
    + (df["Stress_Level"] == "High").astype(int) * 1.0
    + (df["NSAID_Use"] == "Yes").astype(int) * 0.8
    - (df["Diet_Type"] == "Low-FODMAP").astype(int) * 0.7
    - (df["Microbiome_Index"] * 2.0)
    + np.random.normal(0, 0.7, size=n_samples)
)

df["GI_Disease_Outcome"] = (risk_score > 2.5).astype(int)

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["GI_Disease_Outcome"].value_counts())


# STEP 3: Data Preprocessing
X = df.drop(columns=["GI_Disease_Outcome"])
y = df["GI_Disease_Outcome"]

categorical_cols = [
    "H_Pylori_Status",
    "Diet_Type",
    "Food_Intolerance",
    "Stress_Level",
    "NSAID_Use"
]

X_encoded = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# STEP 4: Define Four Machine Learning Models
models = {
    "LightGBM": LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
}


# STEP 5: Train and Evaluate the Models
print("\n=== Gastrointestinal Disease Classifier Metrics ===\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))


# STEP 6: Helper Function for SHAP Values
def get_2d_shap_values(shap_exp):

    if len(shap_exp.shape) == 3:
        return shap_exp[:, :, 1]

    return shap_exp


# STEP 7: SHAP Summary Plot for LightGBM
print("\nGenerating SHAP Plot for LightGBM...")

lgb_explainer = shap.TreeExplainer(models["LightGBM"])
lgb_shap = get_2d_shap_values(
    lgb_explainer(X_test)
)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(lgb_shap, show=False)
plt.title("SHAP Feature Impact: LightGBM")
plt.tight_layout()
plt.show()


# STEP 8: SHAP Summary Plot for Random Forest
print("\nGenerating SHAP Plot for Random Forest...")

rf_explainer = shap.TreeExplainer(models["Random Forest"])
rf_shap = get_2d_shap_values(
    rf_explainer(X_test)
)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(rf_shap, show=False)
plt.title("SHAP Feature Impact: Random Forest")
plt.tight_layout()
plt.show()


# STEP 9: SHAP Summary Plot for XGBoost
print("\nGenerating SHAP Plot for XGBoost...")

xgb_explainer = shap.TreeExplainer(models["XGBoost"])
xgb_shap = get_2d_shap_values(
    xgb_explainer(X_test)
)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(xgb_shap, show=False)
plt.title("SHAP Feature Impact: XGBoost")
plt.tight_layout()
plt.show()


# STEP 10: SHAP Summary Plot for Logistic Regression
print("\nGenerating SHAP Plot for Logistic Regression...")

lr_model = models["Logistic Regression"].named_steps[
    "logisticregression"
]

scaler = models["Logistic Regression"].named_steps[
    "standardscaler"
]

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

lr_explainer = shap.LinearExplainer(
    lr_model,
    X_train_scaled
)

lr_shap = get_2d_shap_values(
    lr_explainer(X_test_scaled)
)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(lr_shap, show=False)
plt.title("SHAP Feature Impact: Logistic Regression")
plt.tight_layout()
plt.show()


# STEP 11: Display Final Dataset
print("\n=== Sample Dataset ===")
print(df.head())

print("\nAll models trained and SHAP plots generated.")
