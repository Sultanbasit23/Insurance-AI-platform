import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# -----------------------------
# Generate Dummy Fraud Dataset
# -----------------------------

np.random.seed(42)

rows = 2000

data = pd.DataFrame({

    "age": np.random.randint(18, 80, rows),

    "policy_type": np.random.randint(1, 4, rows),

    "claim_amount": np.random.randint(5000, 500000, rows),

    "hospital_days": np.random.randint(1, 30, rows),

    "pre_existing": np.random.randint(0, 2, rows)

})


# Fraud logic (for training labels)
def label(row):

    if row["claim_amount"] > 250000:
        return 1

    if row["hospital_days"] > 18:
        return 1

    if row["age"] < 25 and row["claim_amount"] > 150000:
        return 1

    if row["pre_existing"] == 1 and row["claim_amount"] > 200000:
        return 1

    return 0


data["fraud"] = data.apply(label, axis=1)


# -----------------------------
# Train Model
# -----------------------------

X = data.drop("fraud", axis=1)
y = data["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluation
# -----------------------------

pred = model.predict(X_test)

print(classification_report(y_test, pred))


# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "fraud_model.joblib")

print("✅ Fraud model saved")
