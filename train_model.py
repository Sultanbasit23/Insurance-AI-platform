import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ====================================================
# 1️⃣ COST PREDICTION DATA (Regression)
# ====================================================

cost_data = {
    "age": [25, 45, 35, 60, 29, 50, 40, 33],
    "bmi": [22, 30, 25, 28, 21, 27, 26, 24],
    "children": [0, 2, 1, 3, 0, 2, 1, 1],
    "sex_male": [1, 1, 0, 1, 0, 1, 1, 0],
    "smoker_yes": [1, 0, 0, 1, 0, 1, 0, 0],
    "region_northwest": [0, 1, 0, 0, 1, 0, 0, 1],
    "region_southeast": [1, 0, 1, 0, 0, 1, 0, 0],
    "region_southwest": [0, 0, 0, 1, 0, 0, 1, 0],
    "cost": [12000, 25000, 18000, 40000, 15000, 32000, 20000, 17000]
}

df_cost = pd.DataFrame(cost_data)

X_cost = df_cost.drop("cost", axis=1)
y_cost = df_cost["cost"]

cost_model = RandomForestRegressor(n_estimators=200, random_state=42)
cost_model.fit(X_cost, y_cost)

joblib.dump(cost_model, "cost_model.joblib")

print("✅ Cost model saved")


# ====================================================
# 2️⃣ CLAIM APPROVAL DATA (Classification)
# ====================================================

claim_data = {
    "age": [25, 45, 35, 60, 29, 50, 40, 33],
    "policy_type": [1, 2, 1, 3, 2, 3, 1, 2],
    "claim_amount": [50000, 200000, 75000, 300000, 60000, 150000, 90000, 120000],
    "hospital_days": [3, 10, 5, 15, 4, 8, 6, 7],
    "pre_existing": [0, 1, 0, 1, 0, 1, 0, 1],
    "approved": [1, 0, 1, 0, 1, 0, 1, 0]
}

df_claim = pd.DataFrame(claim_data)

X_claim = df_claim.drop("approved", axis=1)
y_claim = df_claim["approved"]

claim_model = RandomForestClassifier(n_estimators=200, random_state=42)
claim_model.fit(X_claim, y_claim)

joblib.dump(claim_model, "claim_model.joblib")

print("✅ Claim model saved")


# ====================================================
# 3️⃣ FRAUD MODEL
# ====================================================

fraud_model = RandomForestClassifier(n_estimators=200, random_state=42)
fraud_model.fit(X_claim, y_claim)

joblib.dump(fraud_model, "fraud_model.joblib")

print("✅ Fraud model saved")


print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY")





