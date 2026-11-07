"""First-pass churn model. Reports validation AUC below -- feel free to
build on this, replace parts of it, or start over. Whatever you submit
just needs to write /app/predictions.json in the format described in the
task instructions."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CAT_COLS = ["contract_type", "payment_method", "autopay_enabled", "last_interaction_type"]
NUM_COLS = ["tenure_months", "monthly_charges", "total_charges",
            "support_tickets_90d", "avg_session_minutes", "days_since_last_login"]

df = pd.read_csv("/app/data/train.csv")
test_df = pd.read_csv("/app/data/test.csv")

# Quick sanity check: how similar is each categorical column's distribution
# between train and test? Large divergences here are worth understanding
# before trusting a feature.
print("train vs test category distribution (by column):")
for col in CAT_COLS:
    train_dist = df[col].value_counts(normalize=True)
    test_dist = test_df[col].value_counts(normalize=True)
    combined = pd.concat([train_dist, test_dist], axis=1, keys=["train", "test"]).fillna(0.0)
    combined["abs_diff"] = (combined["train"] - combined["test"]).abs()
    print(f"\n{col} (max abs diff: {combined['abs_diff'].max():.3f}):")
    print(combined.sort_values("abs_diff", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))

X = pd.get_dummies(df[NUM_COLS + CAT_COLS], columns=CAT_COLS)
y = df["churn"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"\nvalidation AUC: {val_auc:.4f}")
