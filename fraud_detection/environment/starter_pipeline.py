"""First-pass anomaly model. Reports validation AUC below -- feel free to
build on this, replace parts of it, or start over. Whatever you submit
just needs to write /app/predictions.json in the format described in the
task instructions."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def summarize_identity(g):
    amounts = g["amount"].to_numpy()
    return pd.Series({
        "n_events": len(g),
        "amt_mean": amounts.mean(),
        "amt_std": amounts.std(),
        "amt_sum": amounts.sum(),
        "amt_max": amounts.max(),
        "n_merchants": g["merchant"].nunique(),
        "n_mcc": g["mcc"].nunique(),
        "n_devices": g["device"].nunique(),
        "n_locs": g["location"].nunique(),
        "span": g["timestamp"].max() - g["timestamp"].min(),
    })


train_events = pd.read_csv("/app/data/train.csv")
test_events = pd.read_csv("/app/data/test.csv")

train_labels = train_events.groupby("identity_id")["label"].first()
X = train_events.groupby("identity_id", sort=False).apply(summarize_identity, include_groups=False)
X = X.loc[train_labels.index]
y = train_labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"validation AUC (identity-level aggregate features): {val_auc:.4f}")
