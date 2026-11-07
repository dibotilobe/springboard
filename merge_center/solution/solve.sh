#!/bin/bash
set -euo pipefail

cat > /app/resolve.py << 'PYEOF'
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

NUM_COLS = ["tenure_months", "monthly_charges", "total_charges",
            "support_tickets_90d", "avg_session_minutes", "days_since_last_login"]
CAT_COLS = ["contract_type", "payment_method", "autopay_enabled"]
# NOTE: last_interaction_type is deliberately excluded -- it is a post-hoc
# field (a customer's most recent contact type) that is only meaningfully
# populated in historical training data; for currently-active customers in
# test.csv it does not yet reflect any future outcome, so it carries no
# real signal there and only adds noise to a model that leans on it.

train = pd.read_csv("/app/data/train.csv")
test = pd.read_csv("/app/data/test.csv")

X_train = pd.get_dummies(train[NUM_COLS + CAT_COLS], columns=CAT_COLS)
X_test = pd.get_dummies(test[NUM_COLS + CAT_COLS], columns=CAT_COLS)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train["churn"]

model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=0)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
predictions = {cid: float(p) for cid, p in zip(test["customer_id"], proba)}

with open("/app/predictions.json", "w") as f:
    json.dump(predictions, f)
PYEOF

python3 /app/resolve.py
