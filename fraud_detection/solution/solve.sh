#!/bin/bash
set -euo pipefail

cat > /app/resolve.py << 'PYEOF'
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HIGH_RISK_MCC = {f"mcc_{i:02d}" for i in range(6)}


def seq_features(g):
    g = g.sort_values("timestamp")
    n = len(g)
    amounts = g["amount"].to_numpy()
    times = g["timestamp"].to_numpy()
    merchants = g["merchant"].tolist()
    devices = g["device"].tolist()
    locs = g["location"].tolist()
    mccs = g["mcc"].tolist()

    half = max(1, n // 2)
    first_half_amt = amounts[:half].mean()
    second_half_amt = amounts[half:].mean() if n > half else amounts[-1]
    amt_ratio = second_half_amt / (first_half_amt + 1e-6)

    k = min(6, n)
    recent_merchants = set(merchants[-k:])
    recent_devices = set(devices[-k:])
    recent_locs = set(locs[-k:])
    overall_merchants = set(merchants[:-k]) if n > k else set()
    overall_devices = set(devices[:-k]) if n > k else set()
    overall_locs = set(locs[:-k]) if n > k else set()
    new_merchant_frac = len(recent_merchants - overall_merchants) / k
    device_swap_recent = len(recent_devices - overall_devices) / k
    loc_swap_recent = len(recent_locs - overall_locs) / k

    max_velocity = 0
    for i in range(n):
        w = times[(times >= times[i]) & (times < times[i] + 1.0)]
        if len(w) > max_velocity:
            max_velocity = len(w)

    gaps = np.diff(times) if n > 1 else np.array([0.0])
    recent_gaps = gaps[-k:] if len(gaps) >= k else gaps
    early_gaps = gaps[:-k] if len(gaps) > k else gaps
    gap_ratio = (recent_gaps.mean() + 1e-6) / (early_gaps.mean() + 1e-6) if len(early_gaps) else 1.0

    idx = np.arange(n)
    slope = np.polyfit(idx, amounts, 1)[0] if n > 2 else 0.0

    recent_mcc = mccs[-k:]
    hr_mcc_recent = sum(1 for m in recent_mcc if m in HIGH_RISK_MCC) / k
    span = times[-1] - times[0] if n > 1 else 0.0

    return pd.Series({
        "n_events": n,
        "amt_mean": amounts.mean(),
        "amt_std": amounts.std(),
        "amt_sum": amounts.sum(),
        "amt_max": amounts.max(),
        "n_merchants": len(set(merchants)),
        "n_mcc": len(set(mccs)),
        "n_devices": len(set(devices)),
        "n_locs": len(set(locs)),
        "span": span,
        "rate": n / (span + 1e-6),
        "amt_ratio": amt_ratio,
        "new_merchant_frac": new_merchant_frac,
        "device_swap_recent": device_swap_recent,
        "loc_swap_recent": loc_swap_recent,
        "max_velocity": max_velocity,
        "gap_ratio": gap_ratio,
        "amt_slope": slope,
        "hr_mcc_recent": hr_mcc_recent,
    })


def build_features(events_df):
    return events_df.groupby("identity_id", sort=False).apply(seq_features, include_groups=False)


train_events = pd.read_csv("/app/data/train.csv")
test_events = pd.read_csv("/app/data/test.csv")

train_labels = train_events.groupby("identity_id")["label"].first()

X_train = build_features(train_events).loc[train_labels.index]
X_test = build_features(test_events)
test_ids = X_test.index.tolist()

model = RandomForestClassifier(
    n_estimators=400, max_depth=7, min_samples_leaf=5, random_state=0
)
model.fit(X_train, train_labels)

proba = model.predict_proba(X_test)[:, 1]
predictions = {cid: float(p) for cid, p in zip(test_ids, proba)}

with open("/app/predictions.json", "w") as f:
    json.dump(predictions, f)
PYEOF

python3 /app/resolve.py
