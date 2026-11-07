I work on the fraud team at a payments company, and we want to flag
identities showing early signs of synthetic-identity fraud from their
transaction history before it escalates. `/app/data/train.csv` and
`/app/data/test.csv` are long-format event logs, one row per transaction:
`identity_id`, `event_index` (order within that identity's sequence),
`timestamp` (float, since that identity's first event), `amount`,
`merchant`, `mcc` (merchant category code), `device`, `location`,
`channel`. `train.csv` also has `label` (1 = confirmed synthetic-identity
fraud, 0 = legitimate), repeated on every row for a given `identity_id`;
`test.csv` has no `label` — predicting it is the task. Sequence length
varies per identity (roughly a dozen to several dozen events), fraud
identities are a minority mixed in among legitimate ones, and nothing in
the schema flags which is which beyond `label` itself in train.

`/app/starter_pipeline.py` is a first-pass attempt that reports its own
validation AUC — use it as a starting point, replace it, or ignore it,
whatever actually generalizes to test rather than just looking good on a
validation split. No internet at runtime; `requirements.lock` covers
everything installed (numpy, pandas, scikit-learn). Write
`/app/resolve.py`, run as `python3 /app/resolve.py` with no arguments; it
should read both CSVs and write `/app/predictions.json` mapping every
`identity_id` in `test.csv` to a predicted fraud probability between 0 and
1, one entry per identity, each appearing exactly once, like
`{"id_202_00000": 0.04, "id_202_00001": 0.91}`.

Predictions are scored by ROC-AUC against the true held-out labels; you
need AUC ≥ 0.93 to pass.
