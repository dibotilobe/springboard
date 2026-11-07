# Customer Churn Prediction

You work on the data team at a subscription service. The company wants to
predict which currently-active customers are likely to cancel (churn) so
the retention team can prioritize outreach. Build a model that scores each
customer in the test set by churn probability.

## Data

`/app/data/train.csv` contains historical customers with a known outcome.
Columns:

- `customer_id`: unique identifier
- `tenure_months`: months as a customer
- `monthly_charges`: current monthly bill
- `total_charges`: total billed to date
- `support_tickets_90d`: number of support tickets in the last 90 days
- `avg_session_minutes`: average session length in the product
- `days_since_last_login`: days since the customer last logged in
- `contract_type`: `month-to-month`, `one-year`, or `two-year`
- `payment_method`: `credit_card`, `bank_transfer`, `electronic_check`, or
  `mailed_check`
- `autopay_enabled`: `yes` or `no`
- `last_interaction_type`: the type of the customer's most recently
  recorded contact (e.g. `support_ticket`, `renewal_confirmation`,
  `cancellation_request`, `billing_dispute`, `product_inquiry`, `none`)
- `churn`: `1` if the customer canceled, `0` if they stayed

`/app/data/test.csv` contains a different set of currently-active customers
with the same columns except `churn`, which is what you're predicting.

A first-pass attempt at this problem is provided at
`/app/starter_pipeline.py`, along with the validation AUC it reports when
run. You can use it as a starting point, modify it, or ignore it and build
your own approach — whatever gets you the best genuinely-generalizing
model on the test set.

## Deliverable

Write `/app/resolve.py`. Invoke it as `python3 /app/resolve.py`, no
arguments; it reads `/app/data/train.csv` and `/app/data/test.csv` and
writes `/app/predictions.json`:

```json
{
  "cust_04000": 0.13,
  "cust_04001": 0.82
}
```

One entry per `customer_id` in `test.csv`, mapping to your predicted churn
probability (a number between 0 and 1). Every `customer_id` in `test.csv`
must appear exactly once.

## Grading

Your predictions are scored by ROC-AUC against the true (held-out) churn
outcomes. You need AUC ≥ 0.70 to pass.
