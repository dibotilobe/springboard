import csv
import json
import subprocess
from pathlib import Path

import pytest
from sklearn.metrics import roc_auc_score

RESOLVE_PATH = Path("/app/resolve.py")
TRAIN_PATH = Path("/app/data/train.csv")
TEST_PATH = Path("/app/data/test.csv")
PREDICTIONS_PATH = Path("/app/predictions.json")
HIDDEN_LABELS_PATH = Path(__file__).parent / "test_labels.json"

MIN_AUC = 0.93


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_resolve_script_exists():
    assert RESOLVE_PATH.is_file(), "/app/resolve.py was not created"


def test_input_files_intact():
    train_rows = _read_csv(TRAIN_PATH)
    test_rows = _read_csv(TEST_PATH)
    train_ids = {r["identity_id"] for r in train_rows}
    test_ids = {r["identity_id"] for r in test_rows}
    assert len(train_rows) == 94367
    assert len(test_rows) == 35186
    assert len(train_ids) == 4000
    assert len(test_ids) == 1500
    assert "label" in train_rows[0]
    assert "label" not in test_rows[0]


@pytest.fixture(scope="module")
def predictions():
    if PREDICTIONS_PATH.exists():
        PREDICTIONS_PATH.unlink()

    result = subprocess.run(
        ["python3", str(RESOLVE_PATH)],
        cwd="/app",
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"resolve.py exited non-zero.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert PREDICTIONS_PATH.is_file(), "/app/predictions.json was not written"

    with open(PREDICTIONS_PATH) as f:
        return json.load(f)


def test_predictions_cover_every_test_identity(predictions):
    test_rows = _read_csv(TEST_PATH)
    expected_ids = {r["identity_id"] for r in test_rows}

    missing = expected_ids - predictions.keys()
    assert not missing, f"{len(missing)} identity(ies) missing from predictions, e.g. {sorted(missing)[:5]}"

    extra = predictions.keys() - expected_ids
    assert not extra, f"predictions contain unknown identity_id(s), e.g. {sorted(extra)[:5]}"


def test_predictions_are_valid_probabilities(predictions):
    for iid, p in predictions.items():
        assert isinstance(p, (int, float)), f"prediction for {iid} is not a number: {p!r}"
        assert 0.0 <= float(p) <= 1.0, f"prediction for {iid} out of [0, 1] range: {p}"


def test_auc_meets_threshold(predictions):
    with open(HIDDEN_LABELS_PATH) as f:
        true_labels = json.load(f)

    ids = sorted(true_labels.keys())
    y_true = [true_labels[iid] for iid in ids]
    y_score = [float(predictions[iid]) for iid in ids]

    auc = roc_auc_score(y_true, y_score)
    assert auc >= MIN_AUC, f"AUC {auc:.4f} below required threshold {MIN_AUC}"
