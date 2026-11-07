import json
import os
import subprocess
from pathlib import Path

import pytest

DOCS_DIR = Path("/app/data/docs")
EVAL_QUERIES_PATH = Path("/app/data/eval_queries.json")
DEV_QUERIES_PATH = Path("/app/data/dev_queries.json")
RESOLVE_PATH = Path("/app/resolve.py")
PREDICTIONS_PATH = Path("/app/predictions.json")
HIDDEN_LABELS_PATH = Path(__file__).parent / "test_labels.json"

MIN_RECALL_AT_K = 0.45
MIN_MRR = 0.30
MIN_BRIDGE_RECALL_AT_K = 0.65
MIN_PRECISION_ACC = 0.65
K = 3

# Must stay in sync with reference/generate_kb.py's PRECISION_SUBSET_PIDS --
# one single-hop passage per category, each with a genuinely close
# same-category sibling that real embeddings have been observed confusing
# it with. There's no schema field carrying this from generator to test;
# it's a hardcoded pid allowlist kept identical in both places on purpose
# (same pattern as the len(gold_pids) split for single-hop vs bridge:
# never expose the split via naming, only via a fixed, internal set).
PRECISION_SUBSET_PIDS = {
    "acct_lockout",
    "bill_downgrade_plan",
    "sync_manual_trigger",
    "notif_mute",
    "export_scheduled",
    "team_remove_member",
    "api_webhook_setup",
    "sec_ip_allowlist",
    "mobile_offline_mode",
}


def _load_docs():
    docs = {}
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.lower().endswith(".txt"):
            pid = fname[:-4]
            with open(DOCS_DIR / fname, encoding="utf-8") as f:
                docs[pid] = f.read()
    return docs


def _normalize_ws(s):
    return " ".join(s.split())


def _source_passages_for_chunk(chunk_text, docs):
    """Which known passage(s) is this retrieved chunk text drawn from,
    identified by substring containment -- independent of however the
    agent chose to chunk (whole passage, partial excerpt, or otherwise)."""
    norm_chunk = _normalize_ws(chunk_text)
    if not norm_chunk:
        return set()
    matches = set()
    for pid, text in docs.items():
        if norm_chunk in _normalize_ws(text):
            matches.add(pid)
    return matches


def test_resolve_script_exists():
    """Verify the agent created /app/resolve.py as required."""
    assert RESOLVE_PATH.is_file(), "/app/resolve.py was not created"


def test_input_files_intact():
    """Verify the agent didn't delete or truncate the provided corpus or
    eval query set -- a sanity check independent of retrieval quality."""
    docs = _load_docs()
    assert len(docs) == 36, f"expected 36 source passages, found {len(docs)}"

    with open(EVAL_QUERIES_PATH) as f:
        eval_queries = json.load(f)
    assert len(eval_queries) == 54, f"expected 54 eval queries, found {len(eval_queries)}"
    sample_eval = next(iter(eval_queries.values()))
    assert "gold_passage_ids" not in sample_eval, (
        "eval_queries.json must not contain gold labels"
    )
    assert "suggested_passage_id" in sample_eval

    with open(DEV_QUERIES_PATH) as f:
        dev_queries = json.load(f)
    assert len(dev_queries) == 54, f"expected 54 dev queries, found {len(dev_queries)}"
    sample_dev = next(iter(dev_queries.values()))
    assert "gold_passage_ids" in sample_dev
    assert isinstance(sample_dev["gold_passage_ids"], list), (
        "gold_passage_ids must be a list -- some dev queries require two "
        "passages, not one"
    )
    assert "suggested_passage_id" in sample_dev

    # Confirm the dev set actually demonstrates the multi-hop pattern (some
    # entries have 2 gold passages, not just 1) -- if this ever regresses to
    # all-length-1, the labeled set stops teaching the pattern it's meant to.
    gold_list_lengths = {len(info["gold_passage_ids"]) for info in dev_queries.values()}
    assert gold_list_lengths == {1, 2}, (
        f"expected dev gold_passage_ids to show both single-hop (length 1) and "
        f"multi-hop (length 2) queries, found lengths {sorted(gold_list_lengths)}"
    )


@pytest.fixture(scope="module")
def predictions():
    if PREDICTIONS_PATH.exists():
        PREDICTIONS_PATH.unlink()

    result = subprocess.run(
        ["python3", str(RESOLVE_PATH)],
        cwd="/app",
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"resolve.py exited non-zero.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert PREDICTIONS_PATH.is_file(), "/app/predictions.json was not written"

    with open(PREDICTIONS_PATH) as f:
        return json.load(f)


def test_predictions_cover_every_eval_query(predictions):
    with open(EVAL_QUERIES_PATH) as f:
        eval_queries = json.load(f)
    expected_ids = set(eval_queries.keys())

    missing = expected_ids - predictions.keys()
    assert not missing, f"{len(missing)} query id(s) missing from predictions, e.g. {sorted(missing)[:5]}"

    extra = predictions.keys() - expected_ids
    assert not extra, f"predictions contain unknown query id(s), e.g. {sorted(extra)[:5]}"


def test_predictions_are_ranked_text_lists(predictions):
    for qid, chunks in predictions.items():
        assert isinstance(chunks, list), f"prediction for {qid} is not a list"
        assert 1 <= len(chunks) <= K, (
            f"prediction for {qid} has {len(chunks)} entries, expected between 1 and {K}"
        )
        for chunk in chunks:
            assert isinstance(chunk, str) and chunk.strip(), (
                f"prediction for {qid} contains a non-string or empty entry: {chunk!r}"
            )


def test_predictions_are_traceable_to_corpus(predictions):
    """Retrieved chunks must be actual excerpts of the source passages, not
    generated or hallucinated text -- retrieval means finding existing
    content, not synthesizing new content."""
    docs = _load_docs()
    untraceable = []
    for qid, chunks in predictions.items():
        for chunk in chunks:
            if not _source_passages_for_chunk(chunk, docs):
                untraceable.append((qid, chunk[:80]))

    assert not untraceable, (
        f"{len(untraceable)} retrieved chunk(s) could not be traced to any source "
        f"passage, e.g. {untraceable[:3]}"
    )


def _reciprocal_rank(chunks, gold_pids, docs):
    """1/rank of the first chunk (1-indexed, in the submission's own order)
    that traces back to ANY of gold_pids, or 0.0 if none ever appear."""
    gold_set = set(gold_pids)
    for rank, chunk in enumerate(chunks, start=1):
        if _source_passages_for_chunk(chunk, docs) & gold_set:
            return 1.0 / rank
    return 0.0


def _blended_recall_and_mrr(predictions, gold, docs, k=K):
    """Both overall metrics are a 50/50 BLEND of the single-hop average and
    the multi-hop (bridge) average, not a flat macro-average over all
    queries. With only ~10 of 46 eval queries requiring two passages, a
    flat average lets bridge performance get diluted away almost entirely
    -- measured directly: a submission with ZERO multi-hop handling (embeds
    the whole query, ignores that some questions need two passages) scored
    HIGHER on a flat macro-average than a decomposition-aware submission,
    because the 36-query single-hop majority swamped the signal either way.
    The 50/50 blend fixes this by structurally guaranteeing multi-hop
    performance is exactly half of both scores, regardless of how many
    bridge queries exist. Which group a query belongs to is determined only
    from len(gold_pids) -- not from any id pattern, which is deliberately
    not exposed (see reference/generate_kb.py for why an earlier id-prefix
    scheme was itself a leak).

    Also returns bridge_recall_avg on its own (not blended) -- on a real
    Docker run the blended metric alone left only a couple points of margin
    between "handles multi-hop" and "doesn't" (both comfortably clearing
    the blended threshold either way), while the unblended bridge-only
    Recall@K gap was far larger and more reliable (a decomposition-aware
    reference scored 0.600 vs a whole-query-only ablation's 0.450 on the
    connector-diverse bridge set), which is why
    test_bridge_recall_meets_threshold gates on it directly instead of
    trusting the blend alone."""
    single_recall, single_rr, n_single = 0.0, 0.0, 0
    bridge_recall, bridge_rr, n_bridge = 0.0, 0.0, 0
    for qid, gold_pids in gold.items():
        chunks = predictions.get(qid, [])
        retrieved_pids = set()
        for chunk in chunks:
            retrieved_pids |= _source_passages_for_chunk(chunk, docs)
        gold_set = set(gold_pids)
        recall = len(retrieved_pids & gold_set) / len(gold_set)
        rr = _reciprocal_rank(chunks, gold_pids, docs)
        if len(gold_set) == 1:
            single_recall += recall
            single_rr += rr
            n_single += 1
        else:
            bridge_recall += recall
            bridge_rr += rr
            n_bridge += 1
    single_recall_avg = single_recall / n_single if n_single else 0.0
    single_rr_avg = single_rr / n_single if n_single else 0.0
    bridge_recall_avg = bridge_recall / n_bridge if n_bridge else 0.0
    bridge_rr_avg = bridge_rr / n_bridge if n_bridge else 0.0
    recall_blend = 0.5 * single_recall_avg + 0.5 * bridge_recall_avg
    mrr_blend = 0.5 * single_rr_avg + 0.5 * bridge_rr_avg
    return recall_blend, mrr_blend, n_single, n_bridge, bridge_recall_avg


def test_mrr_meets_threshold(predictions):
    """Recall@K alone only checks set membership in the top-K, so a
    submission could return a correct passage buried in last place every
    time and still pass. Mean Reciprocal Rank additionally requires that,
    when a correct passage is found, it's ranked meaningfully high -- i.e.
    the returned order actually reflects relevance, as the instructions
    require ('ordered most to least relevant'). For queries with two gold
    passages, MRR is based on the rank of whichever one is found first.
    See _blended_recall_and_mrr for why this is a 50/50 blend of the
    single-hop and multi-hop averages, not a flat macro-average."""
    docs = _load_docs()
    with open(HIDDEN_LABELS_PATH) as f:
        gold = json.load(f)

    _recall, mrr, _n_single, _n_bridge, _br = _blended_recall_and_mrr(predictions, gold, docs)
    assert mrr >= MIN_MRR, (
        f"MRR {mrr:.4f} below required threshold {MIN_MRR} -- correct "
        f"passages aren't being ranked highly enough within the top-{K}"
    )


def test_recall_at_k_meets_threshold(predictions):
    """The core grading metric: for each hidden eval query, what fraction of
    its gold passage(s) appear among the top-K retrieved chunks? Most
    queries have a single gold passage (fraction is 0 or 1); a subset of
    queries genuinely require synthesizing TWO passages from different
    categories to answer (fraction can be 0, 0.5, or 1) -- finding only one
    of the two earns partial credit, not a full miss. Graded at the passage
    level via substring containment, independent of whatever chunking
    scheme (or none at all) the submission used, and as a 50/50 blend of
    the single-hop and multi-hop averages (see _blended_recall_and_mrr).
    Note: a fraction of the single-hop hidden gold labels are themselves
    intentionally noisy (see reference/generate_kb.py), so 100% is not
    achievable even in principle -- the threshold is calibrated below that
    ceiling, not at it."""
    docs = _load_docs()
    with open(HIDDEN_LABELS_PATH) as f:
        gold = json.load(f)

    recall_at_k, _mrr, n_single, n_bridge, _br = _blended_recall_and_mrr(predictions, gold, docs)
    assert recall_at_k >= MIN_RECALL_AT_K, (
        f"Recall@{K} {recall_at_k:.4f} below required threshold {MIN_RECALL_AT_K} "
        f"(blended 50/50 over {n_single} single-hop and {n_bridge} multi-hop "
        f"hidden eval queries)"
    )


def test_bridge_recall_meets_threshold(predictions):
    """A dedicated gate on the multi-hop (bridge) subset alone, in addition
    to the blended overall Recall@K above. Even the 50/50 blend left only a
    couple points of margin between a decomposition-aware submission and
    one that ignores multi-hop entirely on a real Docker run -- both cleared
    MIN_RECALL_AT_K comfortably, meaning an agent could still skip
    multi-hop handling and pass. The unblended gap on the bridge subset
    specifically was far larger and more reliable (0.600 decomposition-
    aware vs 0.450 whole-query-only ablation, on a bridge set deliberately
    written with asymmetric structure -- one richly-detailed dominant
    clause plus a brief tail question -- since queries where both topics
    get comparably detailed treatment turned out to be easy for a naive
    whole-query embedding too, giving no real signal either way), so this
    test requires clearing that gap directly, not just benefiting from it
    being blended in. MIN_BRIDGE_RECALL_AT_K is calibrated with margin
    below the decomposition-aware reference's real score and with margin
    above the ablation's, not at the ceiling of either -- 100% is not
    expected here (see reference/generate_kb.py CALIBRATION HISTORY for
    the specific bridge queries that remain a genuine, persistent miss
    even for the reference solution)."""
    docs = _load_docs()
    with open(HIDDEN_LABELS_PATH) as f:
        gold = json.load(f)

    _recall, _mrr, _n_single, n_bridge, bridge_recall = _blended_recall_and_mrr(predictions, gold, docs)
    assert bridge_recall >= MIN_BRIDGE_RECALL_AT_K, (
        f"multi-hop (bridge) Recall@{K} {bridge_recall:.4f} below required "
        f"threshold {MIN_BRIDGE_RECALL_AT_K} (averaged over {n_bridge} hidden "
        f"eval queries that require combining two passages from different "
        f"categories)"
    )


def test_precision_discrimination_meets_threshold(predictions):
    """A second, orthogonal difficulty gate alongside the bridge-recall one
    above -- added after real agent runs showed that the bridge gate mainly
    rewards ANY diversity-awareness (e.g. a 'spread results across
    categories' heuristic scored 0.6389, nearly matching a genuine
    decomposition-based reference's 0.694), not specifically multi-hop
    reasoning. This gate targets a different, independently-verified-hard
    source of difficulty: a fixed set of 9 single-hop passages (one per
    category, see PRECISION_SUBSET_PIDS), each with a genuinely close
    same-category sibling that real embeddings have repeatedly been
    observed confusing it with in earlier rounds of testing (e.g. an
    acct_lockout query getting acct_sso_login instead). Unlike bridge
    recall, no diversity trick helps here -- there is exactly ONE correct
    passage to find, and category-level proximity alone isn't enough to
    get the SPECIFIC right one. Graded at STRICT top-1 (the correct
    passage must be the very first chunk returned), not top-3 recall,
    since a 3-slot window gives too much room for a coarse match to
    accidentally include the right passage; discrimination difficulty is
    fundamentally about ranking precision. These 9 queries are also
    excluded from the unrelated noisy-gold-label mechanism (see
    reference/generate_kb.py) so a genuinely correct solution is never
    penalized against a deliberately-wrong recorded label here."""
    docs = _load_docs()
    with open(HIDDEN_LABELS_PATH) as f:
        gold = json.load(f)

    precision_qids = [qid for qid, gp in gold.items() if len(gp) == 1 and gp[0] in PRECISION_SUBSET_PIDS]
    assert precision_qids, "no hidden eval queries matched PRECISION_SUBSET_PIDS -- check the pid list is in sync with reference/generate_kb.py"

    n_correct = 0
    for qid in precision_qids:
        chunks = predictions.get(qid, [])
        if not chunks:
            continue
        top1_pids = _source_passages_for_chunk(chunks[0], docs)
        if gold[qid][0] in top1_pids:
            n_correct += 1
    precision_acc = n_correct / len(precision_qids)

    assert precision_acc >= MIN_PRECISION_ACC, (
        f"strict top-1 precision-discrimination accuracy {precision_acc:.4f} "
        f"below required threshold {MIN_PRECISION_ACC} ({n_correct}/"
        f"{len(precision_qids)} hidden eval queries that require "
        f"distinguishing a passage from a genuinely close same-category "
        f"sibling, not just landing in the right general topic area)"
    )
