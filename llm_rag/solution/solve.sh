#!/bin/bash
set -euo pipefail

cat > /app/resolve.py << 'PYEOF'
"""Reference solution: semantic-embedding retrieval with signal-validated
query decomposition and a conservative diversity fallback.

Deliberately does NOT use suggested_passage_id -- that field is only fully
reliable on dev_queries.json (built from a historical ticket log where the
resolution was already confirmed, and even then it's single-label so it can
cover at most one hop of a two-passage answer); on eval_queries.json it's
assigned to a brand-new, unresolved query and is correct only ~8% of the
time (see reference/generate_kb.py for why). Leaning on it the way
starter_pipeline.py does looks fine on dev and fails on eval, exactly like
last_interaction_type in the churn-prediction task.

Some queries genuinely require combining TWO passages from different
categories to answer (a bridging question, e.g. "if I downgrade my plan,
does my scheduled export still run?"). Embedding the whole query as one
vector and ranking by cosine similarity tends to surface only the more
semantically dominant of the two sub-topics -- the same reason multi-hop QA
is harder for single-pass dense retrieval than single-hop lookup.

Bridge queries are NOT written with a single, consistent textual seam --
some hinge on a comma, some on "and", and some are a single embedded
question with no delimiter at all (see reference/generate_kb.py). An
earlier version of this solution just split blindly on the first comma;
that worked because every bridge query in an earlier round of this task
happened to be comma-hinged by construction, which made blind comma-
splitting a reliable trick rather than a real signal. This version instead:

1. Proposes candidate 2-way splits from textual cues (first comma, then a
   trailing " and " clause) -- but does NOT trust either blindly.
2. VALIDATES a candidate split by checking whether its two clauses'
   top-ranked passages actually differ. If they don't (e.g. the comma just
   separates a real question from a generic follow-up like "any fix" or
   "help", which is common on ordinary single-hop questions), the split is
   rejected and treated as noise, not a genuine two-part question.
3. If a candidate split validates, retrieves separately per clause and
   merges the rankings (clauses first, whole query last as a fallback
   fill) -- same principle as before, now gated on actual retrieval
   evidence instead of blind punctuation-matching.
4. If NO candidate split validates (this is the only path available for a
   bridge query with no textual seam at all, e.g. "would paying for a
   bigger plan also raise my API rate limit?"), falls back to standard
   Maximal Marginal Relevance: rank 1 stays pure cosine similarity (the
   single most reliable signal), but ranks 2 and 3 are chosen by trading
   off raw similarity against dissimilarity to what's already selected,
   giving a genuinely distinct secondary passage a real chance to displace
   a near-duplicate of rank 1. (An earlier, more conservative version only
   opened up the 3rd slot to this, keeping rank 2 fixed too -- real Docker
   testing showed this MMR fallback was already working fine on its own
   for all 3 delimiter-free queries either way; the actual misses turned
   out to be caused by decompose_candidates' "and" logic, see 5 below.)
5. decompose_candidates() picks the FIRST " and " occurrence in a query,
   not the last. A real Docker run with per-query diagnostics showed the
   last-occurrence version reliably missed a query with two "and"s (e.g.
   "I made changes on my phone with no signal and made different changes
   on my laptop at the same time and I'm not sure what happens once my
   phone reconnects") -- splitting on the LAST "and" left clause_a still
   describing both devices and clause_b as a generic, low-signal follow-up
   ("I'm not sure what happens..."), missing both gold passages entirely.
   The first "and" is far more likely to be the real top-level
   conjunction in an ordinary compound question.

Nothing about this approach is prescribed by the task -- a different
embedding model, different split heuristics, or a different multi-hop
detection strategy entirely (e.g. LLM-based sub-query generation, if such a
model were available offline) would all be acceptable, as long as it clears
the Recall@3 / MRR bars on the hidden eval set. Note that 100% is not
achievable even in principle: a fraction of the hidden single-hop gold
labels are themselves noisy, and multi-hop queries only award full credit
when BOTH passages are found (see reference/generate_kb.py), so the
thresholds are calibrated below that ceiling, not at it.
"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = "/app/data/docs"
EVAL_QUERIES_PATH = "/app/data/eval_queries.json"
PREDICTIONS_PATH = "/app/predictions.json"
MODEL_NAME = "paraphrase-MiniLM-L3-v2"
K = 3
MMR_LAMBDA = 0.93


def load_docs(doc_dir):
    docs = {}
    for fname in sorted(os.listdir(doc_dir)):
        if fname.lower().endswith(".txt"):
            pid = fname[:-4]
            with open(os.path.join(doc_dir, fname), encoding="utf-8") as f:
                docs[pid] = f.read()
    return docs


def normalize_rows(mat):
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def rank_pids(query_vec, doc_vecs_norm, pids):
    """Returns (ranked_pids, sims_by_pid) for one query vector against all
    docs, most similar first."""
    qn = query_vec / np.linalg.norm(query_vec)
    sims = doc_vecs_norm @ qn
    order = np.argsort(-sims)
    ranked = [pids[i] for i in order]
    sims_by_pid = {pids[i]: float(sims[i]) for i in range(len(pids))}
    return ranked, sims_by_pid


def decompose_candidates(query):
    """Candidate 2-way splits to TRY, in priority order (first comma, then
    the FIRST ' and ' clause) -- neither is trusted yet here. main()
    validates each candidate against retrieval evidence before using it;
    an untrusted split is exactly as likely to be noise (a generic
    follow-up clause on an ordinary single-hop question) as it is to be a
    genuine second sub-question.

    Uses the FIRST ' and ' occurrence, not the last: for an ordinary
    two-part compound question, the first "and" is the one most likely to
    be the actual top-level conjunction, with any subsequent "and"s more
    likely nested inside one of the two real clauses. Splitting on the
    LAST occurrence instead (an earlier version of this function) failed
    on queries with two "and"s, e.g. "I made changes on my phone with no
    signal and made different changes on my laptop at the same time and
    I'm not sure what happens once my phone reconnects" -- splitting on
    the last "and" left clause_a still compound (mentioning both devices)
    and clause_b a generic, low-signal follow-up ("I'm not sure what
    happens..."), which reliably missed both gold passages. Splitting on
    the first "and" instead keeps each clause anchored to one real device
    / topic."""
    candidates = []
    if "," in query:
        idx = query.index(",")
        a, b = query[:idx].strip(), query[idx + 1:].strip()
        if a and b:
            candidates.append((a, b))
    lowered = query.lower()
    and_idx = lowered.find(" and ")
    if and_idx != -1:
        a, b = query[:and_idx].strip(), query[and_idx + 5:].strip()
        if a and b:
            candidates.append((a, b))
    return candidates


def merge_rankings(ranked_lists, k):
    """Round-robins across each view's ranking, in the order given (clauses
    before the whole query), so every view gets a chance to contribute a
    candidate before any view repeats."""
    result = []
    seen = set()
    pointers = [0] * len(ranked_lists)
    while len(result) < k:
        progressed = False
        for i, ranked in enumerate(ranked_lists):
            if len(result) >= k:
                break
            while pointers[i] < len(ranked) and ranked[pointers[i]] in seen:
                pointers[i] += 1
            if pointers[i] < len(ranked):
                pid = ranked[pointers[i]]
                result.append(pid)
                seen.add(pid)
                pointers[i] += 1
                progressed = True
        if not progressed:
            break
    return result


def mmr_diversify(whole_ranked, whole_sims, doc_vecs_norm, pids, k, lam=MMR_LAMBDA):
    """Rank 1 is kept exactly as pure cosine similarity gives it -- the
    single most reliable signal available, and single-hop recall depends
    on it most. Ranks 2..k are chosen by standard Maximal Marginal
    Relevance: at each step, pick whichever remaining candidate maximizes
    lambda*sim(query, d) - (1-lambda)*max_sim(d, already_selected), so a
    genuinely distinct secondary passage gets a real chance to displace a
    near-duplicate of what's already picked -- the only lever available
    for a bridge query with no textual seam to split on at all."""
    if k < 1 or not whole_ranked:
        return whole_ranked[:k]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    selected = [whole_ranked[0]]
    remaining = whole_ranked[1:]
    while len(selected) < k and remaining:
        best_pid, best_score = None, float("-inf")
        for cand in remaining:
            cand_vec = doc_vecs_norm[pid_to_idx[cand]]
            div_penalty = max(
                float(cand_vec @ doc_vecs_norm[pid_to_idx[s]]) for s in selected
            )
            score = lam * whole_sims[cand] - (1 - lam) * div_penalty
            if score > best_score:
                best_score, best_pid = score, cand
        selected.append(best_pid)
        remaining.remove(best_pid)
    return selected


def main():
    docs = load_docs(DOCS_DIR)
    pids = list(docs.keys())
    texts = [docs[pid] for pid in pids]

    with open(EVAL_QUERIES_PATH) as f:
        eval_queries = json.load(f)
    query_ids = list(eval_queries.keys())

    model = SentenceTransformer(MODEL_NAME)
    doc_vecs_norm = normalize_rows(model.encode(texts, convert_to_numpy=True))

    # Batch-embed every view (whole query + every candidate clause for
    # every query) in one encode() call for efficiency.
    query_candidates = {qid: decompose_candidates(eval_queries[qid]["query"]) for qid in query_ids}
    all_views = []
    for qid in query_ids:
        all_views.append(eval_queries[qid]["query"])
        for a, b in query_candidates[qid]:
            all_views.append(a)
            all_views.append(b)
    view_vecs = normalize_rows(model.encode(all_views, convert_to_numpy=True))

    view_vec_by_key = {}
    i = 0
    for qid in query_ids:
        view_vec_by_key[(qid, "__whole__")] = view_vecs[i]
        i += 1
        for a, b in query_candidates[qid]:
            view_vec_by_key[(qid, a)] = view_vecs[i]
            i += 1
            view_vec_by_key[(qid, b)] = view_vecs[i]
            i += 1

    predictions = {}
    for qid in query_ids:
        whole_ranked, whole_sims = rank_pids(view_vec_by_key[(qid, "__whole__")], doc_vecs_norm, pids)

        chosen_split = None
        for a, b in query_candidates[qid]:
            ranked_a, _ = rank_pids(view_vec_by_key[(qid, a)], doc_vecs_norm, pids)
            ranked_b, _ = rank_pids(view_vec_by_key[(qid, b)], doc_vecs_norm, pids)
            if ranked_a[0] != ranked_b[0]:
                chosen_split = (ranked_a, ranked_b)
                break

        if chosen_split is not None:
            merged = merge_rankings([chosen_split[0], chosen_split[1], whole_ranked], K)
        else:
            merged = mmr_diversify(whole_ranked, whole_sims, doc_vecs_norm, pids, K)

        predictions[qid] = [docs[pid] for pid in merged]

    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
PYEOF

python3 /app/resolve.py
