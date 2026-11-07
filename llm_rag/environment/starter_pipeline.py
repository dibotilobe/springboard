"""First-pass retrieval pipeline. Reports Recall@3 on the labeled dev set
below -- feel free to build on this, replace parts of it, or start over.
Whatever you submit just needs to write /app/predictions.json in the format
described in the task instructions.

Uses suggested_passage_id (provided by an internal keyword-tagging tool,
see the field in dev_queries.json / eval_queries.json) as the top candidate,
and fills the remaining slots by TF-IDF similarity against the rest of the
corpus. Note this embeds the query as a single vector -- it doesn't attempt
to notice or handle questions that need more than one passage to answer.
"""
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = "/app/data/docs"
DEV_QUERIES_PATH = "/app/data/dev_queries.json"
K = 3


def load_docs(doc_dir):
    docs = {}
    for fname in sorted(os.listdir(doc_dir)):
        if fname.lower().endswith(".txt"):
            pid = fname[:-4]
            with open(os.path.join(doc_dir, fname), encoding="utf-8") as f:
                docs[pid] = f.read()
    return docs


def main():
    docs = load_docs(DOCS_DIR)
    pids = list(docs.keys())
    texts = [docs[pid] for pid in pids]

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(texts)

    with open(DEV_QUERIES_PATH) as f:
        dev_queries = json.load(f)

    recall_sum = 0.0
    for info in dev_queries.values():
        suggestion = info["suggested_passage_id"]
        query_vec = vectorizer.transform([info["query"]])
        sims = cosine_similarity(query_vec, doc_matrix)[0]
        ranked = sorted(
            ((s, pid) for s, pid in zip(sims, pids) if pid != suggestion),
            key=lambda x: -x[0],
        )
        top_k_pids = [suggestion] + [pid for _score, pid in ranked[: K - 1]]
        gold_pids = set(info["gold_passage_ids"])
        recall_sum += len(gold_pids & set(top_k_pids)) / len(gold_pids)

    recall_at_3 = recall_sum / len(dev_queries)
    print(f"dev set Recall@{K}: {recall_at_3:.4f} (averaged over {len(dev_queries)} queries, "
          f"partial credit for multi-passage ones)")


if __name__ == "__main__":
    main()
