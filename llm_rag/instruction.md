Our support agents currently search the internal knowledge base by
eyeballing article titles, which is slow and error-prone. Build a
retrieval system that, given a user's question, surfaces the
knowledge-base passage(s) that answer it. `/app/data/docs/` has 36 short
`.txt` passages, one per file, across closely related categories (account,
billing, sync, security, mobile, etc.) — real discrimination, not a coin
flip. `/app/data/dev_queries.json` is a labeled dev set (question, gold
`gold_passage_ids`, and a `suggested_passage_id`); most entries have one
gold passage, a few have two, since the single-label tagging tool only
ever caught one side of a two-part question. `/app/data/eval_queries.json`
is what you answer: question and `suggested_passage_id`, no gold labels —
brand-new questions phrased the way a real user would ask, sharing little
vocabulary with the answering passage(s), and some genuinely need two
passages from different categories combined with no flag telling you
which.

`/app/starter_pipeline.py` is an optional first-pass attempt with its dev
score, if useful — but build whatever actually generalizes to eval, not
just what looks good on dev. No internet at runtime; `requirements.lock`
covers everything installed, including an offline sentence-embedding
model, but no particular chunking, embedding, or index approach is
required. Write `/app/resolve.py`, run as `python3 /app/resolve.py` with
no arguments — read `/app/data/docs/` and `/app/data/eval_queries.json`,
write `/app/predictions.json` mapping each eval query id to up to 3
retrieved chunk texts, most to least relevant: `{"eval_000": ["<top-1
text>", "<top-2 text>"], ...}`. Every chunk must be actual text pulled
from `/app/data/docs/`, not generated.

Grading checks what fraction of the true answer passage(s) land in your
top 3 (matched back to source passage, so chunking scheme doesn't matter),
plus Mean Reciprocal Rank so burying the right answer counts for less.
Overall Recall@3 and MRR, averaged evenly across single- and two-passage
queries, must clear 0.45 and 0.30. Two-passage questions alone need
Recall@3 ≥ 0.65, since treating a query as one whole tends to find one of
the two passages but not the other. A fixed subset of single-passage
questions is graded on strict top-1 — the right passage must be first,
since each has a genuinely close same-category sibling — at ≥ 0.65. A
perfect score isn't expected: some relevance judgments are genuinely
ambiguous, and two-passage queries only get full credit when you find
both.
