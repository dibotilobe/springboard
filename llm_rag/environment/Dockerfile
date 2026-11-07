FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

# tmux + asciinema are required by the terminus-2 agent harness to drive a
# terminal session inside the container -- without them, real agent runs
# fail immediately before the agent ever sees the task.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tmux \
        asciinema \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# faiss-cpu, sentence-transformers, torch, and transformers are all pinned
# explicitly (not left as sentence-transformers' unpinned transitive deps)
# for reasons learned the hard way on an earlier version of this task:
#   - faiss-cpu==1.7.4 (Apr 2023) predates Python 3.13 (Oct 2024) and has no
#     cp313 wheel on PyPI; 1.10.0+ does.
#   - sentence-transformers==2.2.2 imports huggingface_hub.cached_download,
#     removed in huggingface_hub>=0.19; since huggingface_hub is left
#     unpinned (pip resolves the latest at build time), 2.2.2 breaks
#     immediately on import. sentence-transformers>=2.3.0 dropped that
#     dependency.
#   - torch sets no numpy upper bound, so pip could otherwise resolve a
#     build compiled against NumPy 1.x, which raises a hard RuntimeError on
#     import against the NumPy 2.x pinned below. torch==2.5.1 has a
#     Python 3.13 wheel and mature NumPy 2.x support.
# requirements.lock bundles test-time dependencies (pytest) alongside the
# task's ML dependencies (numpy, torch, sentence-transformers, etc.) -- if
# this file is ever regenerated or trimmed, keep pytest pinned here too, or
# tests/test.sh's `python3 -m pytest` call will fail to find it.
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

COPY data/ /app/data/
COPY starter_pipeline.py /app/starter_pipeline.py

# Pre-download and cache an embedding model + tokenizer at build time, so
# it's available offline at runtime (allow_internet=false). This is provided
# for convenience only -- nothing in the task requires using this specific
# model or even a semantic-embedding approach at all; anything already in
# requirements.lock is fair game. Kept as a single Python -c line so no
# physical line in this Dockerfile begins with "from ..." (a multi-line
# heredoc here previously tripped the static checker's FROM-instruction
# parser into treating "from sentence_transformers import ..." as an
# unpinned base image).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# Force offline mode for every HuggingFace Hub / transformers / sentence-
# transformers call at RUNTIME (must come AFTER the prewarm step above, which
# needs real network access during the build). Without this, loading the
# model still sends a HEAD request to huggingface.co on every load to check
# for a newer revision, even though the files are already cached locally --
# in the offline evaluation sandbox (no internet, per task.toml's
# allow_internet=false) that request fails DNS resolution and retries with
# exponential backoff, burning through the agent/verifier timeout budget for
# no reason. This exact failure mode previously caused every submission of
# an earlier version of this task to time out silently -- offline mode skips
# the network check entirely and goes straight to the cache.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["/bin/bash"]
