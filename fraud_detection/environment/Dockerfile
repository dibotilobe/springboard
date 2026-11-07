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

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

COPY data/ /app/data/
COPY starter_pipeline.py /app/starter_pipeline.py

CMD ["/bin/bash"]
