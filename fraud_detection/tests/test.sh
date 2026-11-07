#!/bin/bash
# OFFLINE (allow_internet=false): all deps pre-installed in Dockerfile.
set -euo pipefail

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    exit 1
fi

mkdir -p /logs/verifier

# Disable exit-on-error so we can capture pytest's exit code below and
# always write a reward file, even when tests fail.
set +e
python3 -m pytest \
    -o cache_dir=/tmp/pytest_cache \
    /tests/test_outputs.py -rA
RC=$?

if [ "$RC" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
