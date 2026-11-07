#!/bin/bash
# OFFLINE (allow_internet=false): all deps pre-installed in Dockerfile.
# NOTE: -e is intentionally omitted so the script always reaches the reward
# write below, even when pytest exits non-zero (its exit code is captured
# explicitly in RC and used to decide the reward, rather than aborting here).
set -uo pipefail

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    exit 1
fi

mkdir -p /logs/verifier

python3 -m pytest \
    -o cache_dir=/tmp/pytest_cache \
    /tests/test_outputs.py -rA
RC=$?

if [ "$RC" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
