#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "$0")/run_folds.sh" test CMR 1
