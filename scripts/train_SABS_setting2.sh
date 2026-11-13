#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "$0")/run_folds.sh" train SABS 2
