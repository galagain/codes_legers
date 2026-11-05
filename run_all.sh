#!/usr/bin/env bash
set -euo pipefail
bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_RUNNER:-python3}" "$bundle_root/tools/run_all.py" "$@"
