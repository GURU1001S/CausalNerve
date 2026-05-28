#!/bin/bash
# Wrapper for reproducibility checks
# Resolves the absolute directory of this bash script, then calls reproduce.py from there.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
python "$SCRIPT_DIR/reproduce.py" "$@"
