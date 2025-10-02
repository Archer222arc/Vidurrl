#!/bin/bash
# Run toy model simulation

set -e  # Exit on error

CONFIG="toymodel/configs/config.json"

python -m toymodel.scripts.run_simulation "$CONFIG"
