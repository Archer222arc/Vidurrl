#!/bin/bash
# Compare all scheduler types

set -e  # Exit on error

CONFIG="toymodel/configs/config.json"

python -m toymodel.scripts.compare_schedulers "$CONFIG"
