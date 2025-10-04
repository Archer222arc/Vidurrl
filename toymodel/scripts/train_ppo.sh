#!/bin/bash
# Train PPO model for toy model queue scheduling

set -e  # Exit on error

CONFIG="toymodel/configs/ppo_config.json"

python -m toymodel.scripts.train_ppo --config "$CONFIG"

