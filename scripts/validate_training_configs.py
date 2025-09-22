#!/usr/bin/env python3
"""
Validate that training scripts and config files implement the reward improvements.

This script checks that all training configurations have been updated to use:
1. Linear reward scaling (not tanh)
2. Enhanced load balance penalty (weight >= 1.0)
3. Improved exploration parameters (entropy_coef = 0.02)
4. Optimized temperature control (base_temp = 1.5)
"""

import json
import os
import sys
from pathlib import Path

def load_json_config(config_path):
    """Load and parse JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {config_path}: {e}")
        return None

def validate_ppo_config(config, config_name):
    """Validate PPO-specific configuration parameters."""
    print(f"\n🔍 Validating {config_name}...")

    issues = []

    # Check PPO hyperparameters
    ppo_config = config.get('ppo_config', {})

    # Entropy coefficient check
    entropy_coef = ppo_config.get('entropy_coef', 0)
    if entropy_coef != 0.02:
        issues.append(f"entropy_coef should be 0.02, found: {entropy_coef}")
    else:
        print(f"  ✅ entropy_coef: {entropy_coef}")

    # GAE lambda check
    gae_lambda = ppo_config.get('gae_lambda', 0)
    if gae_lambda > 0.90:
        issues.append(f"gae_lambda should be <= 0.90, found: {gae_lambda}")
    else:
        print(f"  ✅ gae_lambda: {gae_lambda}")

    # Check reward configuration
    reward_config = config.get('reward_config', {})

    # Load balance penalty check
    load_balance_penalty = reward_config.get('load_balance_penalty', 0)
    if load_balance_penalty < 1.0:
        issues.append(f"load_balance_penalty should be >= 1.0, found: {load_balance_penalty}")
    else:
        print(f"  ✅ load_balance_penalty: {load_balance_penalty}")

    # Reward scaling check
    reward_scaling = reward_config.get('reward_scaling', {})
    if reward_scaling.get('type') != 'linear':
        issues.append(f"reward_scaling.type should be 'linear', found: {reward_scaling.get('type', 'missing')}")
    else:
        print(f"  ✅ reward_scaling.type: {reward_scaling.get('type')}")

    # Temperature control check
    temp_config = config.get('temperature_control', {})
    base_temp = temp_config.get('base_temperature', 0)
    if base_temp < 1.5:
        issues.append(f"base_temperature should be >= 1.5, found: {base_temp}")
    else:
        print(f"  ✅ base_temperature: {base_temp}")

    max_temp = temp_config.get('max_temperature', 0)
    if max_temp < 3.0:
        issues.append(f"max_temperature should be >= 3.0, found: {max_temp}")
    else:
        print(f"  ✅ max_temperature: {max_temp}")

    return issues

def validate_training_scripts():
    """Check that training scripts are properly configured."""
    print("\n🔍 Validating training scripts...")

    script_paths = [
        "scripts/train_ppo_warmstart_optimized.sh",
        "scripts/train_ppo_with_external_pretrain.sh"
    ]

    issues = []

    for script_path in script_paths:
        if os.path.exists(script_path):
            print(f"  ✅ Found: {script_path}")

            # Check if script uses config files (this is expected)
            with open(script_path, 'r') as f:
                content = f.read()
                if 'ppo_warmstart.json' in content or 'train_ppo_warmstart_optimized.sh' in content:
                    print(f"    ✅ Uses improved configuration (directly or via delegation)")
                else:
                    issues.append(f"{script_path} doesn't use improved configurations")
        else:
            issues.append(f"Missing training script: {script_path}")

    return issues

def main():
    """Run all validation checks."""
    print("PPO Training Configuration Validation")
    print("=" * 50)

    # Ensure we're in the right directory
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    all_issues = []

    # Check config files
    config_files = [
        ("configs/ppo_warmstart.json", validate_ppo_config),
        ("configs/standalone_pretrain.json", None)  # Only PPO configs need validation
    ]

    for config_path, validator in config_files:
        if os.path.exists(config_path):
            config = load_json_config(config_path)
            if config and validator:
                issues = validator(config, config_path)
                all_issues.extend(issues)
            elif not validator:
                print(f"\n📄 Found {config_path} (pretraining config - no validation needed)")
        else:
            all_issues.append(f"Missing config file: {config_path}")

    # Check training scripts
    script_issues = validate_training_scripts()
    all_issues.extend(script_issues)

    # Generate summary
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION VALIDATION SUMMARY")
    print("=" * 60)

    if not all_issues:
        print("🎉 ALL CHECKS PASSED!")
        print()
        print("✅ Training scripts and configs implement all reward improvements:")
        print("   - Linear reward scaling (prevents saturation)")
        print("   - Enhanced load balance penalty (weight = 1.0)")
        print("   - Optimized exploration (entropy_coef = 0.02)")
        print("   - Better temperature control (base = 1.5, max = 3.0)")
        print()
        print("🚀 Ready to run improved training with:")
        print("   ./scripts/train_ppo_warmstart_optimized.sh")
        print("   ./scripts/train_ppo_with_external_pretrain.sh <model.pt>")
        return 0
    else:
        print(f"⚠️  FOUND {len(all_issues)} ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        print()
        print("🔧 Please fix these issues before running training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())