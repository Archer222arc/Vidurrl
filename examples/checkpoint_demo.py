#!/usr/bin/env python3
"""
PPO Checkpoint Management Demo

Demonstrates checkpoint saving, loading, and inference-only mode
for the modular PPO scheduler.
"""

import subprocess
import time
from pathlib import Path


def run_training_with_checkpoints():
    """Run PPO training with checkpoint saving enabled."""

    print("🚀 PPO Checkpoint Management Demo")
    print("=" * 60)

    # Create checkpoint directory
    checkpoint_dir = Path("./outputs/checkpoints_demo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Checkpoint directory: {checkpoint_dir.absolute()}")
    print("🎯 Training configuration: 2 replicas, 50 requests, 1QPS")
    print("💾 Checkpoint interval: every 10 steps")
    print("-" * 60)

    # Training command with checkpoints
    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--p_p_o_global_scheduler_modular_config_enable_checkpoints",
        "--p_p_o_global_scheduler_modular_config_checkpoint_dir", str(checkpoint_dir),
        "--p_p_o_global_scheduler_modular_config_checkpoint_interval", "10",
        "--p_p_o_global_scheduler_modular_config_max_checkpoints", "3",
        "--p_p_o_global_scheduler_modular_config_rollout_len", "8",
        "--p_p_o_global_scheduler_modular_config_lr", "0.001",
        "--cluster_config_num_replicas", "2",
        "--synthetic_request_generator_config_num_requests", "50",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "1",
        "--metrics_config_subsamples", "5000"
    ]

    print("💡 Features being tested:")
    print("  - Automatic checkpoint saving every 10 steps")
    print("  - Maximum 3 checkpoints kept")
    print("  - Model state, normalizer, and training metadata saved")
    print("-" * 60)

    try:
        print("🎬 Starting training with checkpoint saving...")
        result = subprocess.run(cmd, timeout=60)  # 1 minute timeout

        if result.returncode == 0:
            print("✅ Training completed successfully!")
        else:
            print("⚠️  Training ended early")

    except subprocess.TimeoutExpired:
        print("⏰ Training demo completed (1 minute)")
    except KeyboardInterrupt:
        print("🛑 User interrupted training")
    except Exception as e:
        print(f"❌ Training error: {e}")

    # Check saved checkpoints
    print("\n" + "=" * 60)
    print("📋 Checkpoint Analysis")
    print("-" * 60)

    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if checkpoints:
        print(f"✅ Found {len(checkpoints)} checkpoint(s):")
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  - {ckpt.name} ({size_mb:.2f} MB)")

        # Check for latest symlink
        latest_link = checkpoint_dir / "latest.pt"
        if latest_link.exists():
            print(f"  - latest.pt -> {latest_link.readlink().name}")

        print(f"\n🔍 To inspect checkpoint contents:")
        print(f"python -c \"import torch; print(torch.load('{checkpoints[-1]}', weights_only=False).keys())\"")

        print(f"\n🎯 To run inference-only mode:")
        print(f"python -m vidur.main --global_scheduler_config_type ppo_modular \\")
        print(f"  --p_p_o_global_scheduler_modular_config_load_checkpoint '{checkpoints[-1]}' \\")
        print(f"  --p_p_o_global_scheduler_modular_config_inference_only \\")
        print(f"  --cluster_config_num_replicas 2 --synthetic_request_generator_config_num_requests 20")

    else:
        print("❌ No checkpoints found - training may have been too short")

    print("=" * 60)


def test_inference_mode():
    """Test inference-only mode with a saved checkpoint."""

    checkpoint_dir = Path("./outputs/checkpoints_demo")
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

    if not checkpoints:
        print("⚠️  No checkpoints found for inference test")
        return

    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))

    print(f"\n🎯 Testing Inference Mode")
    print("-" * 60)
    print(f"📂 Loading checkpoint: {latest_checkpoint.name}")

    cmd = [
        "python", "-m", "vidur.main",
        "--global_scheduler_config_type", "ppo_modular",
        "--p_p_o_global_scheduler_modular_config_load_checkpoint", str(latest_checkpoint),
        "--p_p_o_global_scheduler_modular_config_inference_only",
        "--p_p_o_global_scheduler_modular_config_enable_tensorboard", "false",
        "--cluster_config_num_replicas", "2",
        "--synthetic_request_generator_config_num_requests", "20",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "2",
        "--metrics_config_subsamples", "1000"
    ]

    try:
        print("🎬 Running inference-only mode...")
        result = subprocess.run(cmd, timeout=30)

        if result.returncode == 0:
            print("✅ Inference mode completed successfully!")
        else:
            print("⚠️  Inference mode ended with issues")

    except subprocess.TimeoutExpired:
        print("⏰ Inference demo completed")
    except Exception as e:
        print(f"❌ Inference error: {e}")


if __name__ == "__main__":
    run_training_with_checkpoints()
    test_inference_mode()