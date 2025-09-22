#!/usr/bin/env python3
"""
Mixed demo collection module for PPO warm start training.

Extracts the mixed policy demonstration collection logic from bash scripts
to comply with CLAUDE.md modularization standards.
"""

import pickle
import numpy as np
import os
from pathlib import Path
from typing import List
import subprocess
import sys


class MixedDemoDataProcessor:
    """Processes and merges demonstration data from multiple policies."""

    def __init__(self, policies: List[str], temp_dir: str, output_path: str):
        self.policies = policies
        self.temp_dir = Path(temp_dir)
        self.output_path = output_path

    def merge_policy_data(self) -> None:
        """Merge demonstration data from multiple policies."""
        all_data = []
        stats = {
            'total_samples': 0,
            'policy_distribution': {},
            'action_distribution': {}
        }

        # Load data from each policy
        for policy in self.policies:
            policy_file = self.temp_dir / f'{policy}_demo.pkl'
            if policy_file.exists():
                with open(policy_file, 'rb') as f:
                    data = pickle.load(f)
                policy_samples = data['demo_data']
                all_data.extend(policy_samples)
                stats['policy_distribution'][policy] = len(policy_samples)
                print(f'📊 {policy}: {len(policy_samples)} 样本')

        # Load imbalanced scenario data
        imbalanced_files = list(self.temp_dir.glob('*_imbalanced_*.pkl'))
        imbalanced_count = 0
        for imbalanced_file in imbalanced_files:
            if imbalanced_file.exists():
                with open(imbalanced_file, 'rb') as f:
                    data = pickle.load(f)
                imbalanced_samples = data['demo_data']
                all_data.extend(imbalanced_samples)
                imbalanced_count += len(imbalanced_samples)
                print(f'🔥 {imbalanced_file.stem}: {len(imbalanced_samples)} 样本 (不均衡场景)')

        if imbalanced_count > 0:
            stats['policy_distribution']['imbalanced_scenarios'] = imbalanced_count

        # Shuffle mixed data
        np.random.shuffle(all_data)
        stats['total_samples'] = len(all_data)

        # Calculate action distribution
        for item in all_data:
            action = item['action']
            stats['action_distribution'][action] = stats['action_distribution'].get(action, 0) + 1

        stats['state_dim'] = len(all_data[0]['state']) if all_data else 0

        # Save merged data
        with open(self.output_path, 'wb') as f:
            pickle.dump({'demo_data': all_data, 'stats': stats}, f)

        print(f'💾 混合数据已保存: {self.output_path}')
        print(f'📊 总样本数: {stats["total_samples"]}')
        print(f'📊 策略分布: {stats["policy_distribution"]}')
        print(f'📊 动作分布: {stats["action_distribution"]}')


def collect_mixed_demo(output_path: str, policies: List[str], steps_per_policy: int,
                      num_replicas: int, qps: float, temp_dir: str,
                      include_imbalanced: bool = True,
                      simulator_output_base: str = "./data/pretraining/simulator_temp") -> int:
    """
    Collect mixed policy demonstration data.

    Args:
        output_path: Path to save merged demo data
        policies: List of policy names to collect from
        steps_per_policy: Number of steps per policy
        num_replicas: Number of replicas
        qps: Queries per second
        temp_dir: Temporary directory for individual policy data
        include_imbalanced: Whether to include imbalanced scenario data
        simulator_output_base: Base directory for simulator temporary outputs

    Returns:
        0 for success, non-zero for failure
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    print(f"🎯 开始收集混合策略示教数据: {' '.join(policies)}")
    if include_imbalanced:
        print("📊 包含极端不均衡场景数据收集")

    # Collect data for each policy
    for policy in policies:
        print(f"🔄 收集策略: {policy}")
        policy_output = temp_path / f"{policy}_demo.pkl"

        # Run collection script for this policy
        cmd = [
            sys.executable, "scripts/collect_demo.py",
            "--policy", policy,
            "--steps", str(steps_per_policy),
            "--replicas", str(num_replicas),
            "--qps", str(qps),
            "--output", str(policy_output)
        ]

        # 设置模拟器输出目录环境变量
        env = {"SIMULATOR_OUTPUT_BASE": simulator_output_base}
        result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})

        if result.returncode == 0:
            print(f"✅ {policy} 收集完成")
        else:
            print(f"❌ {policy} 收集失败")
            print(f"Error: {result.stderr}")
            return result.returncode

    # Collect additional imbalanced scenarios if requested
    if include_imbalanced:
        print("🔥 收集极端不均衡场景数据...")
        imbalanced_scenarios = [
            {"qps": qps * 2.0, "suffix": "high_load"},
            {"qps": qps * 0.5, "suffix": "low_load"},
        ]

        for scenario in imbalanced_scenarios:
            scenario_qps = scenario["qps"]
            scenario_suffix = scenario["suffix"]
            print(f"   🎯 场景: {scenario_suffix} (QPS={scenario_qps})")

            # Collect with different QPS for the primary policy
            primary_policy = policies[0]  # Use first policy for imbalanced scenarios
            scenario_output = temp_path / f"{primary_policy}_imbalanced_{scenario_suffix}.pkl"

            cmd = [
                sys.executable, "scripts/collect_demo.py",
                "--policy", primary_policy,
                "--steps", str(steps_per_policy // 2),  # Half steps for each scenario
                "--replicas", str(num_replicas),
                "--qps", str(scenario_qps),
                "--output", str(scenario_output)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
            if result.returncode == 0:
                print(f"   ✅ {scenario_suffix} 场景收集完成")
            else:
                print(f"   ⚠️ {scenario_suffix} 场景收集失败，继续其他收集")

    # Merge the collected data
    processor = MixedDemoDataProcessor(policies, temp_dir, output_path)
    processor.merge_policy_data()

    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_path)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect mixed policy demonstration data")
    parser.add_argument("--output", required=True, help="Output path for merged data")
    parser.add_argument("--policies", nargs="+", required=True, help="List of policies")
    parser.add_argument("--steps_per_policy", type=int, required=True, help="Steps per policy")
    parser.add_argument("--num_replicas", type=int, required=True, help="Number of replicas")
    parser.add_argument("--qps", type=float, required=True, help="Queries per second")
    parser.add_argument("--temp_dir", required=True, help="Temporary directory")
    parser.add_argument("--include_imbalanced", action="store_true",
                       help="Include imbalanced scenario data collection")
    parser.add_argument("--simulator_output_base", default="./data/pretraining/simulator_temp",
                       help="Base directory for simulator temporary outputs")

    args = parser.parse_args()

    exit_code = collect_mixed_demo(
        output_path=args.output,
        policies=args.policies,
        steps_per_policy=args.steps_per_policy,
        num_replicas=args.num_replicas,
        qps=args.qps,
        temp_dir=args.temp_dir,
        include_imbalanced=args.include_imbalanced,
        simulator_output_base=args.simulator_output_base
    )

    sys.exit(exit_code)