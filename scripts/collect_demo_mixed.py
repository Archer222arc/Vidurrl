#!/usr/bin/env python3
"""
混合策略示教数据收集脚本 - 优化版

收集多种启发式策略的状态-动作对，提高行为克隆的多样性和鲁棒性。
支持Round Robin、LOR、Random的混合数据收集。
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np


def _bootstrap_repo_path() -> None:
    """Ensure imports resolve to the current repo checkout."""
    repo_root = Path(__file__).resolve().parent.parent
    conflict_root = repo_root.parent / "Vidur"

    def _same_path(a: str, b: Path) -> bool:
        try:
            return Path(a).resolve() == b.resolve()
        except (OSError, RuntimeError):
            return False

    sys.path[:] = [p for p in sys.path if not _same_path(p, conflict_root)]

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_path()

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.entities import Replica


class MixedDemoCollector:
    """混合策略示教数据收集器"""

    def __init__(self, policies: List[str] = None):
        """
        初始化混合收集器

        Args:
            policies: 策略列表，默认为 ["round_robin", "lor", "random"]
        """
        self.policies = policies or ["round_robin", "lor", "random"]
        self.all_demo_data = []

    def collect_mixed_demonstrations(
        self,
        steps_per_policy: int = 1000,
        num_replicas: int = 4,
        qps: float = 3.0,
        output_path: str = "./mixed_demo_data.pkl"
    ) -> None:
        """
        收集混合策略示教数据

        Args:
            steps_per_policy: 每个策略收集的步数
            num_replicas: 副本数量
            qps: 请求生成速率
            output_path: 输出文件路径
        """
        print(f"🎯 开始收集混合策略示教数据")
        print(f"📊 配置: policies={self.policies}, steps_per_policy={steps_per_policy}")
        print(f"   replicas={num_replicas}, qps={qps}")

        for policy in self.policies:
            print(f"\n🔄 收集策略: {policy}")
            policy_data = self._collect_single_policy(
                policy=policy,
                num_steps=steps_per_policy,
                num_replicas=num_replicas,
                qps=qps
            )
            self.all_demo_data.extend(policy_data)
            print(f"✅ {policy} 收集完成: {len(policy_data)} 个样本")

        # 打乱混合数据
        np.random.shuffle(self.all_demo_data)

        print(f"\n✅ 混合数据收集完成: 总计 {len(self.all_demo_data)} 个状态-动作对")

        # 保存数据
        self._save_mixed_demo_data(output_path)

    def _collect_single_policy(
        self,
        policy: str,
        num_steps: int,
        num_replicas: int,
        qps: float
    ) -> List[Dict[str, Any]]:
        """收集单个策略的示教数据"""

        # 构建配置
        import sys
        original_argv = sys.argv.copy()

        config_args = [
            "collect_demo_mixed.py",
            "--global_scheduler_config_type", policy,
            "--cluster_config_num_replicas", str(num_replicas),
            "--synthetic_request_generator_config_num_requests", str(num_steps),
            "--interval_generator_config_type", "poisson",
            "--poisson_request_interval_generator_config_qps", str(qps),
            "--metrics_config_subsamples", "200000",
        ]

        # 临时替换sys.argv并创建配置
        sys.argv = config_args
        try:
            config = SimulationConfig.create_from_cli_args()
        finally:
            sys.argv = original_argv

        # 运行模拟并收集数据
        simulator = Simulator(config)
        policy_demo_data = []

        # Hook到调度器收集状态-动作对
        original_schedule = simulator._scheduler.schedule

        def schedule_with_collection():
            # 获取状态
            if hasattr(simulator._scheduler, 'get_current_state'):
                state = simulator._scheduler.get_current_state()
            else:
                # 对于非PPO调度器，手动构建状态
                state = self._build_state_for_heuristic(simulator._scheduler)

            # 执行原始调度
            result = original_schedule()

            # 记录动作
            if result:
                actions = [replica_id for replica_id, _ in result]
                # 记录状态-动作对
                for action in actions:
                    policy_demo_data.append({
                        'state': state.copy() if isinstance(state, np.ndarray) else state,
                        'action': action,
                        'policy': policy
                    })

            return result

        # 替换调度方法
        simulator._scheduler.schedule = schedule_with_collection

        try:
            # 运行模拟
            simulator.run()
        except Exception as e:
            print(f"❌ {policy} 收集失败: {e}")
            return []

        return policy_demo_data

    def _build_state_for_heuristic(self, scheduler) -> np.ndarray:
        """为启发式调度器构建状态向量"""
        try:
            # 基础状态特征
            state_features = []

            # 队列长度
            queue_len = len(getattr(scheduler, '_request_queue', []))
            state_features.append(queue_len)

            # 副本状态
            replica_ids = getattr(scheduler, '_replica_ids', list(range(4)))
            for replica_id in replica_ids:
                replica_scheduler = scheduler.get_replica_scheduler(replica_id)

                # 分配的blocks数量
                num_alloc = getattr(replica_scheduler, '_num_allocated_blocks', 0)
                num_total = getattr(replica_scheduler._config, 'num_blocks', 100)
                utilization = num_alloc / max(num_total, 1)

                state_features.extend([
                    utilization,
                    num_alloc,
                    len(getattr(replica_scheduler, '_running_requests', []))
                ])

            # 时间特征
            current_time = getattr(scheduler, '_current_time', 0.0)
            state_features.append(current_time % 100)

            return np.array(state_features, dtype=np.float32)

        except Exception as e:
            print(f"⚠️ 状态构建失败，使用默认状态: {e}")
            return np.zeros(20, dtype=np.float32)

    def _save_mixed_demo_data(self, output_path: str) -> None:
        """保存混合示教数据"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 统计数据
        policy_counts = {}
        action_counts = {}

        for item in self.all_demo_data:
            policy = item['policy']
            action = item['action']

            policy_counts[policy] = policy_counts.get(policy, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1

        mixed_stats = {
            'total_samples': len(self.all_demo_data),
            'policies': self.policies,
            'policy_distribution': policy_counts,
            'action_distribution': action_counts,
            'state_dim': len(self.all_demo_data[0]['state']) if self.all_demo_data else 0
        }

        # 保存数据和统计
        with open(output_path, 'wb') as f:
            pickle.dump({
                'demo_data': self.all_demo_data,
                'stats': mixed_stats
            }, f)

        print(f"💾 混合数据已保存: {output_path}")
        print(f"📊 策略分布: {policy_counts}")
        print(f"📊 动作分布: {action_counts}")
        print(f"🎯 状态维度: {mixed_stats['state_dim']}")


def main():
    parser = argparse.ArgumentParser(description="收集混合策略示教数据用于PPO热身")
    parser.add_argument("--policies", nargs="+",
                       choices=["round_robin", "lor", "random"],
                       default=["round_robin", "lor", "random"],
                       help="要收集的策略列表")
    parser.add_argument("--steps_per_policy", type=int, default=1000,
                       help="每个策略收集的步数")
    parser.add_argument("--replicas", type=int, default=4, help="副本数量")
    parser.add_argument("--qps", type=float, default=3.0, help="请求生成速率")
    parser.add_argument("--output", type=str, default="./outputs/mixed_demo_data.pkl",
                       help="输出文件路径")

    args = parser.parse_args()

    # 收集混合示教数据
    collector = MixedDemoCollector(policies=args.policies)
    collector.collect_mixed_demonstrations(
        steps_per_policy=args.steps_per_policy,
        num_replicas=args.replicas,
        qps=args.qps,
        output_path=args.output
    )


if __name__ == "__main__":
    main()