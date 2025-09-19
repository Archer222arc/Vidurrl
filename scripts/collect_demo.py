#!/usr/bin/env python3
"""
示教数据收集脚本 - 用于PPO热身启动

收集启发式策略(Round Robin/LOR/Random)的状态-动作对，
用于后续的行为克隆预训练。
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


class DemoCollector:
    """示教数据收集器"""

    def __init__(self, policy_type: str = "round_robin"):
        """
        初始化收集器

        Args:
            policy_type: 启发式策略类型 (round_robin, lor, random)
        """
        self.policy_type = policy_type
        self.demo_data = []

    def collect_demonstrations(
        self,
        num_steps: int = 4096,
        num_replicas: int = 4,
        qps: float = 2.0,
        output_path: str = "./demo_data.pkl"
    ) -> None:
        """
        收集示教数据

        Args:
            num_steps: 收集的步数
            num_replicas: 副本数量
            qps: 请求生成速率
            output_path: 输出文件路径
        """
        print(f"🎯 开始收集示教数据: {self.policy_type}")
        print(f"📊 配置: steps={num_steps}, replicas={num_replicas}, qps={qps}")

        # 构建配置 - 临时修改sys.argv
        import sys
        original_argv = sys.argv.copy()

        config_args = [
            "collect_demo.py",  # 程序名
            "--global_scheduler_config_type", self.policy_type,
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
            sys.argv = original_argv  # 恢复原始argv

        # 运行模拟并收集数据
        simulator = Simulator(config)

        # Hook到调度器收集状态-动作对
        original_schedule = simulator._scheduler.schedule

        def schedule_with_collection():
            # 获取状态（这里需要根据具体调度器实现调整）
            if hasattr(simulator._scheduler, 'get_current_state'):
                state = simulator._scheduler.get_current_state()
            else:
                # 对于非PPO调度器，我们需要手动构建状态
                state = self._build_state_for_heuristic(simulator._scheduler)

            # 执行原始调度
            result = original_schedule()

            # 记录动作（假设result是[(replica_id, request)]的列表）
            if result:
                actions = [replica_id for replica_id, _ in result]
                # 记录状态-动作对
                for action in actions:
                    self.demo_data.append({
                        'state': state.copy() if isinstance(state, np.ndarray) else state,
                        'action': action,
                        'policy': self.policy_type
                    })

            return result

        # 替换调度方法
        simulator._scheduler.schedule = schedule_with_collection

        try:
            # 运行模拟
            simulator.run()

            print(f"✅ 收集完成: {len(self.demo_data)} 个状态-动作对")

            # 保存数据
            self._save_demo_data(output_path)

        except Exception as e:
            print(f"❌ 收集失败: {e}")
            raise

    def _build_state_for_heuristic(self, scheduler) -> np.ndarray:
        """
        为启发式调度器构建状态向量

        Args:
            scheduler: 调度器实例

        Returns:
            状态向量
        """
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
            state_features.append(current_time % 100)  # 周期性时间特征

            return np.array(state_features, dtype=np.float32)

        except Exception as e:
            print(f"⚠️ 状态构建失败，使用默认状态: {e}")
            # 返回默认状态（4个副本的简单状态）
            return np.zeros(20, dtype=np.float32)

    def _save_demo_data(self, output_path: str) -> None:
        """保存示教数据"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 数据统计
        actions = [item['action'] for item in self.demo_data]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        demo_stats = {
            'total_samples': len(self.demo_data),
            'policy_type': self.policy_type,
            'action_distribution': action_counts,
            'state_dim': len(self.demo_data[0]['state']) if self.demo_data else 0
        }

        # 保存数据和统计
        with open(output_path, 'wb') as f:
            pickle.dump({
                'demo_data': self.demo_data,
                'stats': demo_stats
            }, f)

        print(f"💾 数据已保存: {output_path}")
        print(f"📊 动作分布: {action_counts}")
        print(f"🎯 状态维度: {demo_stats['state_dim']}")


def main():
    parser = argparse.ArgumentParser(description="收集示教数据用于PPO热身")
    parser.add_argument("--policy", choices=["round_robin", "lor", "random"],
                       default="round_robin", help="启发式策略类型")
    parser.add_argument("--steps", type=int, default=4096, help="收集的步数")
    parser.add_argument("--replicas", type=int, default=4, help="副本数量")
    parser.add_argument("--qps", type=float, default=2.0, help="请求生成速率")
    parser.add_argument("--output", type=str, default="./outputs/demo_data.pkl",
                       help="输出文件路径")

    args = parser.parse_args()

    # 收集示教数据
    collector = DemoCollector(policy_type=args.policy)
    collector.collect_demonstrations(
        num_steps=args.steps,
        num_replicas=args.replicas,
        qps=args.qps,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
