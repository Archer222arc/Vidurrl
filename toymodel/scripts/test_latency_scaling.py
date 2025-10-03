#!/usr/bin/env python3
"""
Test latency scaling with different simulation times.
"""

import numpy as np
from toymodel.src.environment import QueueEnvironment
from toymodel.src.config import load_config
from toymodel.schedulers import PPOScheduler, RandomScheduler

def test_latency_scaling():
    """Test how latency scales with simulation time."""
    print("=" * 60)
    print("测试不同时间长度下的延迟表现")
    print("=" * 60)
    
    # Load config
    config = load_config("toymodel/configs/config.json")
    
    # Test different simulation times
    test_times = [50, 100, 200, 300, 400, 500]
    
    # Load PPO model
    ppo_model_path = "toymodel/outputs/models/ppo_model_latest.pt"
    ppo_scheduler = PPOScheduler(
        num_replicas=2,
        model_path=ppo_model_path,
        n_requests=3
    )
    ppo_scheduler.set_eval_mode()
    
    # Random scheduler for comparison
    random_scheduler = RandomScheduler(num_replicas=2, seed=42)
    
    print(f"{'Time':<8} {'PPO Lat':<10} {'PPO Acc':<8} {'Random Lat':<12} {'Random Acc':<10} {'PPO Req':<8} {'Rand Req':<8}")
    print("-" * 80)
    
    for max_time in test_times:
        # Test PPO
        env_ppo = QueueEnvironment(
            num_replicas=config.environment.num_replicas,
            max_time=max_time,
            service_rates=config.environment.service_rates,
            arrival_rates=config.environment.arrival_rates,
            seed=42,
            tensorboard_enabled=False,
        )
        
        def ppo_policy(request, replicas):
            return ppo_scheduler.schedule(request, replicas)
        
        ppo_requests = env_ppo.run_simulation(ppo_policy)
        ppo_latencies = [req.total_time for req in ppo_requests]
        ppo_accuracy = sum(1 for req in ppo_requests if req.assigned_replica == req.request_type) / len(ppo_requests) * 100
        
        # Test Random
        env_random = QueueEnvironment(
            num_replicas=config.environment.num_replicas,
            max_time=max_time,
            service_rates=config.environment.service_rates,
            arrival_rates=config.environment.arrival_rates,
            seed=42,
            tensorboard_enabled=False,
        )
        
        def random_policy(request, replicas):
            return random_scheduler.schedule(request, replicas)
        
        random_requests = env_random.run_simulation(random_policy)
        random_latencies = [req.total_time for req in random_requests]
        random_accuracy = sum(1 for req in random_requests if req.assigned_replica == req.request_type) / len(random_requests) * 100
        
        print(f"{max_time:<8} {np.mean(ppo_latencies):<10.4f} {ppo_accuracy:<8.1f}% {np.mean(random_latencies):<12.4f} {random_accuracy:<10.1f}% {len(ppo_requests):<8} {len(random_requests):<8}")
    
    print("\n分析:")
    print("- 随着时间增加，请求数量增加，队列积压更严重")
    print("- PPO在短时间下表现好，但长时间下性能下降")
    print("- 这说明PPO模型在训练环境（100秒）下过拟合")

if __name__ == "__main__":
    test_latency_scaling()
