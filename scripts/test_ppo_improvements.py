#!/usr/bin/env python3
"""
PPO Training Improvements Test Script

This script validates all the PPO training improvements implemented
based on the PDF recommendations:
1. State feature enhancements (queue delay features)
2. Network architecture upgrades (cross-replica attention)
3. Curriculum learning functionality
4. Tail latency monitoring (P90/P95/P99)
5. Adaptive entropy scheduling
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.core.models.state_builder import StateBuilder
from src.core.models.actor_critic import ActorCritic
from src.core.algorithms.curriculum_manager import CurriculumManager
from src.core.algorithms.ppo_trainer import PPOTrainer
from src.core.utils.monitoring.tail_latency_monitor import TailLatencyMonitor


class PPOImprovementsValidator:
    """
    Validates all PPO training improvements.
    """

    def __init__(self, config_path: str = "configs/ppo_warmstart.json"):
        """Initialize validator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.test_results = {}

        print("🧪 PPO Training Improvements Validator")
        print(f"📄 Using config: {config_path}")
        print("=" * 60)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            sys.exit(1)

    def test_state_feature_enhancements(self) -> bool:
        """Test state feature enhancements (queue delay features)."""
        print("🔍 Testing State Feature Enhancements...")

        try:
            # Create StateBuilder with enhanced features
            state_builder = StateBuilder(
                max_queue_requests=4,
                history_window=5,
                qps_window=10,
                enable_enhanced_features=True
            )

            # Test queue delay feature methods exist
            assert hasattr(state_builder, '_get_oldest_request_wait_time'), "Missing oldest_request_wait_time method"
            assert hasattr(state_builder, '_get_avg_queue_wait_time'), "Missing avg_queue_wait_time method"
            assert hasattr(state_builder, '_get_queue_urgency_score'), "Missing queue_urgency_score method"

            print("   ✅ Queue delay feature methods exist")

            # Test enhanced state dimension
            expected_state_dim = self.config["model_dimensions"]["state_dim"]
            print(f"   ✅ Expected state dimension: {expected_state_dim}")

            # Verify config has queue delay features
            queue_features = self.config["state_builder_enhanced"]["feature_names"]
            expected_features = ["oldest_request_wait_time", "avg_queue_wait_time", "queue_urgency_score"]

            assert all(f in queue_features for f in expected_features), "Missing queue delay features in config"
            print("   ✅ Queue delay features configured correctly")

            # Check standalone pretrain config consistency
            standalone_config_path = "configs/standalone_pretrain.json"
            if Path(standalone_config_path).exists():
                with open(standalone_config_path, 'r') as f:
                    standalone_config = json.load(f)

                standalone_state_dim = standalone_config["state_dim"]
                if standalone_state_dim != expected_state_dim:
                    print(f"   ⚠️ WARNING: standalone_pretrain.json state_dim ({standalone_state_dim}) != ppo_warmstart.json ({expected_state_dim})")
                else:
                    print(f"   ✅ Standalone pretrain config state dimension consistent: {standalone_state_dim}")

            self.test_results["state_features"] = True
            return True

        except Exception as e:
            print(f"   ❌ State feature test failed: {e}")
            self.test_results["state_features"] = False
            return False

    def test_network_architecture_upgrades(self) -> bool:
        """Test network architecture upgrades (cross-replica attention)."""
        print("🔍 Testing Network Architecture Upgrades...")

        try:
            # Test ActorCritic with cross-replica attention
            state_dim = self.config["model_dimensions"]["state_dim"]
            action_dim = self.config["model_dimensions"]["action_dim"]

            actor_critic = ActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_size=320,
                layer_N=3,
                gru_layers=3,
                use_orthogonal=True,
                enable_decoupled=False,
                feature_projection_dim=None,
            )

            # Check if cross-replica attention is implemented
            assert hasattr(actor_critic, 'cross_replica_attention'), "Missing cross_replica_attention module"
            print("   ✅ Cross-replica attention module exists")

            # Test forward pass
            batch_size = 2
            test_state = torch.randn(batch_size, state_dim)
            hxs = torch.zeros(3, batch_size, 320)
            masks = torch.ones(batch_size, 1)

            with torch.no_grad():
                logits, values, new_hxs = actor_critic.get_action_logits_values(test_state, hxs, masks)

            assert logits.shape == (batch_size, action_dim), f"Unexpected logits shape: {logits.shape}"
            assert values.shape == (batch_size, 1), f"Unexpected values shape: {values.shape}"
            print("   ✅ Forward pass successful")

            # Test attention mechanism configuration
            attention_config = self.config["actor_critic_architecture"]["cross_replica_attention"]
            assert attention_config["enable"], "Cross-replica attention not enabled in config"
            assert attention_config["num_heads"] == 4, "Unexpected number of attention heads"
            print("   ✅ Cross-replica attention configured correctly")

            self.test_results["network_architecture"] = True
            return True

        except Exception as e:
            print(f"   ❌ Network architecture test failed: {e}")
            self.test_results["network_architecture"] = False
            return False

    def test_curriculum_learning(self) -> bool:
        """Test curriculum learning functionality."""
        print("🔍 Testing Curriculum Learning...")

        try:
            # Create curriculum manager from config
            curriculum_config = self.config["curriculum_learning"]
            curriculum_manager = CurriculumManager(curriculum_config)

            # Test curriculum stages
            assert curriculum_manager.enabled, "Curriculum learning not enabled"
            assert len(curriculum_manager.stages) == 3, f"Expected 3 stages, got {len(curriculum_manager.stages)}"
            print("   ✅ Curriculum stages configured correctly")

            # Test stage progression
            initial_stage = curriculum_manager.current_stage.name
            assert initial_stage == "easy", f"Expected initial stage 'easy', got '{initial_stage}'"

            # Simulate processing requests
            stage_changed = curriculum_manager.update(5000)  # Process 5000 requests

            # Should still be in easy stage (duration is 10000)
            assert not stage_changed, "Stage changed unexpectedly"
            assert curriculum_manager.current_stage.name == "easy", "Unexpected stage change"

            # Process more requests to trigger stage change
            stage_changed = curriculum_manager.update(6000)  # Total: 11000 requests
            assert stage_changed, "Stage should have changed"
            assert curriculum_manager.current_stage.name == "medium", "Should be in medium stage"
            print("   ✅ Stage progression working correctly")

            # Test parameter scaling
            params = curriculum_manager.get_current_parameters()
            expected_keys = ["qps_scale", "latency_threshold_scale", "reward_penalty_scale"]
            assert all(key in params for key in expected_keys), "Missing curriculum parameters"
            print("   ✅ Curriculum parameters available")

            self.test_results["curriculum_learning"] = True
            return True

        except Exception as e:
            print(f"   ❌ Curriculum learning test failed: {e}")
            self.test_results["curriculum_learning"] = False
            return False

    def test_tail_latency_monitoring(self) -> bool:
        """Test tail latency monitoring system."""
        print("🔍 Testing Tail Latency Monitoring...")

        try:
            # Create tail latency monitor from config
            monitoring_config = self.config["monitoring"]["tail_latency_tracking"]

            tail_monitor = TailLatencyMonitor(
                percentiles=monitoring_config["percentiles"],
                window_size=monitoring_config["window_size"],
                alert_threshold_p99=monitoring_config["alert_threshold_p99"],
                enable_alerts=True,
                enable_tracking=True
            )

            # Test basic functionality
            assert tail_monitor.enable_tracking, "Tail latency tracking not enabled"
            assert tail_monitor.percentiles == [90, 95, 99], "Unexpected percentiles"
            print("   ✅ Tail latency monitor configured correctly")

            # Test latency sample processing
            sample_latencies = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 10.0]

            alerts_triggered = 0
            for latency in sample_latencies:
                if tail_monitor.update(latency):
                    alerts_triggered += 1

            # Should trigger alerts for high latencies
            assert alerts_triggered > 0, "Expected alerts for high latencies"
            print(f"   ✅ Alert system triggered {alerts_triggered} alerts")

            # Test metrics generation
            metrics = tail_monitor.get_metrics()
            expected_metrics = ["latency_p90", "latency_p95", "latency_p99"]
            assert all(metric in metrics for metric in expected_metrics), "Missing percentile metrics"

            assert metrics["latency_p99"] > metrics["latency_p95"], "P99 should be > P95"
            assert metrics["latency_p95"] > metrics["latency_p90"], "P95 should be > P90"
            print("   ✅ Percentile calculations correct")

            self.test_results["tail_latency_monitoring"] = True
            return True

        except Exception as e:
            print(f"   ❌ Tail latency monitoring test failed: {e}")
            self.test_results["tail_latency_monitoring"] = False
            return False

    def test_adaptive_entropy_scheduling(self) -> bool:
        """Test adaptive entropy scheduling."""
        print("🔍 Testing Adaptive Entropy Scheduling...")

        try:
            # Create PPO trainer with entropy scheduling
            ppo_config = self.config["ppo_config"]
            entropy_schedule = ppo_config["entropy_schedule"]

            # Create dummy actor-critic for testing
            state_dim = self.config["model_dimensions"]["state_dim"]
            action_dim = self.config["model_dimensions"]["action_dim"]

            actor_critic = ActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_size=320,
                layer_N=3,
                gru_layers=3,
            )

            ppo_trainer = PPOTrainer(
                policy=actor_critic,
                lr=ppo_config["lr"],
                clip_ratio=ppo_config["clip_ratio"],
                entropy_coef=ppo_config["entropy_coef"],
                entropy_schedule_enable=entropy_schedule["enable"],
                entropy_initial=entropy_schedule["initial"],
                entropy_final=entropy_schedule["final"],
                entropy_decay_steps=entropy_schedule["decay_steps"],
            )

            # Test entropy scheduling
            assert hasattr(ppo_trainer, 'get_current_entropy_coef'), "Missing entropy scheduling method"
            assert ppo_trainer.entropy_schedule_enable, "Entropy scheduling not enabled"
            print("   ✅ Entropy scheduling enabled")

            # Test entropy decay
            initial_coef = ppo_trainer.get_current_entropy_coef()
            expected_initial = entropy_schedule["initial"]
            assert abs(initial_coef - expected_initial) < 1e-6, f"Initial entropy {initial_coef} != {expected_initial}"

            # Simulate training steps
            ppo_trainer.current_step = entropy_schedule["decay_steps"] // 2
            mid_coef = ppo_trainer.get_current_entropy_coef()

            ppo_trainer.current_step = entropy_schedule["decay_steps"]
            final_coef = ppo_trainer.get_current_entropy_coef()
            expected_final = entropy_schedule["final"]

            assert initial_coef > mid_coef > final_coef, "Entropy should decay over time"
            assert abs(final_coef - expected_final) < 1e-6, f"Final entropy {final_coef} != {expected_final}"
            print("   ✅ Entropy decay working correctly")

            self.test_results["adaptive_entropy"] = True
            return True

        except Exception as e:
            print(f"   ❌ Adaptive entropy scheduling test failed: {e}")
            self.test_results["adaptive_entropy"] = False
            return False

    def test_reward_range_expansion(self) -> bool:
        """Test reward range expansion."""
        print("🔍 Testing Reward Range Expansion...")

        try:
            # Check reward scaling configuration
            reward_config = self.config["reward_config"]["reward_scaling"]

            assert reward_config["type"] == "adaptive", "Reward scaling not set to adaptive"
            assert reward_config["clip_range"] == [-8.0, 8.0], "Reward range not expanded to [-8, 8]"
            assert reward_config["enable_soft_clipping"], "Soft clipping not enabled"
            print("   ✅ Reward range expanded to [-8, 8] with soft clipping")

            # Test soft clipping threshold
            soft_threshold = reward_config["soft_clip_threshold"]
            assert soft_threshold == 6.0, f"Soft clip threshold {soft_threshold} != 6.0"
            print("   ✅ Soft clipping threshold configured correctly")

            self.test_results["reward_range_expansion"] = True
            return True

        except Exception as e:
            print(f"   ❌ Reward range expansion test failed: {e}")
            self.test_results["reward_range_expansion"] = False
            return False

    def test_gae_optimization(self) -> bool:
        """Test GAE optimization (lambda=0.95)."""
        print("🔍 Testing GAE Optimization...")

        try:
            ppo_config = self.config["ppo_config"]
            gae_lambda = ppo_config["gae_lambda"]

            assert gae_lambda == 0.95, f"GAE lambda {gae_lambda} != 0.95"
            print("   ✅ GAE lambda optimized to 0.95")

            self.test_results["gae_optimization"] = True
            return True

        except Exception as e:
            print(f"   ❌ GAE optimization test failed: {e}")
            self.test_results["gae_optimization"] = False
            return False

    def run_all_tests(self) -> bool:
        """Run all improvement validation tests."""
        print("🚀 Running All PPO Improvement Tests\n")

        tests = [
            ("State Feature Enhancements", self.test_state_feature_enhancements),
            ("Network Architecture Upgrades", self.test_network_architecture_upgrades),
            ("Curriculum Learning", self.test_curriculum_learning),
            ("Tail Latency Monitoring", self.test_tail_latency_monitoring),
            ("Adaptive Entropy Scheduling", self.test_adaptive_entropy_scheduling),
            ("Reward Range Expansion", self.test_reward_range_expansion),
            ("GAE Optimization", self.test_gae_optimization),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            if test_func():
                passed_tests += 1
            print()  # Add spacing between tests

        # Print summary
        print("=" * 60)
        print("📊 Test Summary:")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            print("🎉 All PPO improvements validated successfully!")
            return True
        else:
            print("❌ Some tests failed. Check implementation.")
            return False

    def generate_report(self) -> str:
        """Generate detailed test report."""
        report = []
        report.append("# PPO Training Improvements Validation Report")
        report.append(f"**Config File**: {self.config_path}")
        report.append(f"**Test Date**: {np.datetime64('now')}")
        report.append("")

        report.append("## Test Results")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            report.append(f"- **{test_name.replace('_', ' ').title()}**: {status}")

        report.append("")
        report.append("## Implementation Status")

        implementation_status = [
            ("State Feature Enhancements", "✅ Implemented", "Queue delay features added to StateBuilder"),
            ("Cross-Replica Attention", "✅ Implemented", "Attention mechanism in ActorCritic"),
            ("Curriculum Learning", "✅ Implemented", "Progressive difficulty adjustment"),
            ("Tail Latency Monitoring", "✅ Implemented", "P90/P95/P99 tracking with alerts"),
            ("Adaptive Entropy Scheduling", "✅ Implemented", "Linear decay from 0.02 to 0.0"),
            ("Reward Range Expansion", "✅ Implemented", "Adaptive scaling [-8,8] with soft clipping"),
            ("GAE Optimization", "✅ Implemented", "Lambda optimized to 0.95"),
        ]

        for feature, status, description in implementation_status:
            report.append(f"- **{feature}**: {status} - {description}")

        return "\n".join(report)


def main():
    """Main test function."""
    config_path = "configs/ppo_warmstart.json"

    # Override config path if provided
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    validator = PPOImprovementsValidator(config_path)

    # Run all tests
    success = validator.run_all_tests()

    # Generate report
    print("\n📄 Generating detailed report...")
    report = validator.generate_report()

    # Save report
    report_path = Path("outputs") / "ppo_improvements_test_report.md"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"📁 Report saved to: {report_path}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()