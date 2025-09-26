#!/usr/bin/env python3
"""
Revolutionary PPO Collapse Prevention System Deployment Script.

This script integrates all the revolutionary solutions developed based on
the deep analysis of policy collapse patterns in load balancing systems.

Key components integrated:
1. Enhanced Collapse Detection with aggressive early warning (CV > 0.3)
2. Gradient Preservation Suite to prevent entropy = 0.0000 periods
3. Dynamic Reward System with progressive penalties
4. Emergency Intervention with 20x entropy boost
5. Real-time monitoring with 10-step detection frequency

Based on collapse analysis:
- CV progression: 0.228 → 1.317 → 1.627 → 1.697
- Reward degradation: -4.29 → -11.24
- Entropy collapse: 0.0000 for extended periods
- Late intervention: 0 → 232 interventions (failed)

Usage:
    python scripts/deploy_revolutionary_collapse_prevention.py --config configs/revolutionary_collapse_prevention.json
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.algorithms.enhanced_collapse_detector import EnhancedCollapseDetector
from src.core.algorithms.gradient_preservation import GradientPreservationSuite
from src.core.algorithms.dynamic_reward_system import DynamicRewardSystem

logger = logging.getLogger(__name__)


class RevolutionaryCollapsePreventionSystem:
    """
    Revolutionary collapse prevention system integrating all advanced techniques.

    This system implements the complete solution to PPO collapse based on
    empirical analysis of failure patterns in production load balancing systems.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the revolutionary collapse prevention system.

        Args:
            config: Complete system configuration
        """
        self.config = config
        self.num_replicas = config['training']['num_replicas']

        # Extract subsystem configurations
        collapse_config = config['cluster_config']['global_scheduler_config'].get(
            'enhanced_collapse_detection', {}
        )
        gradient_config = config['ppo_config']['revolutionary_features'].get(
            'gradient_preservation', {}
        )
        reward_config = config['cluster_config']['global_scheduler_config'].get(
            'dynamic_reward_system', {}
        )

        # Initialize core components
        self.collapse_detector = EnhancedCollapseDetector(
            num_actions=self.num_replicas,
            cv_warning_threshold=collapse_config.get('cv_warning_threshold', 0.3),
            cv_emergency_threshold=collapse_config.get('cv_emergency_threshold', 0.8),
            min_action_freq_threshold=collapse_config.get('min_action_freq_threshold', 0.15),
            gradient_norm_threshold=collapse_config.get('gradient_norm_threshold', 1e-6),
            entropy_collapse_threshold=collapse_config.get('entropy_collapse_threshold', 0.1),
            emergency_entropy_boost=collapse_config.get('emergency_entropy_boost', 20.0),
            forced_exploration_steps=collapse_config.get('forced_exploration_steps', 50),
            intervention_cooldown=collapse_config.get('intervention_cooldown', 10),
            detection_window=collapse_config.get('detection_window', 20),
            history_window=collapse_config.get('history_window', 100),
            enable_adaptive_thresholds=collapse_config.get('enable_adaptive_thresholds', True),
            performance_decline_threshold=collapse_config.get('performance_decline_threshold', -0.1)
        )

        self.dynamic_reward_system = DynamicRewardSystem(
            num_actions=self.num_replicas,
            config={
                'imbalance_penalty_weight': reward_config.get('imbalance_penalty_weight', 2.0),
                'emergency_penalty_multiplier': reward_config.get('emergency_penalty_multiplier', 10.0),
                'target_reward_range': reward_config.get('target_reward_range', [-2.0, 2.0]),
                'normalization_rate': reward_config.get('normalization_rate', 0.05),
                'emergency_boost_strength': reward_config.get('emergency_boost_strength', 3.0),
                'emergency_boost_duration': reward_config.get('emergency_boost_duration', 100),
                'adaptive_objective_weights': reward_config.get('adaptive_objective_weights', True),
                'objective_weights': reward_config.get('objective_weights', {
                    'throughput': 1.0,
                    'latency': 1.5,
                    'fairness': 2.0,
                    'balance': 3.0
                })
            }
        )

        # Will be initialized with actual model and optimizer
        self.gradient_preservation_suite = None
        self.gradient_config = gradient_config

        # System state
        self.step_count = 0
        self.total_interventions = 0
        self.system_status = "NORMAL"
        self.last_intervention_step = -1000

        # Performance tracking
        self.performance_history = []
        self.collapse_events = []

        logger.info("🚀 Revolutionary Collapse Prevention System initialized")
        logger.info(f"   • Collapse Detection: CV thresholds {collapse_config.get('cv_warning_threshold', 0.3):.2f}/{collapse_config.get('cv_emergency_threshold', 0.8):.2f}")
        logger.info(f"   • Emergency Boost: {collapse_config.get('emergency_entropy_boost', 20.0)}x entropy increase")
        logger.info(f"   • Detection Frequency: Every {collapse_config.get('detection_window', 20)} steps")
        logger.info(f"   • Min Action Frequency: {collapse_config.get('min_action_freq_threshold', 0.15):.1%}")

    def initialize_gradient_preservation(self, model, optimizer):
        """Initialize gradient preservation suite with actual model and optimizer."""
        self.gradient_preservation_suite = GradientPreservationSuite(
            model=model,
            optimizer=optimizer,
            config={
                'min_gradient_norm': self.gradient_config.get('min_gradient_norm', 1e-6),
                'max_gradient_norm': self.gradient_config.get('max_gradient_norm', 10.0),
                'emergency_boost_factor': self.gradient_config.get('emergency_boost_factor', 100.0),
                'use_spectral_norm': self.gradient_config.get('use_spectral_norm', True),
                'monitor_frequency': self.gradient_config.get('monitor_frequency', 10),
                'injection_strength': self.gradient_config.get('injection_strength', 0.01),
                'base_lr': optimizer.param_groups[0]['lr'],
                'emergency_lr_boost': self.gradient_config.get('emergency_lr_boost', 10.0)
            }
        )
        logger.info("🧠 Gradient Preservation Suite initialized")

    def process_training_step(
        self,
        action: int,
        reward: float,
        entropy: float,
        gradient_norm: float,
        objective_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Process a single training step through all revolutionary systems.

        Args:
            action: Action taken by the policy
            reward: Raw reward received
            entropy: Current policy entropy
            gradient_norm: Current gradient norm
            objective_values: Individual objective values

        Returns:
            Dictionary with processed results and system status
        """
        self.step_count += 1

        # 1. Update collapse detection metrics
        self.collapse_detector.update_metrics(
            action=action,
            reward=reward,
            entropy=entropy,
            gradient_norm=gradient_norm,
            step=self.step_count
        )

        # 2. Run collapse detection
        detection_result = self.collapse_detector.detect_collapse()
        alert_level = detection_result['alert_level']
        needs_intervention = detection_result['needs_intervention']

        # 3. Process reward through dynamic system
        system_state = {
            'coefficient_variation': detection_result.get('details', {}).get(
                'cv_analysis', {}
            ).get('coefficient_variation', 0.0),
            'performance_decline': detection_result.get('details', {}).get(
                'performance_analysis', {}
            ).get('performance_decline', 0.0)
        }

        processed_reward, reward_stats = self.dynamic_reward_system.process_reward(
            raw_reward=reward,
            action=action,
            objective_values=objective_values,
            system_state=system_state,
            emergency_mode=(alert_level == 'EMERGENCY')
        )

        # 4. Apply gradient preservation if available
        gradient_stats = {}
        if self.gradient_preservation_suite:
            gradient_stats = self.gradient_preservation_suite.preserve_gradients(
                force_emergency=(alert_level == 'EMERGENCY')
            )

        # 5. Handle interventions
        intervention_applied = False
        if needs_intervention and self.step_count > self.last_intervention_step + 10:
            intervention_type = self._determine_intervention_type(detection_result)
            intervention_applied = self.collapse_detector.apply_intervention(intervention_type)
            if intervention_applied:
                self.total_interventions += 1
                self.last_intervention_step = self.step_count
                logger.warning(f"🚨 INTERVENTION #{self.total_interventions} at step {self.step_count}: "
                              f"{intervention_type} (Alert: {alert_level})")

        # 6. Update system status
        self.system_status = alert_level
        if alert_level == 'EMERGENCY':
            self.collapse_events.append({
                'step': self.step_count,
                'cv': detection_result.get('details', {}).get(
                    'cv_analysis', {}
                ).get('coefficient_variation', 0.0),
                'entropy': entropy,
                'gradient_norm': gradient_norm,
                'intervention_applied': intervention_applied
            })

        # 7. Track performance
        self.performance_history.append({
            'step': self.step_count,
            'raw_reward': reward,
            'processed_reward': processed_reward,
            'entropy': entropy,
            'cv': system_state.get('coefficient_variation', 0.0),
            'alert_level': alert_level
        })

        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

        # 8. Generate comprehensive status report
        status_report = {
            'step': self.step_count,
            'system_status': self.system_status,
            'alert_level': alert_level,
            'intervention_applied': intervention_applied,
            'total_interventions': self.total_interventions,

            'rewards': {
                'raw_reward': reward,
                'processed_reward': processed_reward,
                'reward_processing': reward_stats
            },

            'collapse_detection': detection_result,

            'gradient_health': gradient_stats,

            'system_metrics': {
                'cv': system_state.get('coefficient_variation', 0.0),
                'entropy': entropy,
                'gradient_norm': gradient_norm,
                'performance_decline': system_state.get('performance_decline', 0.0)
            },

            'forced_exploration': self.collapse_detector.should_force_exploration(),

            'recommendations': self._generate_recommendations(detection_result, reward_stats, gradient_stats)
        }

        return status_report

    def _determine_intervention_type(self, detection_result: Dict[str, Any]) -> str:
        """Determine the type of intervention based on detection results."""
        alert_level = detection_result['alert_level']
        interventions = detection_result.get('interventions', {})

        if alert_level == 'EMERGENCY':
            if interventions.get('reset_to_checkpoint', False):
                return 'SYSTEM_RESET'
            elif 'GRADIENT_BOOST' in interventions.get('emergency_actions', []):
                return 'EMERGENCY_GRADIENT_BOOST'
            else:
                return 'EMERGENCY_ENTROPY_BOOST'
        elif alert_level == 'WARNING':
            return 'WARNING_ENTROPY_BOOST'
        else:
            return 'GENTLE_ADJUSTMENT'

    def _generate_recommendations(
        self,
        detection_result: Dict[str, Any],
        reward_stats: Dict[str, Any],
        gradient_stats: Dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations based on system state."""
        recommendations = []

        # Collapse-related recommendations
        alert_level = detection_result['alert_level']
        if alert_level == 'EMERGENCY':
            recommendations.append("🚨 CRITICAL: Policy collapse detected - applying emergency interventions")
            recommendations.append("📊 Monitor action distribution closely over next 100 steps")
            recommendations.append("🔄 Consider checkpoint rollback if collapse persists")
        elif alert_level == 'WARNING':
            recommendations.append("⚠️ Early collapse indicators - preventive measures active")
            recommendations.append("📈 Increase monitoring frequency")

        # Gradient-related recommendations
        if gradient_stats.get('gradient_health_score', 1.0) < 0.5:
            recommendations.append("🧠 Gradient health degraded - preservation measures active")

        # Reward-related recommendations
        if reward_stats.get('emergency_boost', {}).get('boost_active', False):
            recommendations.append("🎯 Emergency reward boost active - expect temporary reward increases")

        # Performance recommendations
        cv = detection_result.get('details', {}).get('cv_analysis', {}).get('coefficient_variation', 0.0)
        if cv > 1.0:
            recommendations.append("📉 Severe load imbalance - consider increasing exploration")

        if not recommendations:
            recommendations.append("✅ System operating normally - all metrics healthy")

        return recommendations

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        return {
            'system_overview': {
                'step_count': self.step_count,
                'system_status': self.system_status,
                'total_interventions': self.total_interventions,
                'collapse_events_count': len(self.collapse_events)
            },

            'collapse_detector_status': self.collapse_detector.get_status_summary(),

            'reward_system_status': self.dynamic_reward_system.get_system_status(),

            'gradient_preservation_status': (
                self.gradient_preservation_suite.get_status()
                if self.gradient_preservation_suite else {}
            ),

            'recent_performance': self.performance_history[-10:] if self.performance_history else [],

            'collapse_events': self.collapse_events,

            'configuration_summary': {
                'cv_warning_threshold': self.collapse_detector.cv_warning_threshold,
                'cv_emergency_threshold': self.collapse_detector.cv_emergency_threshold,
                'emergency_entropy_boost': self.collapse_detector.emergency_entropy_boost,
                'detection_window': self.collapse_detector.detection_window
            }
        }

    def export_performance_data(self, output_path: str) -> None:
        """Export performance data for analysis."""
        import pandas as pd

        if not self.performance_history:
            logger.warning("No performance data to export")
            return

        df = pd.DataFrame(self.performance_history)
        df.to_csv(output_path, index=False)
        logger.info(f"📊 Performance data exported to {output_path}")


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('revolutionary_collapse_prevention.log')
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    required_sections = [
        'training',
        'ppo_config',
        'cluster_config',
        'monitoring'
    ]

    for section in required_sections:
        if section not in config:
            logger.error(f"❌ Missing required configuration section: {section}")
            return False

    logger.info("✅ Configuration validation passed")
    return True


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy Revolutionary PPO Collapse Prevention System"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without deployment'
    )
    parser.add_argument(
        '--export-sample-data',
        action='store_true',
        help='Export sample performance data for testing'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("🚀 Revolutionary PPO Collapse Prevention System")
    logger.info("=" * 60)

    try:
        # Load and validate configuration
        config = load_config(args.config)
        if not validate_config(config):
            sys.exit(1)

        if args.validate_only:
            logger.info("✅ Configuration validation completed successfully")
            sys.exit(0)

        # Initialize the revolutionary system
        prevention_system = RevolutionaryCollapsePreventionSystem(config)

        logger.info("🎯 System Components Initialized:")
        logger.info(f"   • Enhanced Collapse Detector: ✅")
        logger.info(f"   • Dynamic Reward System: ✅")
        logger.info(f"   • Gradient Preservation Suite: 🔄 (requires model)")

        # Display key settings
        logger.info("⚙️ Key Configuration Settings:")
        collapse_config = config['cluster_config']['global_scheduler_config'].get(
            'enhanced_collapse_detection', {}
        )
        logger.info(f"   • CV Warning Threshold: {collapse_config.get('cv_warning_threshold', 0.3)}")
        logger.info(f"   • CV Emergency Threshold: {collapse_config.get('cv_emergency_threshold', 0.8)}")
        logger.info(f"   • Emergency Entropy Boost: {collapse_config.get('emergency_entropy_boost', 20.0)}x")
        logger.info(f"   • Detection Window: {collapse_config.get('detection_window', 20)} steps")

        if args.export_sample_data:
            logger.info("📊 Exporting sample performance data...")
            # Create sample data for testing
            import numpy as np
            sample_data = []
            for i in range(1000):
                sample_data.append({
                    'step': i,
                    'raw_reward': np.random.normal(-5, 2),
                    'processed_reward': np.random.normal(-2, 1),
                    'entropy': max(0.01, np.random.normal(1.0, 0.5)),
                    'cv': max(0.1, np.random.normal(0.4, 0.2)),
                    'alert_level': np.random.choice(['NORMAL', 'WARNING', 'EMERGENCY'], p=[0.7, 0.2, 0.1])
                })
            prevention_system.performance_history = sample_data
            prevention_system.export_performance_data('sample_performance_data.csv')

        logger.info("✅ Revolutionary Collapse Prevention System ready for deployment!")
        logger.info("🎯 Integration points:")
        logger.info("   1. Call initialize_gradient_preservation(model, optimizer) after model creation")
        logger.info("   2. Call process_training_step() during each training iteration")
        logger.info("   3. Monitor system status and apply recommendations")

    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()