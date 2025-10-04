#!/usr/bin/env python3
"""
Compare performance of different latency predictors.

Loads trained PPO models and evaluates their performance.
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from toymodel.src.environment import QueueEnvironment
from toymodel.src.config import load_config
from toymodel.src.rl_components import (
    SimpleActorCritic,
    QueueStateBuilder,
    LatencyRewardCalculator,
    create_latency_predictor
)
from toymodel.schedulers.ppo_scheduler import PPOScheduler


class PredictorComparator:
    """Compare performance of different latency predictors."""

    def __init__(self, output_dir: str = "toymodel/outputs/comparison"):
        """
        Initialize comparator.

        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {}

    def load_model(self, config_path: str, model_path: str) -> tuple:
        """
        Load trained model.

        Args:
            config_path: Path to configuration file
            model_path: Path to model checkpoint

        Returns:
            Tuple of (config, policy, predictor_type)
        """
        # Load configuration
        config = load_config(config_path)

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Get PPO configuration
        ppo_config = config.ppo
        n_requests = ppo_config.n_requests

        # Create state builder
        state_builder = QueueStateBuilder(
            num_replicas=config.environment.num_replicas,
            n_requests=n_requests,
            normalize=True
        )

        state_dim = state_builder.get_state_dim()
        action_dim = state_builder.get_action_dim()

        # Create policy
        policy = SimpleActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=ppo_config.hidden_dim
        )

        # Load policy weights
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()

        predictor_type = ppo_config.predictor_type

        return config, policy, predictor_type

    def evaluate_model(
        self,
        config: Any,
        policy: torch.nn.Module,
        predictor_type: str,
        num_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            config: Configuration object
            policy: Trained policy
            predictor_type: Type of predictor ('simple' or 'system_aware')
            num_episodes: Number of evaluation episodes

        Returns:
            Dictionary of evaluation metrics
        """
        ppo_config = config.ppo

        # Create environment
        env = QueueEnvironment(
            num_replicas=config.environment.num_replicas,
            service_rates=config.environment.service_rates,
            arrival_rates=config.environment.arrival_rates,
            max_time=config.environment.max_time,
            seed=config.experiment.seed
        )

        # Create state builder
        state_builder = QueueStateBuilder(
            num_replicas=config.environment.num_replicas,
            n_requests=ppo_config.n_requests,
            normalize=True
        )

        # Collect metrics across episodes
        all_latencies = []
        all_accuracies = []
        all_rewards = []

        for episode in range(num_episodes):
            env.reset()

            episode_latencies = []
            episode_correct = 0
            episode_total = 0
            episode_reward = 0.0

            # Run episode
            for step in range(ppo_config.max_episode_length):
                # Get next request
                request = env.step_until_next_arrival()
                if request is None:
                    break

                # Build state
                state = state_builder.build_state(request, env.replicas)

                # Get action from policy
                with torch.no_grad():
                    action, _, _ = policy.get_action_and_value(state.unsqueeze(0))
                    action = action.item()

                # Route request
                env.route_request(request, action)

                # Track accuracy
                episode_total += 1
                if action == request.request_type:
                    episode_correct += 1

            # Process remaining requests
            env._drain_queues()

            # Collect episode metrics
            if env.completed_requests:
                episode_latencies = [req.total_time for req in env.completed_requests]
                all_latencies.extend(episode_latencies)

                episode_accuracy = episode_correct / episode_total if episode_total > 0 else 0.0
                all_accuracies.append(episode_accuracy)

        # Calculate aggregate metrics
        metrics = {
            'predictor_type': predictor_type,
            'num_episodes': num_episodes,
            'total_requests': len(all_latencies),
            'mean_latency': np.mean(all_latencies),
            'std_latency': np.std(all_latencies),
            'p50_latency': np.percentile(all_latencies, 50),
            'p95_latency': np.percentile(all_latencies, 95),
            'p99_latency': np.percentile(all_latencies, 99),
            'min_latency': np.min(all_latencies),
            'max_latency': np.max(all_latencies),
            'mean_accuracy': np.mean(all_accuracies) * 100,  # Convert to percentage
            'std_accuracy': np.std(all_accuracies) * 100
        }

        return metrics

    def compare_models(
        self,
        models: Dict[str, Dict[str, str]],
        num_episodes: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models.

        Args:
            models: Dictionary mapping model name to config/model paths
                   Format: {'name': {'config': 'path/to/config.json',
                                     'model': 'path/to/model.pt'}}
            num_episodes: Number of evaluation episodes per model

        Returns:
            Dictionary mapping model name to evaluation metrics
        """
        results = {}

        for name, paths in models.items():
            print(f"\n{'=' * 80}")
            print(f"Evaluating: {name}")
            print(f"{'=' * 80}")

            # Load model
            config, policy, predictor_type = self.load_model(
                paths['config'],
                paths['model']
            )

            # Evaluate model
            metrics = self.evaluate_model(
                config,
                policy,
                predictor_type,
                num_episodes
            )

            results[name] = metrics

            # Print metrics
            print(f"\nPredictor Type: {predictor_type}")
            print(f"Total Requests: {metrics['total_requests']}")
            print(f"Mean Latency: {metrics['mean_latency']:.4f} ± {metrics['std_latency']:.4f}")
            print(f"P50 Latency: {metrics['p50_latency']:.4f}")
            print(f"P95 Latency: {metrics['p95_latency']:.4f}")
            print(f"P99 Latency: {metrics['p99_latency']:.4f}")
            print(f"Latency Range: [{metrics['min_latency']:.4f}, {metrics['max_latency']:.4f}]")
            print(f"Routing Accuracy: {metrics['mean_accuracy']:.2f}% ± {metrics['std_accuracy']:.2f}%")

        self.results = results
        return results

    def save_results(self):
        """Save comparison results to JSON."""
        output_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to: {output_path}")

    def plot_comparison(self):
        """Generate comparison plots."""
        if not self.results:
            print("No results to plot. Run compare_models() first.")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Predictor Performance Comparison', fontsize=16, fontweight='bold')

        model_names = list(self.results.keys())

        # 1. Mean Latency Comparison
        ax = axes[0, 0]
        mean_latencies = [self.results[name]['mean_latency'] for name in model_names]
        std_latencies = [self.results[name]['std_latency'] for name in model_names]

        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, mean_latencies, yerr=std_latencies, capsize=5, alpha=0.7)
        ax.set_xlabel('Predictor Type')
        ax.set_ylabel('Mean Latency')
        ax.set_title('Mean Latency with Standard Deviation')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # 2. Latency Percentiles Comparison
        ax = axes[0, 1]
        p50_latencies = [self.results[name]['p50_latency'] for name in model_names]
        p95_latencies = [self.results[name]['p95_latency'] for name in model_names]
        p99_latencies = [self.results[name]['p99_latency'] for name in model_names]

        width = 0.25
        x_pos = np.arange(len(model_names))
        ax.bar(x_pos - width, p50_latencies, width, label='P50', alpha=0.7)
        ax.bar(x_pos, p95_latencies, width, label='P95', alpha=0.7)
        ax.bar(x_pos + width, p99_latencies, width, label='P99', alpha=0.7)

        ax.set_xlabel('Predictor Type')
        ax.set_ylabel('Latency')
        ax.set_title('Latency Percentiles')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 3. Routing Accuracy Comparison
        ax = axes[1, 0]
        accuracies = [self.results[name]['mean_accuracy'] for name in model_names]
        std_accuracies = [self.results[name]['std_accuracy'] for name in model_names]

        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, accuracies, yerr=std_accuracies, capsize=5, alpha=0.7, color='green')
        ax.set_xlabel('Predictor Type')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Routing Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)

        # 4. Summary Table
        ax = axes[1, 1]
        ax.axis('off')

        # Create comparison table
        table_data = []
        headers = ['Metric', 'Simple', 'System Aware', 'Improvement']

        if len(model_names) >= 2:
            simple_results = self.results[model_names[0]]
            system_aware_results = self.results[model_names[1]]

            # Mean Latency
            simple_latency = simple_results['mean_latency']
            system_latency = system_aware_results['mean_latency']
            latency_improvement = ((simple_latency - system_latency) / simple_latency) * 100
            table_data.append([
                'Mean Latency',
                f"{simple_latency:.4f}",
                f"{system_latency:.4f}",
                f"{latency_improvement:+.2f}%"
            ])

            # P99 Latency
            simple_p99 = simple_results['p99_latency']
            system_p99 = system_aware_results['p99_latency']
            p99_improvement = ((simple_p99 - system_p99) / simple_p99) * 100
            table_data.append([
                'P99 Latency',
                f"{simple_p99:.4f}",
                f"{system_p99:.4f}",
                f"{p99_improvement:+.2f}%"
            ])

            # Accuracy
            simple_acc = simple_results['mean_accuracy']
            system_acc = system_aware_results['mean_accuracy']
            acc_improvement = system_acc - simple_acc
            table_data.append([
                'Accuracy (%)',
                f"{simple_acc:.2f}",
                f"{system_acc:.2f}",
                f"{acc_improvement:+.2f}%"
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Performance Comparison Summary', fontweight='bold', pad=20)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, 'predictor_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {plot_path}")

        plt.close()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare predictor performance')
    parser.add_argument('--simple-config', type=str,
                       default='toymodel/configs/ppo_config_simple.json',
                       help='Path to simple predictor config')
    parser.add_argument('--simple-model', type=str,
                       default='toymodel/outputs/models/ppo_model_latest.pt',
                       help='Path to simple predictor model')
    parser.add_argument('--system-aware-config', type=str,
                       default='toymodel/configs/ppo_config_system_aware.json',
                       help='Path to system-aware predictor config')
    parser.add_argument('--system-aware-model', type=str,
                       default='toymodel/outputs/models/ppo_model_latest.pt',
                       help='Path to system-aware predictor model')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--output-dir', type=str,
                       default='toymodel/outputs/comparison',
                       help='Directory to save comparison results')

    args = parser.parse_args()

    # Create comparator
    comparator = PredictorComparator(output_dir=args.output_dir)

    # Define models to compare
    models = {
        'simple_predictor': {
            'config': args.simple_config,
            'model': args.simple_model
        },
        'system_aware_predictor': {
            'config': args.system_aware_config,
            'model': args.system_aware_model
        }
    }

    # Compare models
    print("\n" + "=" * 80)
    print("PREDICTOR PERFORMANCE COMPARISON")
    print("=" * 80)

    results = comparator.compare_models(models, num_episodes=args.num_episodes)

    # Save results
    comparator.save_results()

    # Generate plots
    comparator.plot_comparison()

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
