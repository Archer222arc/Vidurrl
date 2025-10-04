#!/usr/bin/env python3
"""
PPO training script for toy model queue scheduling.

Trains a PPO agent to learn optimal routing policies.
"""

import os
import json
import argparse
import numpy as np
import torch
from typing import Dict, Any

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Training will continue without logging.")

from toymodel.src.environment import QueueEnvironment
from toymodel.src.config import load_config
from toymodel.src.rl_components import (
    SimpleActorCritic, 
    SimplePPOTrainer, 
    SimpleRolloutBuffer,
    QueueStateBuilder,
    LatencyRewardCalculator,
    create_latency_predictor
)
from toymodel.schedulers.ppo_scheduler import PPOScheduler


class PPOTrainer:
    """
    PPO trainer for toy model queue scheduling.
    
    Manages the complete training loop including environment interaction,
    experience collection, and policy updates.
    """

    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize PPO trainer.
        
        Args:
            config: Training configuration
            device: Device for computations
        """
        self.config = config
        self.device = device
        
        # Environment configuration
        self.env = QueueEnvironment(
            num_replicas=config.environment.num_replicas,
            service_rates=config.environment.service_rates,
            arrival_rates=config.environment.arrival_rates,
            max_time=config.environment.max_time,
            seed=config.experiment.seed,
            tensorboard_enabled=config.tensorboard.enabled,
            tensorboard_log_dir=config.tensorboard.log_dir
        )
        
        # PPO configuration - load from JSON file directly
        import json
        with open('toymodel/configs/ppo_config.json', 'r') as f:
            ppo_config = json.load(f)['ppo']
        self.ppo_config = ppo_config  # Save for later use
        self.n_requests = ppo_config.get('n_requests', 3)
        
        # Initialize components
        self.state_builder = QueueStateBuilder(
            num_replicas=config.environment.num_replicas,
            n_requests=self.n_requests,
            normalize=True
        )
        # Create latency predictor if enabled
        use_prediction = ppo_config.get('use_prediction', False)
        prediction_weight = ppo_config.get('prediction_weight', 0.5)
        predictor_type = ppo_config.get('predictor_type', 'simple')
        impact_weight = ppo_config.get('impact_weight', 1.0)

        if use_prediction:
            latency_predictor = create_latency_predictor(
                predictor_type=predictor_type,
                prediction_weight=prediction_weight,
                impact_weight=impact_weight,
                num_replicas=config.environment.num_replicas,
                num_request_types=len(config.environment.arrival_rates)
            )
        else:
            latency_predictor = None
        
        self.reward_calculator = LatencyRewardCalculator(
            latency_weight=ppo_config.get('latency_weight', 1.0),
            use_prediction=use_prediction,
            prediction_weight=prediction_weight,
            latency_predictor=latency_predictor
        )
        
        # Store predictor for updating
        self.latency_predictor = latency_predictor
        
        # Get dimensions from state builder
        self.state_dim = self.state_builder.get_state_dim()
        self.action_dim = self.state_builder.get_action_dim()
        
        # Policy network
        self.policy = SimpleActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=ppo_config.get('hidden_dim', 64)
        ).to(device)
        
        # PPO trainer
        self.ppo_trainer = SimplePPOTrainer(
            policy=self.policy,
            learning_rate=ppo_config.get('learning_rate', 3e-4),
            clip_ratio=ppo_config.get('clip_ratio', 0.2),
            entropy_coef=ppo_config.get('entropy_coef', 0.01),
            value_coef=ppo_config.get('value_coef', 0.5),
            epochs=ppo_config.get('epochs', 4),
            minibatch_size=ppo_config.get('minibatch_size', 64),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            device=device
        )
        
        # Rollout buffer
        self.rollout_buffer = SimpleRolloutBuffer(
            state_dim=self.state_dim,
            buffer_size=ppo_config.get('rollout_length', 1000),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            device=device
        )
        
        # Training parameters
        self.num_episodes = ppo_config.get('num_episodes', 1000)
        self.eval_interval = ppo_config.get('eval_interval', 100)
        self.save_interval = ppo_config.get('save_interval', 200)
        self.max_episode_length = ppo_config.get('max_episode_length', 1000)
        
        # Output directories - use config output_dir
        self.output_dir = config.output.dir
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.eval_dir = os.path.join(self.output_dir, 'eval')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = []
        self.global_step = 0  # Global step counter for TensorBoard
        
        # Track all historical completed requests across all episodes
        self.all_historical_requests = []
        
        # TensorBoard writer
        self.tb_writer = None
        if config.tensorboard.enabled and TENSORBOARD_AVAILABLE:
            # Create unique run directory with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ppo_run_{timestamp}"
            tb_log_dir = os.path.join(config.tensorboard.log_dir, run_name)
            self.tb_writer = SummaryWriter(tb_log_dir)
            print(f"TensorBoard logging enabled: {tb_log_dir}")
            print(f"Run name: {run_name}")
            
            # Log hyperparameters to TensorBoard
            self.tb_writer.add_hparams({
                'learning_rate': ppo_config.get('learning_rate', 3e-4),
                'clip_ratio': ppo_config.get('clip_ratio', 0.2),
                'entropy_coef': ppo_config.get('entropy_coef', 0.01),
                'value_coef': ppo_config.get('value_coef', 0.5),
                'epochs': ppo_config.get('epochs', 4),
                'rollout_length': ppo_config.get('rollout_length', 1000),
                'max_episode_length': ppo_config.get('max_episode_length', 1000),
                'gamma': ppo_config.get('gamma', 0.99),
                'gae_lambda': ppo_config.get('gae_lambda', 0.95),
                'minibatch_size': ppo_config.get('minibatch_size', 64),
                'max_grad_norm': ppo_config.get('max_grad_norm', 0.5),
                'latency_weight': ppo_config.get('latency_weight', 1.0),
                'n_requests': ppo_config.get('n_requests', 3),
                'hidden_dim': ppo_config.get('hidden_dim', 64)
            }, {
                'final_reward': 0.0,  # Will be updated at the end
                'final_latency': 0.0,  # Will be updated at the end
                'final_accuracy': 0.0  # Will be updated at the end
            })
            
        elif config.tensorboard.enabled and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but not available. Continuing without logging.")
    
    def collect_rollout(self) -> Dict[str, Any]:
        """
        Collect a single rollout episode.
        
        Returns:
            Dictionary of rollout statistics
        """
        self.rollout_buffer.reset()
        self.env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        completed_requests = []
        
        # Create PPO scheduler for this episode
        scheduler = PPOScheduler(
            num_replicas=self.env.num_replicas,
            n_requests=self.n_requests,
            device=self.device
        )
        scheduler.policy = self.policy  # Use current policy
        
        # Run episode
        while episode_length < self.max_episode_length:
            # Get next request
            request = self.env.step_until_next_arrival()
            if request is None:
                break
            
            # Build state
            state = self.state_builder.build_state(request, self.env.replicas)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action_and_value(
                    state.unsqueeze(0).to(self.device)
                )
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()
            
            # Route request
            self.env.route_request(request, action)
            
            # Use actual average latency from completed requests (recorded by environment)
            # This matches the evaluation's latency definition and uses real data
            if len(self.env.completed_requests) > 0:
                # Calculate average latency from recent completed requests
                recent_requests = self.env.completed_requests[-min(10, len(self.env.completed_requests)):]
                avg_latency = np.mean([req.total_time for req in recent_requests])
                latency = avg_latency
            else:
                # Fallback to theoretical service time if no completed requests yet
                service_rate = self.env.replicas[action].get_service_rate(request.request_type)
                latency = 1.0 / service_rate
            
            reward_dict = self.reward_calculator.calculate_reward(
                request, self.env.replicas, action, latency
            )
            reward = reward_dict['total']

            # Add to buffer
            self.rollout_buffer.add(
                state=state,
                action=torch.tensor(action, dtype=torch.long),
                reward=reward,
                value=torch.tensor(value, dtype=torch.float32),
                log_prob=torch.tensor(log_prob, dtype=torch.float32),
                done=False  # Episodes don't have natural termination
            )
            
            episode_reward += reward
            episode_length += 1
            
            # Check if buffer is full
            if self.rollout_buffer.is_full():
                break
        
        # Process remaining requests
        self.env._drain_queues()
        
        # Add final completed requests to statistics
        completed_requests.extend(self.env.completed_requests)
        
        # Add to historical tracking
        self.all_historical_requests.extend(self.env.completed_requests)
        
        # Update latency predictor with completed requests
        if self.latency_predictor is not None:
            for req in self.env.completed_requests:
                if req.is_completed and req.assigned_replica is not None:
                    # Calculate actual processing time (service_time only)
                    actual_processing_time = req.service_time
                    self.latency_predictor.update_processing_time(
                        replica_id=req.assigned_replica,
                        request_type=req.request_type,
                        actual_processing_time=actual_processing_time
                    )
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'completed_requests': len(completed_requests),
            'buffer_size': self.rollout_buffer.size
        }
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected rollouts.
        
        Returns:
            Dictionary of training statistics
        """
        if self.rollout_buffer.size == 0:
            return {}
        
        # Get bootstrap value for final state
        with torch.no_grad():
            final_state = self.rollout_buffer.states[-1].unsqueeze(0).to(self.device)
            _, _, final_value = self.policy.get_action_and_value(final_state)
            final_value = final_value.item()
        
        # Compute GAE advantages and returns
        states, actions, log_probs, values, returns, advantages = self.rollout_buffer.compute_gae(
            torch.tensor(final_value, dtype=torch.float32)
        )
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy
        stats = self.ppo_trainer.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            old_values=values,
            returns=returns,
            advantages=advantages
        )
        
        return stats
    
    def evaluate_policy(self) -> Dict[str, float]:
        """
        Evaluate current policy performance using average latency of all historical completed tasks.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Use average latency from all historical completed requests across all episodes
        if self.all_historical_requests:
            # Get all historical completed requests from training
            all_latencies = [req.total_time for req in self.all_historical_requests]
            all_routing_accuracy = sum(
                1 for req in self.all_historical_requests 
                if req.assigned_replica == req.request_type
            ) / len(self.all_historical_requests)
            
            metrics = {
                'mean_latency': np.mean(all_latencies),
                'p50_latency': np.percentile(all_latencies, 50),
                'p99_latency': np.percentile(all_latencies, 99),
                'routing_accuracy': all_routing_accuracy,
                'total_requests': len(self.all_historical_requests)
            }
        else:
            # Fallback: run a quick evaluation episode
            eval_scheduler = PPOScheduler(
                num_replicas=self.env.num_replicas,
                n_requests=self.n_requests,
                device=self.device
            )
            eval_scheduler.policy = self.policy
            eval_scheduler.set_eval_mode()
            
            # Run evaluation episode
            self.env.reset()
            completed_requests = self.env.run_simulation(eval_scheduler.schedule)
            
            # Calculate metrics
            if completed_requests:
                latencies = [req.total_time for req in completed_requests]
                routing_accuracy = sum(
                    1 for req in completed_requests 
                    if req.assigned_replica == req.request_type
                ) / len(completed_requests)
                
                metrics = {
                    'mean_latency': np.mean(latencies),
                    'p50_latency': np.percentile(latencies, 50),
                    'p99_latency': np.percentile(latencies, 99),
                    'routing_accuracy': routing_accuracy,
                    'total_requests': len(completed_requests)
                }
            else:
                metrics = {
                    'mean_latency': 0.0,
                    'p50_latency': 0.0,
                    'p99_latency': 0.0,
                    'routing_accuracy': 0.0,
                    'total_requests': 0
                }
        
        return metrics
    
    def save_model(self, episode: int):
        """
        Save model checkpoint.
        
        Args:
            episode: Current episode number
        """
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        model_path = os.path.join(self.model_dir, f'ppo_model_episode_{episode}.pt')
        torch.save(checkpoint, model_path)
        
        # Also save as latest
        latest_path = os.path.join(self.model_dir, 'ppo_model_latest.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"Saved model checkpoint at episode {episode}")
    
    def train(self):
        """Run complete training loop."""
        print("=" * 80)
        print("Starting PPO training in Vidur environment...")
        print("=" * 80)
        print(f"Training for {self.num_episodes} episodes")
        print(f"Device: {self.device}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Environment: Vidur")
        print(f"TensorBoard enabled: {self.tb_writer is not None}")
        if self.tb_writer:
            print(f"TensorBoard log dir: {self.tb_writer.log_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Model dir: {self.model_dir}")
        print(f"Eval dir: {self.eval_dir}")
        print("=" * 80)
        
        for episode in range(1, self.num_episodes + 1):
            # Collect rollout
            rollout_stats = self.collect_rollout()
            
            # Update policy
            if self.rollout_buffer.size > 0:
                training_stats = self.update_policy()
            else:
                training_stats = {}
            
            # Store statistics
            self.episode_rewards.append(rollout_stats['episode_reward'])
            self.episode_lengths.append(rollout_stats['episode_length'])
            
            episode_stats = {
                'episode': episode,
                'episode_reward': rollout_stats['episode_reward'],
                'episode_length': rollout_stats['episode_length'],
                'completed_requests': rollout_stats['completed_requests'],
                'buffer_size': rollout_stats['buffer_size'],
                **training_stats
            }
            self.training_stats.append(episode_stats)
            
            # Update global step counter (increment by episode length)
            self.global_step += rollout_stats['episode_length']
            
            # Debug: Print global step every 10 episodes
            if episode % 10 == 0:
                print(f"  - Global step: {self.global_step}")
            
            # Log every episode to TensorBoard (not just every 10)
            if self.tb_writer:
                self.tb_writer.add_scalar('Episode/Reward_Individual', rollout_stats['episode_reward'], self.global_step)
                self.tb_writer.add_scalar('Episode/Length_Individual', rollout_stats['episode_length'], self.global_step)
                self.tb_writer.add_scalar('System/Buffer_Size_Individual', rollout_stats['buffer_size'], self.global_step)
                
                # Log training metrics for every episode
                if training_stats:
                    for key, value in training_stats.items():
                        if key in ['pi_loss', 'vf_loss']:
                            self.tb_writer.add_scalar(f'Loss/{key}_Individual', value, self.global_step)
                        elif key in ['entropy']:
                            self.tb_writer.add_scalar(f'Policy/{key}_Individual', value, self.global_step)
                        elif key in ['approx_kl', 'clipfrac']:
                            self.tb_writer.add_scalar(f'Policy/{key}_Individual', value, self.global_step)
                        elif key in ['grad_norm']:
                            self.tb_writer.add_scalar(f'Optimization/{key}_Individual', value, self.global_step)
            
            # Print progress and log to TensorBoard
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, Avg Length = {avg_length:.1f}")
                print(f"  - Rollout length: {rollout_stats['buffer_size']}")
                print(f"  - Completed requests: {rollout_stats['completed_requests']}")
                
                # Print predictor estimates for 4 combinations
                if self.latency_predictor is not None and hasattr(self.latency_predictor, 'processing_times'):
                    print(f"  - Predictor estimates:")
                    for (replica_id, request_type), (avg_time, count) in self.latency_predictor.processing_times.items():
                        print(f"    Replica {replica_id}, Type {request_type}: {avg_time:.3f} (samples: {count})")
                    if not self.latency_predictor.processing_times:
                        print(f"    No data yet (using defaults)")
                
                # Log to TensorBoard with organized groups
                if self.tb_writer:
                    # Episode metrics
                    self.tb_writer.add_scalar('Episode/Reward', rollout_stats['episode_reward'], self.global_step)
                    self.tb_writer.add_scalar('Episode/Length', rollout_stats['episode_length'], self.global_step)
                    self.tb_writer.add_scalar('Episode/Avg_Reward_10', avg_reward, self.global_step)
                    self.tb_writer.add_scalar('Episode/Avg_Length_10', avg_length, self.global_step)
                    
                    # System metrics
                    self.tb_writer.add_scalar('System/Rollout_Length', rollout_stats['buffer_size'], self.global_step)
                    self.tb_writer.add_scalar('System/Completed_Requests', rollout_stats['completed_requests'], self.global_step)
                    
                    # Training metrics with detailed categorization
                    for key, value in training_stats.items():
                        if key in ['pi_loss', 'vf_loss']:
                            self.tb_writer.add_scalar(f'Loss/{key}', value, self.global_step)
                        elif key in ['entropy']:
                            self.tb_writer.add_scalar(f'Policy/{key}', value, self.global_step)
                        elif key in ['approx_kl', 'clipfrac']:
                            self.tb_writer.add_scalar(f'Policy/{key}', value, self.global_step)
                        elif key in ['grad_norm']:
                            self.tb_writer.add_scalar(f'Optimization/{key}', value, self.global_step)
                        else:
                            self.tb_writer.add_scalar(f'Training/{key}', value, self.global_step)
                        print(f"  - {key}: {value:.4f}")
                    
                    # Additional training statistics
                    self.tb_writer.add_scalar('Training/Total_Episodes', episode, self.global_step)
                    self.tb_writer.add_scalar('Training/Total_Steps', self.global_step, self.global_step)
                    
                    # Learning rate tracking
                    current_lr = self.ppo_trainer.optimizer.param_groups[0]['lr']
                    self.tb_writer.add_scalar('Optimization/Learning_Rate', current_lr, self.global_step)
            
            # Evaluation
            if episode % self.eval_interval == 0:
                eval_metrics = self.evaluate_policy()
                
                # Display historical average latency from all completed requests (aligned with compare_schedulers.py)
                if self.all_historical_requests:
                    # Calculate metrics exactly like compare_schedulers.py
                    all_latencies = [req.total_time for req in self.all_historical_requests]
                    all_routing_accuracy = sum(
                        1 for req in self.all_historical_requests 
                        if req.assigned_replica == req.request_type
                    ) / len(self.all_historical_requests) * 100  # Convert to percentage like compare_schedulers.py
                    
                    mean_latency = np.mean(all_latencies)
                    p50_latency = np.percentile(all_latencies, 50)
                    p99_latency = np.percentile(all_latencies, 99)
                    total_requests = len(self.all_historical_requests)
                    
                    print(f"Episode {episode} - Eval: Mean Latency = {mean_latency:.3f}, "
                          f"P50 = {p50_latency:.3f}, P99 = {p99_latency:.3f} "
                          f"(from {total_requests} historical completed requests), "
                          f"Accuracy = {all_routing_accuracy:.1f}%")
                else:
                    print(f"Episode {episode} - Eval: Latency = {eval_metrics['mean_latency']:.3f}, "
                          f"Accuracy = {eval_metrics['routing_accuracy']:.3f}")
                
                # Store evaluation metrics for final logging
                self.eval_metrics = eval_metrics
                
                # Log evaluation metrics to TensorBoard with specific naming
                if self.tb_writer:
                    # Use specific names to avoid cross-contamination
                    self.tb_writer.add_scalar('Evaluation/Latency_Mean', eval_metrics.get('mean_latency', 0.0), self.global_step)
                    self.tb_writer.add_scalar('Evaluation/Latency_Std', eval_metrics.get('std_latency', 0.0), self.global_step)
                    self.tb_writer.add_scalar('Evaluation/Routing_Accuracy', eval_metrics.get('routing_accuracy', 0.0), self.global_step)
                    self.tb_writer.add_scalar('Evaluation/Throughput', eval_metrics.get('throughput', 0.0), self.global_step)
                    
                    # Log additional metrics if available
                    for key, value in eval_metrics.items():
                        if key not in ['mean_latency', 'std_latency', 'routing_accuracy', 'throughput']:
                            self.tb_writer.add_scalar(f'Evaluation/{key}', value, self.global_step)
                
                # Save evaluation results
                eval_path = os.path.join(self.eval_dir, f'eval_episode_{episode}.json')
                with open(eval_path, 'w') as f:
                    json.dump(eval_metrics, f, indent=2)
            
            # Save model
            if episode % self.save_interval == 0:
                self.save_model(episode)
        
        # Save final model
        self.save_model(self.num_episodes)
        
        # Save training statistics
        stats_path = os.path.join(self.output_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()
        
        print("Training completed!")
        
        # Log final training statistics
        if self.tb_writer:
            # Final episode statistics
            if self.episode_rewards:
                final_reward = np.mean(self.episode_rewards[-10:])
                max_reward = np.max(self.episode_rewards)
                min_reward = np.min(self.episode_rewards)
                reward_std = np.std(self.episode_rewards)
                
                self.tb_writer.add_scalar('Final/Max_Reward', max_reward, self.global_step)
                self.tb_writer.add_scalar('Final/Min_Reward', min_reward, self.global_step)
                self.tb_writer.add_scalar('Final/Reward_Std', reward_std, self.global_step)
                self.tb_writer.add_scalar('Final/Avg_Reward_Last_10', final_reward, self.global_step)
            
            if self.episode_lengths:
                avg_length = np.mean(self.episode_lengths)
                self.tb_writer.add_scalar('Final/Avg_Episode_Length', avg_length, self.global_step)
            
            # Final evaluation metrics
            final_latency = 0.0
            final_accuracy = 0.0
            
            if hasattr(self, 'eval_metrics') and self.eval_metrics:
                final_latency = self.eval_metrics.get('mean_latency', 0.0)
                final_accuracy = self.eval_metrics.get('routing_accuracy', 0.0)
                
                self.tb_writer.add_scalar('Final/Latency', final_latency, self.global_step)
                self.tb_writer.add_scalar('Final/Accuracy', final_accuracy, self.global_step)
            
            # Update hparams with final metrics
            self.tb_writer.add_hparams({
                'learning_rate': self.ppo_config.get('learning_rate', 3e-4),
                'clip_ratio': self.ppo_config.get('clip_ratio', 0.2),
                'entropy_coef': self.ppo_config.get('entropy_coef', 0.01),
                'value_coef': self.ppo_config.get('value_coef', 0.5),
                'epochs': self.ppo_config.get('epochs', 4),
                'rollout_length': self.ppo_config.get('rollout_length', 1000),
                'max_episode_length': self.ppo_config.get('max_episode_length', 1000),
                'gamma': self.ppo_config.get('gamma', 0.99),
                'gae_lambda': self.ppo_config.get('gae_lambda', 0.95),
                'minibatch_size': self.ppo_config.get('minibatch_size', 64),
                'max_grad_norm': self.ppo_config.get('max_grad_norm', 0.5),
                'latency_weight': self.ppo_config.get('latency_weight', 1.0),
                'n_requests': self.ppo_config.get('n_requests', 3),
                'hidden_dim': self.ppo_config.get('hidden_dim', 64)
            }, {
                'final_reward': final_reward if 'final_reward' in locals() else 0.0,
                'final_latency': final_latency,
                'final_accuracy': final_accuracy
            })
            
            self.tb_writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PPO for toy model queue scheduling')
    parser.add_argument('--config', type=str, default='toymodel/configs/ppo_config.json',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for training (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = PPOTrainer(config, device=args.device)
    
    # Run training
    trainer.train()


if __name__ == '__main__':
    main()
