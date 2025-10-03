#!/usr/bin/env python3
"""
Test TensorBoard logging to ensure all metrics are properly recorded.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

def test_tensorboard_logging():
    """Test TensorBoard logging with sample data."""
    
    # Create test log directory
    log_dir = "toymodel/outputs/tensorboard/test_run"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    print("Testing TensorBoard logging...")
    
    # Simulate training data
    for episode in range(1, 21):
        # Episode metrics
        reward = np.random.normal(100, 20)
        length = 32
        writer.add_scalar('Episode/Reward', reward, episode)
        writer.add_scalar('Episode/Length', length, episode)
        writer.add_scalar('Episode/Reward_Individual', reward, episode)
        
        # Training metrics
        pi_loss = np.random.normal(-0.001, 0.0005)
        vf_loss = np.random.normal(0.5, 0.1)
        entropy = np.random.normal(0.6, 0.1)
        grad_norm = np.random.normal(0.5, 0.2)
        
        writer.add_scalar('Loss/pi_loss', pi_loss, episode)
        writer.add_scalar('Loss/vf_loss', vf_loss, episode)
        writer.add_scalar('Policy/entropy', entropy, episode)
        writer.add_scalar('Optimization/grad_norm', grad_norm, episode)
        
        # Individual metrics
        writer.add_scalar('Loss/pi_loss_Individual', pi_loss, episode)
        writer.add_scalar('Loss/vf_loss_Individual', vf_loss, episode)
        writer.add_scalar('Policy/entropy_Individual', entropy, episode)
        writer.add_scalar('Optimization/grad_norm_Individual', grad_norm, episode)
        
        # Evaluation metrics (every 5 episodes)
        if episode % 5 == 0:
            latency = np.random.normal(1.0, 0.2)
            accuracy = np.random.normal(0.55, 0.05)
            
            writer.add_scalar('Evaluation/Latency_Mean', latency, episode)
            writer.add_scalar('Evaluation/Routing_Accuracy', accuracy, episode)
        
        # System metrics
        writer.add_scalar('System/Rollout_Length', 32, episode)
        writer.add_scalar('System/Completed_Requests', 32, episode)
        
        print(f"Episode {episode}: Reward={reward:.2f}, Loss={vf_loss:.3f}")
    
    # Log hyperparameters
    writer.add_hparams({
        'learning_rate': 3e-4,
        'clip_ratio': 0.2,
        'entropy_coef': 0.1,
        'value_coef': 0.1,
        'rollout_length': 32,
        'num_episodes': 200
    }, {
        'final_reward': 120.0,
        'final_latency': 0.8,
        'final_accuracy': 0.58
    })
    
    writer.close()
    print(f"✅ Test data written to: {log_dir}")
    print("Check TensorBoard at: http://localhost:6006")

if __name__ == "__main__":
    test_tensorboard_logging()

