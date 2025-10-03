#!/usr/bin/env python3
"""
Clear TensorBoard logs to avoid overlapping plots.
"""

import os
import shutil
import argparse

def clear_tensorboard_logs(log_dir: str = "toymodel/outputs/tensorboard"):
    """Clear all TensorBoard log directories."""
    if os.path.exists(log_dir):
        print(f"Clearing TensorBoard logs from: {log_dir}")
        shutil.rmtree(log_dir)
        print("✅ TensorBoard logs cleared successfully!")
    else:
        print(f"❌ TensorBoard log directory not found: {log_dir}")

def main():
    parser = argparse.ArgumentParser(description='Clear TensorBoard logs')
    parser.add_argument('--log_dir', type=str, 
                       default='toymodel/outputs/tensorboard',
                       help='TensorBoard log directory to clear')
    
    args = parser.parse_args()
    clear_tensorboard_logs(args.log_dir)

if __name__ == "__main__":
    main()

