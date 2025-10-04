#!/bin/bash
# Complete pipeline: collect data + train predictor

set -e  # Exit on error

echo "======================================================================"
echo "Neural Latency Predictor Training Pipeline"
echo "======================================================================"

# Step 1: Collect training data
echo ""
echo "Step 1/2: Collecting training data..."
echo "----------------------------------------------------------------------"
python3 toymodel/scripts/collect_latency_data.py \
    --config toymodel/configs/ppo_config.json \
    --policy mixed \
    --num_episodes 200 \
    --output toymodel/data/latency_training_data.pkl

# Step 2: Train predictor
echo ""
echo "Step 2/2: Training neural predictor..."
echo "----------------------------------------------------------------------"
python3 toymodel/scripts/train_predictor_offline.py \
    --data toymodel/data/latency_training_data.pkl \
    --output toymodel/outputs/predictor_checkpoint.pt \
    --hidden_dim 128 \
    --learning_rate 1e-3 \
    --batch_size 128 \
    --epochs 100 \
    --val_split 0.2

echo ""
echo "======================================================================"
echo "✅ Pipeline completed successfully!"
echo "======================================================================"
echo "Trained model saved to: toymodel/outputs/predictor_checkpoint.pt"
echo ""
echo "To use the trained predictor in PPO training, update ppo_config.json:"
echo '  "predictor_type": "learned",'
echo '  "checkpoint_path": "toymodel/outputs/predictor_checkpoint.pt"'
