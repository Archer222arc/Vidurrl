# Training Components

Modular components for training the neural latency predictor.

## Overview

This module provides policy-independent training infrastructure for the neural latency predictor:
- **Data Collection**: Per-replica state-latency pairs from simulation
- **Supervised Learning**: MSE regression with validation and early stopping

## Components

### 1. LatencyDataCollector (`data_collector.py`)

Collects training data using policy-independent simulation.

**Key Features**:
- Per-replica design: Extracts features for ALL replicas per request
- Policy-independent: Uses simple policies (random/round-robin/mixed)
- Ground truth only: Labels data only for actually-routed replicas

**Usage**:
```python
from toymodel.src.training import LatencyDataCollector
from toymodel.src.config import load_config

config = load_config('toymodel/configs/ppo_config.json')
collector = LatencyDataCollector(config, policy='mixed')

# Collect data from 100 episodes
states, labels = collector.collect_data(num_episodes=100)

# Get statistics
stats = collector.get_statistics()
```

### 2. PredictorTrainer (`predictor_trainer.py`)

Trains neural predictor using supervised learning.

**Key Features**:
- AdamW optimizer with L2 regularization
- ReduceLROnPlateau scheduler
- Early stopping (patience=20)
- Gradient clipping (max_norm=1.0)

**Usage**:
```python
from toymodel.src.training import PredictorTrainer, LatencyDataset
from toymodel.src.predictors import NeuralLatencyPredictor
from torch.utils.data import DataLoader

# Create predictor
predictor = NeuralLatencyPredictor(
    num_replicas=2,
    num_request_types=2,
    hidden_dim=128
)

# Create dataset and loaders
dataset = LatencyDataset(states, labels)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Train
trainer = PredictorTrainer(predictor, learning_rate=1e-3)
stats = trainer.train(train_loader, val_loader, num_epochs=100)
```

## Scripts

### Data Collection

```bash
python toymodel/scripts/collect_latency_data.py \
    --config toymodel/configs/ppo_config.json \
    --policy mixed \
    --num_episodes 200 \
    --output toymodel/data/latency_training_data.pkl
```

**Arguments**:
- `--config`: Configuration file path
- `--policy`: Collection policy (random/round_robin/mixed)
- `--num_episodes`: Number of simulation episodes
- `--output`: Output dataset path

### Training

```bash
python toymodel/scripts/train_predictor_offline.py \
    --data toymodel/data/latency_training_data.pkl \
    --output toymodel/outputs/predictor_checkpoint.pt \
    --hidden_dim 128 \
    --epochs 100 \
    --batch_size 128
```

**Arguments**:
- `--data`: Training dataset path
- `--output`: Model checkpoint output path
- `--hidden_dim`: Hidden layer dimension
- `--learning_rate`: Learning rate (default: 1e-3)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--val_split`: Validation split ratio (default: 0.2)

### One-Click Pipeline

```bash
./toymodel/scripts/train_predictor.sh
```

Executes complete pipeline:
1. Collect 200 episodes of data
2. Train for 100 epochs with early stopping
3. Save checkpoint to `toymodel/outputs/predictor_checkpoint.pt`

## Data Format

### Collected Dataset
```python
{
    'states': np.ndarray,      # (N, 523) state features
    'labels': np.ndarray,      # (N, 2) [self_latency, avg_impact]
    'config': {
        'num_replicas': int,
        'num_request_types': int,
        'service_rates': dict,
        'arrival_rates': dict,
        'policy': str
    },
    'stats': {
        'num_samples': int,
        'mean_self_latency': float,
        'std_self_latency': float,
        'mean_impact': float,
        'std_impact': float
    }
}
```

### Model Checkpoint
```python
{
    'model_state_dict': OrderedDict,
    'num_replicas': int,
    'num_request_types': int,
    'hidden_dim': int,
    'max_queue_obs': int,
    'input_dim': int,
    'train_stats': {
        'final_train_loss': float,
        'final_val_loss': float,
        'best_val_loss': float,
        'train_losses': list,
        'val_losses': list
    }
}
```

## Architecture

### Neural Predictor Model
```
Input(523) → Linear(128) → LayerNorm → ReLU → Dropout(0.1)
           → Linear(128) → LayerNorm → ReLU → Dropout(0.1)
           → Linear(64)  → ReLU
           → Linear(2)   → [self_latency, avg_impact]
```

**Input Features (523 dimensions)**:
- Current request type (1)
- Per-replica features (261 × 2):
  - Queue length (1)
  - Queue request types (128)
  - Queue position mask (128)
  - Service rates (2)
  - Current serving type (1)
  - Busy until time (1)

## Integration with PPO

After training, use the predictor in PPO by updating `ppo_config.json`:
```json
{
  "ppo": {
    "predictor_type": "learned",
    "max_queue_obs": 128,
    "checkpoint_path": "toymodel/outputs/predictor_checkpoint.pt"
  }
}
```

## Performance Considerations

**Data Collection**:
- ~2-3 episodes/second
- ~60K samples from 200 episodes
- Collection time: ~2-3 minutes

**Training**:
- ~30K samples/epoch
- Training time: ~5-10 minutes (100 epochs, CPU)
- Early stopping typically at 40-60 epochs

## Design Principles

1. **Per-Replica Prediction**: Predictor is replica-specific, not action-specific
2. **Policy Independence**: Training data collected with simple policies
3. **Ground Truth Only**: No counterfactual labels to avoid bias
4. **Modular Design**: Separate data collection and training concerns
5. **Reproducibility**: Fixed seeds and deterministic training
