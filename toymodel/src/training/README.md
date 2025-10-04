# Training Module

Training infrastructure for toy model components.

## Structure

```
training/
├── __init__.py              # Module exports
├── README.md                # This file
└── predictor/               # Neural latency predictor training
    ├── __init__.py
    ├── data_collector.py    # LatencyDataCollector
    ├── predictor_trainer.py # PredictorTrainer, LatencyDataset
    └── README.md            # Detailed predictor training docs
```

## Submodules

### predictor/

Training components for neural latency predictor:
- **Data Collection**: Policy-independent simulation for training data
- **Supervised Learning**: MSE regression with validation and early stopping

See [`predictor/README.md`](predictor/README.md) for detailed documentation.

## Usage

### Python API

```python
from toymodel.src.training import (
    LatencyDataCollector,
    PredictorTrainer,
    LatencyDataset
)

# Collect data
collector = LatencyDataCollector(config, policy='mixed')
states, labels = collector.collect_data(num_episodes=200)

# Train predictor
trainer = PredictorTrainer(predictor)
stats = trainer.train(train_loader, val_loader)
```

### Command Line

```bash
# Collect data
python toymodel/scripts/collect_latency_data.py --num_episodes 200

# Train predictor
python toymodel/scripts/train_predictor_offline.py --epochs 100

# One-click pipeline
./toymodel/scripts/train_predictor.sh
```

## Future Extensions

This module can be extended to include training for other components:
- `training/ppo/` - PPO policy training utilities
- `training/evaluation/` - Model evaluation and benchmarking
- `training/data_augmentation/` - Data augmentation strategies
