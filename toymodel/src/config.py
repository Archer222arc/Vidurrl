"""
Configuration management for toy model.

Loads and validates configuration from JSON files.
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str
    seed: int


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    num_replicas: int
    max_time: float
    service_rates: Dict[int, Dict[int, float]]
    arrival_rates: Dict[int, float]


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    type: str
    options: Dict[str, Any]


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    save_csv: bool
    csv_path: str
    save_raw_data: bool


@dataclass
class TensorBoardConfig:
    """TensorBoard configuration."""
    enabled: bool
    log_dir: str
    port: int
    clean_previous_runs: bool = True


@dataclass
class ToyModelConfig:
    """
    Complete toy model configuration.

    Loaded from JSON file and validated.
    """
    experiment: ExperimentConfig
    environment: EnvironmentConfig
    scheduler: SchedulerConfig
    metrics: MetricsConfig
    tensorboard: TensorBoardConfig

    @classmethod
    def from_json(cls, json_path: str) -> "ToyModelConfig":
        """
        Load configuration from JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            ToyModelConfig instance

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config validation fails
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        # Validate required sections
        required_sections = ['experiment', 'environment', 'scheduler', 'metrics']
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required config section: {section}")

        # Parse experiment config
        exp_dict = config_dict['experiment']
        experiment = ExperimentConfig(
            name=exp_dict['name'],
            description=exp_dict['description'],
            seed=exp_dict['seed'],
        )

        # Parse environment config
        env_dict = config_dict['environment']

        # Convert service_rates keys from string to int
        service_rates = {}
        for replica_id_str, rates in env_dict['service_rates'].items():
            replica_id = int(replica_id_str)
            service_rates[replica_id] = {
                int(req_type): float(rate)
                for req_type, rate in rates.items()
            }

        # Convert arrival_rates keys from string to int
        arrival_rates = {
            int(req_type): float(rate)
            for req_type, rate in env_dict['arrival_rates'].items()
        }

        environment = EnvironmentConfig(
            num_replicas=env_dict['num_replicas'],
            max_time=env_dict['max_time'],
            service_rates=service_rates,
            arrival_rates=arrival_rates,
        )

        # Parse scheduler config
        sched_dict = config_dict['scheduler']
        scheduler = SchedulerConfig(
            type=sched_dict['type'],
            options=sched_dict.get('options', {}),
        )

        # Parse metrics config
        metrics_dict = config_dict['metrics']
        metrics = MetricsConfig(
            save_csv=metrics_dict['save_csv'],
            csv_path=metrics_dict['csv_path'],
            save_raw_data=metrics_dict['save_raw_data'],
        )

        # Parse tensorboard config
        tb_dict = config_dict.get('tensorboard', {
            'enabled': False,
            'log_dir': 'outputs/toymodel/tensorboard',
            'port': 6006,
            'clean_previous_runs': True
        })
        tensorboard = TensorBoardConfig(
            enabled=tb_dict.get('enabled', False),
            log_dir=tb_dict.get('log_dir', 'outputs/toymodel/tensorboard'),
            port=tb_dict.get('port', 6006),
            clean_previous_runs=tb_dict.get('clean_previous_runs', True),
        )

        return cls(
            experiment=experiment,
            environment=environment,
            scheduler=scheduler,
            metrics=metrics,
            tensorboard=tensorboard,
        )

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If validation fails
        """
        # Validate num_replicas
        if self.environment.num_replicas != 2:
            raise ValueError(f"num_replicas must be 2, got {self.environment.num_replicas}")

        # Validate service_rates
        for replica_id in range(self.environment.num_replicas):
            if replica_id not in self.environment.service_rates:
                raise ValueError(f"Missing service_rates for replica {replica_id}")

            for req_type in self.environment.arrival_rates.keys():
                if req_type not in self.environment.service_rates[replica_id]:
                    raise ValueError(
                        f"Missing service rate for replica {replica_id}, request type {req_type}"
                    )

                rate = self.environment.service_rates[replica_id][req_type]
                if rate <= 0:
                    raise ValueError(
                        f"Service rate must be positive, got {rate} for "
                        f"replica {replica_id}, request type {req_type}"
                    )

        # Validate arrival_rates
        for req_type, rate in self.environment.arrival_rates.items():
            if rate <= 0:
                raise ValueError(f"Arrival rate must be positive, got {rate} for type {req_type}")

        # Validate max_time
        if self.environment.max_time <= 0:
            raise ValueError(f"max_time must be positive, got {self.environment.max_time}")

        # Validate scheduler type
        valid_schedulers = ['oracle', 'random', 'round_robin', 'shortest_queue', 'ppo']
        if self.scheduler.type not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler type: {self.scheduler.type}. "
                f"Must be one of {valid_schedulers}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            'experiment': {
                'name': self.experiment.name,
                'description': self.experiment.description,
                'seed': self.experiment.seed,
            },
            'environment': {
                'num_replicas': self.environment.num_replicas,
                'max_time': self.environment.max_time,
                'service_rates': {
                    str(replica_id): {
                        str(req_type): rate
                        for req_type, rate in rates.items()
                    }
                    for replica_id, rates in self.environment.service_rates.items()
                },
                'arrival_rates': {
                    str(req_type): rate
                    for req_type, rate in self.environment.arrival_rates.items()
                },
            },
            'scheduler': {
                'type': self.scheduler.type,
                'options': self.scheduler.options,
            },
            'metrics': {
                'save_csv': self.metrics.save_csv,
                'csv_path': self.metrics.csv_path,
                'save_raw_data': self.metrics.save_raw_data,
            },
            'tensorboard': {
                'enabled': self.tensorboard.enabled,
                'log_dir': self.tensorboard.log_dir,
                'port': self.tensorboard.port,
                'clean_previous_runs': self.tensorboard.clean_previous_runs,
            },
        }

    def save(self, json_path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            json_path: Path to save JSON file
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: str) -> ToyModelConfig:
    """
    Load and validate configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Validated ToyModelConfig instance

    Example:
        >>> config = load_config('configs/toymodel/base.json')
        >>> print(config.experiment.name)
        toymodel_base
    """
    config = ToyModelConfig.from_json(config_path)
    config.validate()
    return config
