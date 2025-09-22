import json
import os
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from typing import Literal
from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.device_sku_config import BaseDeviceSKUConfig
from vidur.config.flat_dataclass import create_flat_dataclass
from vidur.config.model_config import BaseModelConfig
from vidur.config.node_sku_config import BaseNodeSKUConfig
from vidur.config.utils import dataclass_to_dict
from vidur.logger import init_logger
from vidur.types import (
    ExecutionTimePredictorType,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)
from enum import Enum

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens."},
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace request interval generator file."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00",
        metadata={"help": "Start time of the trace request interval generator."},
    )
    end_time: str = field(
        default="1970-01-04 15:00:00",
        metadata={"help": "End time of the trace request interval generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Time scale factor for the trace request interval generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.5,
        metadata={"help": "Queries per second for Poisson Request Interval Generator."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.2,
        metadata={"help": "Queries per second for Gamma Request Interval Generator."},
    )
    cv: float = field(
        default=0.5,
        metadata={
            "help": "Coefficient of variation for Gamma Request Interval Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace request length generator file."},
    )
    prefill_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6,
        metadata={"help": "Theta for Zipf Request Length Generator."},
    )
    scramble: bool = field(
        default=False,
        metadata={"help": "Scramble for Zipf Request Length Generator."},
    )
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Zipf Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={"help": "Prefill to decode ratio for Zipf Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Uniform Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={
            "help": "Prefill to decode ratio for Uniform Request Length Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=2048,
        metadata={"help": "Prefill tokens for Fixed Request Length Generator."},
    )
    decode_tokens: int = field(
        default=512,
        metadata={"help": "Decode tokens for Fixed Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig,
        metadata={"help": "Length generator config for Synthetic Request Generator."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator config for Synthetic Request Generator."},
    )
    num_requests: Optional[int] = field(
        default=128,
        metadata={"help": "Number of requests for Synthetic Request Generator."},
    )
    duration: Optional[float] = field(
        default=None,
        metadata={"help": "Duration of the synthetic request generator."},
    )

    def __post_init__(self):
        self.max_tokens = self.length_generator_config.max_tokens

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/splitwise_conv.csv",
        metadata={"help": "Path to the trace request generator file."},
    )
    prefill_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Prefill scale factor for the trace request generator."},
    )
    decode_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Decode scale factor for the trace request generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Time scale factor for the trace request generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE_REPLAY


@dataclass
class BaseReplicaSchedulerConfig(BasePolyConfig):
    batch_size_cap: int = field(
        default=128,
        metadata={"help": "Maximum batch size cap."},
    )
    block_size: int = field(
        default=16,
        metadata={"help": "Block size."},
    )
    watermark_blocks_fraction: float = field(
        default=0.01,
        metadata={"help": "Watermark blocks fraction."},
    )
    num_blocks: Optional[int] = field(
        default=None,
        metadata={"help": "Number of blocks."},
    )


@dataclass
class VllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for vLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM


@dataclass
class LightllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for LightLLM."},
    )
    max_waiting_iters: int = field(
        default=10,
        metadata={"help": "Maximum waiting iterations for LightLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.LIGHTLLM


@dataclass
class OrcaSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for Sarathi."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SARATHI


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},
    )
    write_json_trace: bool = field(
        default=False,
        metadata={"help": "Whether to write json trace."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep id."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run id."},
    )
    enable_chrome_trace: bool = field(
        default=True,
        metadata={"help": "Enable Chrome tracing."},
    )
    save_table_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to save table to wandb."},
    )
    store_plots: bool = field(
        default=True,
        metadata={"help": "Whether to store plots."},
    )
    store_operation_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store operation metrics."},
    )
    store_token_completion_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store token completion metrics."},
    )
    store_request_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store request metrics."},
    )
    store_batch_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store batch metrics."},
    )
    store_utilization_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store utilization metrics."},
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to keep individual batch metrics."},
    )
    subsamples: Optional[int] = field(
        default=None,
        metadata={"help": "Subsamples."},
    )
    min_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum batch index."},
    )
    max_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum batch index."},
    )
    output_dir: str = field(
        default="outputs/simulator_output",
        metadata={"help": "Output directory."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},
    )

    def __post_init__(self):
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ReplicaConfig:
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Model name."},
    )
    memory_margin_fraction: float = field(
        default=0.1,
        metadata={"help": "Memory margin fraction."},
    )
    num_pipeline_stages: int = field(
        default=1,
        metadata={"help": "Number of pipeline stages."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},
    )
    device: str = field(
        default="a100",
        metadata={"help": "Device."},
    )
    network_device: str = field(
        default="a100_pairwise_nvlink",
        metadata={"help": "Network device."},
    )

    def __post_init__(self):
        self.world_size = self.num_pipeline_stages * self.tensor_parallel_size
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model_name
        )
        self.device_config: BaseDeviceSKUConfig = (
            BaseDeviceSKUConfig.create_from_type_string(self.device)
        )
        self.node_config: BaseNodeSKUConfig = BaseNodeSKUConfig.create_from_type_string(
            self.network_device
        )


@dataclass
class BaseGlobalSchedulerConfig(BasePolyConfig):
    pass


@dataclass
class RandomGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANDOM


@dataclass
class RoundRobinGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.ROUND_ROBIN


@dataclass
class LORGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOR
@dataclass
class RandomWithStateGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    """随机路由 + 全量 state 采集/可选打印"""
    debug_dump_global_state: bool = field(
        default=False,
        metadata={"help": "Print detailed replica state at each schedule()."},
    )
    max_queue_requests_per_replica: int = field(
        default=4,
        metadata={"help": "How many per-replica queued requests to include in state."},
    )

    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANDOM_WITH_STATE
class RewardMode(str, Enum):
    delta = "delta"
    instant = "instant"
    hybrid = "hybrid"
@dataclass
class DQNGlobalSchedulerOnlineConfig(BaseGlobalSchedulerConfig):
    """在线 DQN 的所有可调超参"""
    # 状态构造
    max_queue_requests_per_replica: int = field(
        default=4,
        metadata={"help": "How many per-replica queued requests to include in state."},
    )
    debug_dump_global_state: bool = field(
        default=False,
        metadata={"help": "Print detailed replica state at each schedule()."},
    )
    # 奖励
    reward_latency_weight: float = field(
        default=1.0,
        metadata={"help": "Penalty weight for latency increase in the reward."},
    )
    balance_penalty_weight: float = field(
    default=0.1,
    metadata={"help": "用于负载平衡惩罚的权重；0禁用。"},
)
    # DQN 超参
    lr: float = field(default=5e-3, metadata={"help": "Learning rate."})
    gamma: float = field(default=0.95, metadata={"help": "Discount factor."})
    epsilon: float = field(default=0.1, metadata={"help": "Epsilon for ε-greedy."})
    epsilon_min: float = field(default=0.01, metadata={"help": "Min epsilon."})
    epsilon_decay: float = field(default=0.995, metadata={"help": "Epsilon decay."})
    buffer_size: int = field(default=10000, metadata={"help": "Replay buffer size."})
    batch_size: int = field(default=32, metadata={"help": "SGD batch size."})
        # ===== 新增：PPO 相关可调超参 =====
    reward_mode: RewardMode = RewardMode.hybrid   # 混合模式：绝对价值 + 平滑差分，避免静止状态零奖励
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15  # Reduced to prevent aggressive updates
    entropy_coef: float = 0.15  # Increased further to combat stagnation and encourage active exploration
    value_coef: float = 0.5
    epochs: int = 8  # Increased for better convergence
    rollout_len: int = 128
    minibatch_size: int = 64
    max_grad_norm: float = 1.0
    hidden_size: int = 128
    layer_N: int = 2
    gru_layers: int = 2

    @staticmethod
    def get_type():
        return GlobalSchedulerType.GLOBALSCHEDULEONLINE
from dataclasses import dataclass
from vidur.types import GlobalSchedulerType

@dataclass
class PPOGlobalSchedulerOnlineConfig(DQNGlobalSchedulerOnlineConfig):
    """
    PPO Online 调度器的配置。为了省事，复用 DQN 的字段和 CLI 参数；
    仅 get_type() 不同，使其在 CLI 上成为一个新的可选类型 'ppoonline'。
    """
     # 状态构造
    max_queue_requests_per_replica: int = field(
        default=4,
        metadata={"help": "How many per-replica queued requests to include in state."},
    )
    debug_dump_global_state: bool = field(
        default=False,
        metadata={"help": "Print detailed replica state at each schedule()."},
    )
    # 奖励
    reward_latency_weight: float = field(
        default=1.0,
        metadata={"help": "Penalty weight for latency increase in the reward."},
    )
    balance_penalty_weight: float = field(
    default=0.1,
    metadata={"help": "用于负载平衡惩罚的权重；0禁用。"},
)
    # DQN 超参
    lr: float = field(default=5e-3, metadata={"help": "Learning rate."})
    gamma: float = field(default=0.95, metadata={"help": "Discount factor."})
    epsilon: float = field(default=0.1, metadata={"help": "Epsilon for ε-greedy."})
    epsilon_min: float = field(default=0.01, metadata={"help": "Min epsilon."})
    epsilon_decay: float = field(default=0.995, metadata={"help": "Epsilon decay."})
    buffer_size: int = field(default=10000, metadata={"help": "Replay buffer size."})
    batch_size: int = field(default=32, metadata={"help": "SGD batch size."})
        # ===== 新增：PPO 相关可调超参 =====
    reward_mode: RewardMode = RewardMode.hybrid   # 混合模式：绝对价值 + 平滑差分，避免静止状态零奖励
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15  # Reduced to prevent aggressive updates
    entropy_coef: float = 0.02  # Increased from 0.005 to 0.02 for better exploration as per improvement plan
    value_coef: float = 0.5  # Keep at 0.5 initially, may reduce to 0.3 if value function dominates
    epochs: int = 8  # Increased for better convergence
    rollout_len: int = 32
    minibatch_size: int = 64
    max_grad_norm: float = 1.0
    # Enhanced exploration and regularization
    target_kl: float = field(
        default=0.01,
        metadata={"help": "Target KL divergence for early stopping to prevent policy collapse."},
    )
    entropy_min: float = field(
        default=0.5,
        metadata={"help": "Minimum entropy threshold to maintain exploration."},
    )
    kl_coef: float = field(
        default=0.2,
        metadata={"help": "Coefficient for KL regularization loss."},
    )

    # Warm start and KL regularization parameters
    enable_warm_start: bool = field(
        default=False,
        metadata={"help": "Enable behavior cloning warm start from demonstration data."},
    )
    demo_data_path: str = field(
        default="",
        metadata={"help": "Path to demonstration data file for warm start."},
    )
    pretrained_actor_path: str = field(
        default="",
        metadata={"help": "Path to pretrained actor model for warm start."},
    )
    stabilization_steps: int = field(
        default=1000,
        metadata={"help": "Number of initial steps during which parameters are not updated (stabilization period)."},
    )
    kl_ref_coef_initial: float = field(
        default=0.5,
        metadata={"help": "Initial KL coefficient for reference policy regularization."},
    )
    kl_ref_coef_final: float = field(
        default=0.0,
        metadata={"help": "Final KL coefficient for reference policy regularization."},
    )
    kl_ref_decay_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps to decay KL reference coefficient."},
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Number of warm-up steps with enhanced exploration."},
    )
    entropy_warmup_coef: float = field(
        default=0.5,
        metadata={"help": "Additional entropy coefficient during warm-up."},
    )

    hidden_size: int = 128
    layer_N: int = 2
    gru_layers: int = 2
    @staticmethod
    def get_type():
        return GlobalSchedulerType.PPOONLINE


@dataclass
class PPOGlobalSchedulerModularConfig(PPOGlobalSchedulerOnlineConfig):
    """
    Modular PPO Global Scheduler configuration.

    Uses the same parameters as PPOGlobalSchedulerOnlineConfig but
    registers as a different scheduler type for the modular implementation.
    Includes additional TensorBoard monitoring capabilities.
    """

    # TensorBoard monitoring configuration
    enable_tensorboard: bool = field(
        default=True,
        metadata={"help": "Enable TensorBoard monitoring for PPO training metrics."},
    )
    tensorboard_log_dir: str = field(
        default="./outputs/runs/ppo_training",
        metadata={"help": "Directory for TensorBoard logs."},
    )
    tensorboard_auto_start: bool = field(
        default=True,
        metadata={"help": "Automatically start TensorBoard server during training."},
    )
    tensorboard_port: int = field(
        default=6006,
        metadata={"help": "Port for TensorBoard server."},
    )
    tensorboard_start_retries: int = field(
        default=3,
        metadata={"help": "Number of retries for TensorBoard server startup."},
    )
    tensorboard_retry_delay: float = field(
        default=5.0,
        metadata={"help": "Delay in seconds between TensorBoard startup retries."},
    )
    tensorboard_force_kill: bool = field(
        default=False,
        metadata={"help": "Force terminate existing TensorBoard process listening on the configured port."},
    )

    # Metrics export configuration
    metrics_export_enabled: bool = field(
        default=False,
        metadata={"help": "Enable metrics export to CSV/Parquet files."},
    )
    metrics_export_format: str = field(
        default="csv",
        metadata={"help": "Export format: 'csv' or 'parquet'."},
    )
    metrics_export_path: str = field(
        default="./outputs/runs/ppo_training/exports",
        metadata={"help": "Directory for exported metrics files."},
    )
    metrics_export_interval: int = field(
        default=50,
        metadata={"help": "Steps between metrics export flushes."},
    )

    # Checkpoint management configuration
    enable_checkpoints: bool = field(
        default=True,
        metadata={"help": "Enable automatic checkpoint saving during training."},
    )
    checkpoint_dir: str = field(
        default="./outputs/checkpoints",
        metadata={"help": "Directory for saving model checkpoints."},
    )
    checkpoint_interval: int = field(
        default=100,
        metadata={"help": "Steps between automatic checkpoint saves."},
    )
    max_checkpoints: int = field(
        default=5,
        metadata={"help": "Maximum number of checkpoints to keep."},
    )
    load_checkpoint: str = field(
        default="",
        metadata={"help": "Path to checkpoint file to load. Empty string means start fresh."},
    )
    inference_only: bool = field(
        default=False,
        metadata={"help": "Run in inference-only mode without training. Requires load_checkpoint."},
    )

    # StateBuilder enhanced features configuration
    enable_enhanced_features: bool = field(
        default=True,
        metadata={"help": "Enable enhanced state features for high-frequency dynamics capture."},
    )
    state_history_window: int = field(
        default=5,
        metadata={"help": "Number of historical steps to track for each replica."},
    )
    qps_window: int = field(
        default=10,
        metadata={"help": "Window size for QPS computation and trend analysis."},
    )

    # Enhanced reward calculation parameters
    latency_threshold: float = field(
        default=6.0,
        metadata={"help": "Soft latency threshold for penalty activation (seconds) - increased to reduce penalty frequency."},
    )
    latency_penalty_scale: float = field(
        default=0.5,
        metadata={"help": "Scale factor for latency threshold penalty - significantly reduced to allow other rewards to matter."},
    )
    load_balance_penalty: float = field(
        default=0.15,
        metadata={"help": "Weight for replica load balance penalty based on queue variance - increased to enforce distribution."},
    )

    # Restructured reward system parameters
    throughput_target: float = field(
        default=0.05,  # Lower target to make current performance more rewarding
        metadata={"help": "Target throughput for normalization in absolute score calculation (requests/second)."},
    )
    absolute_weight: float = field(
        default=0.8,  # Increase absolute weight to amplify positive signals
        metadata={"help": "Weight for absolute score component (w_abs) - primary reward signal."},
    )
    delta_weight: float = field(
        default=0.2,  # Reduce delta weight since it's often zero
        metadata={"help": "Weight for delta score component (w_delta) - improvement signal."},
    )
    alpha: float = field(
        default=0.1,  # Drastically reduce latency penalty to allow positive rewards
        metadata={"help": "Balance factor in absolute score (throughput vs latency weight)."},
    )
    beta: float = field(
        default=0.3,
        metadata={"help": "Weight for normalized throughput delta in delta score."},
    )
    gamma: float = field(
        default=0.2,
        metadata={"help": "Weight for normalized latency delta in delta score."},
    )
    kappa: float = field(
        default=0.05,  # Drastically reduce logistic penalty weight
        metadata={"help": "Weight for logistic latency penalty (smooth replacement for threshold penalty)."},
    )
    sigma: float = field(
        default=2.0,  # Increase smoothness to make penalty more gradual
        metadata={"help": "Scale parameter for logistic penalty smoothness."},
    )
    ema_alpha: float = field(
        default=0.1,
        metadata={"help": "Alpha parameter for exponential moving averages in reward calculation."},
    )

    # Enhanced Actor-Critic architecture parameters
    enable_decoupled_ac: bool = field(
        default=True,
        metadata={"help": "Enable decoupled Actor-Critic architecture for better learning."},
    )
    feature_projection_dim: int = field(
        default=256,
        metadata={"help": "Dimension for feature projection layer (auto-sized if 0)."},
    )

    # Dynamic temperature control parameters
    enable_dynamic_temperature: bool = field(
        default=True,
        metadata={"help": "Enable dynamic temperature scaling for exploration control."},
    )
    base_temperature: float = field(
        default=1.5,
        metadata={"help": "Base temperature for action selection - increased for more exploration."},
    )
    min_temperature: float = field(
        default=0.8,
        metadata={"help": "Minimum allowed temperature - increased to prevent over-exploitation."},
    )
    max_temperature: float = field(
        default=3.0,
        metadata={"help": "Maximum allowed temperature - increased for stronger exploration."},
    )
    qps_sensitivity: float = field(
        default=0.1,
        metadata={"help": "Sensitivity to QPS pressure for temperature adjustment."},
    )
    latency_sensitivity: float = field(
        default=0.2,
        metadata={"help": "Sensitivity to latency pressure for temperature adjustment."},
    )

    # Statistics stabilization configuration
    enable_statistics_stabilization: bool = field(
        default=True,
        metadata={"help": "Enable statistics stabilization phase before PPO training starts."},
    )
    statistics_stabilization_steps: int = field(
        default=100,
        metadata={"help": "Number of random steps to collect statistics before PPO training."},
    )
    stabilization_policy: str = field(
        default="random",
        metadata={"help": "Policy to use during stabilization phase (random, uniform)."},
    )
    collect_baseline_stats: bool = field(
        default=True,
        metadata={"help": "Collect baseline statistics during stabilization phase."},
    )
    freeze_normalizers_during_stabilization: bool = field(
        default=False,
        metadata={"help": "Freeze normalizers during stabilization phase."},
    )
    enable_stabilization_logging: bool = field(
        default=True,
        metadata={"help": "Enable detailed logging during statistics stabilization phase."},
    )
    stabilization_action_distribution: str = field(
        default="uniform",
        metadata={"help": "Action distribution during stabilization (uniform, weighted)."},
    )

    # NEW: Entropy schedule parameters (adaptive entropy for exploration-exploitation balance)
    entropy_schedule_enable: bool = field(
        default=False,
        metadata={"help": "Enable adaptive entropy scheduling during training."},
    )
    entropy_initial: float = field(
        default=0.02,
        metadata={"help": "Initial entropy coefficient for entropy schedule."},
    )
    entropy_final: float = field(
        default=0.0,
        metadata={"help": "Final entropy coefficient for entropy schedule."},
    )
    entropy_decay_steps: int = field(
        default=40000,
        metadata={"help": "Number of steps for entropy decay from initial to final value."},
    )

    # NEW: Curriculum learning parameters (progressive difficulty training)
    enable_curriculum_learning: bool = field(
        default=False,
        metadata={"help": "Enable curriculum learning with progressive difficulty stages."},
    )
    curriculum_stages_json: str = field(
        default="[]",
        metadata={"help": "JSON string containing curriculum learning stages configuration."},
    )
    curriculum_stages_json_base64: str = field(
        default="",
        metadata={"help": "Base64-encoded JSON string for curriculum learning stages (shell-safe)."},
    )

    # NEW: Tail latency monitoring parameters (performance tracking)
    tail_latency_tracking_enable: bool = field(
        default=True,
        metadata={"help": "Enable tail latency monitoring and alerting."},
    )
    tail_latency_alert_threshold_p99: float = field(
        default=5.0,
        metadata={"help": "P99 latency threshold for alerts (seconds)."},
    )
    tail_latency_window_size: int = field(
        default=1000,
        metadata={"help": "Window size for tail latency statistics."},
    )

    # NEW: State builder enhanced features parameters (queue delay awareness)
    enable_queue_delay_features: bool = field(
        default=False,
        metadata={"help": "Enable queue delay features in state builder for enhanced awareness."},
    )
    queue_delay_max_wait_time: float = field(
        default=10.0,
        metadata={"help": "Maximum wait time for queue delay normalization (seconds)."},
    )
    queue_delay_urgency_scale: float = field(
        default=10.0,
        metadata={"help": "Urgency scale factor for queue delay calculations."},
    )
    queue_delay_priority_weight: float = field(
        default=1.0,
        metadata={"help": "Priority weight for queue delay calculations."},
    )

    # NEW: Network architecture parameters (enhanced Actor-Critic configuration)
    enable_cross_replica_attention: bool = field(
        default=True,
        metadata={"help": "Enable cross-replica attention mechanism for enhanced feature comparison."},
    )
    cross_replica_attention_heads: int = field(
        default=4,
        metadata={"help": "Number of attention heads for cross-replica attention."},
    )
    cross_replica_num_replicas: int = field(
        default=4,
        metadata={"help": "Number of replicas for cross-replica attention."},
    )
    actor_hidden_size: int = field(
        default=320,
        metadata={"help": "Hidden size for actor network."},
    )
    actor_gru_layers: int = field(
        default=3,
        metadata={"help": "Number of GRU layers for actor network."},
    )
    critic_hidden_size: int = field(
        default=384,
        metadata={"help": "Hidden size for critic network."},
    )
    critic_gru_layers: int = field(
        default=3,
        metadata={"help": "Number of GRU layers for critic network."},
    )
    enable_temperature_scaling: bool = field(
        default=False,
        metadata={"help": "Enable temperature scaling for action probability distribution."},
    )

    @staticmethod
    def get_type():
        return GlobalSchedulerType.PPO_MODULAR


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    compute_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv",
        metadata={"help": "Path to the compute input file."},
    )
    attention_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
        metadata={"help": "Path to the attention input file."},
    )
    all_reduce_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path to the all reduce input file."},
    )
    send_recv_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path to the send recv input file."},
    )
    cpu_overhead_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv",
        metadata={"help": "Path to the cpu overhead input file."},
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k fold cross validation splits."},
    )
    no_cache: bool = field(
        default=False,
        metadata={"help": "Whether to cache prediction models."},
    )
    kv_cache_prediction_granularity: int = field(
        default=64,
        metadata={"help": "KV cache prediction granularity."},
    )
    prediction_max_prefill_chunk_size: int = field(
        default=4096,
        metadata={"help": "Max prefill chunk size for prediction."},
    )
    prediction_max_batch_size: int = field(
        default=128,
        metadata={"help": "Max batch size for prediction."},
    )
    prediction_max_tokens_per_request: int = field(
        default=4096,
        metadata={"help": "Max tokens per request for prediction."},
    )
    attention_decode_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention decode batching overhead fraction."},
    )
    attention_prefill_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention prefill batching overhead fraction."},
    )
    nccl_cpu_launch_overhead_ms: float = field(
        default=0.02,
        metadata={"help": "NCCL CPU launch overhead in ms."},
    )
    nccl_cpu_skew_overhead_per_device_ms: float = field(
        default=0.0,
        metadata={"help": "NCCL CPU skew overhead per device in ms."},
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads."},
    )
    skip_cpu_overhead_modeling: bool = field(
        default=True,
        metadata={"help": "Whether to skip CPU overhead modeling."},
    )


@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(
        default_factory=lambda: list(range(1, 6)),
        metadata={"help": "Polynomial degree for linear regression."},
    )
    polynomial_include_bias: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial include bias for linear regression."},
    )
    polynomial_interaction_only: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial interaction only for linear regression."},
    )
    fit_intercept: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Fit intercept for linear regression."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION


@dataclass
class RandomForrestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(
        default_factory=lambda: [250, 500, 750],
        metadata={"help": "Number of estimators for random forest."},
    )
    max_depth: List[int] = field(
        default_factory=lambda: [8, 16, 32],
        metadata={"help": "Maximum depth for random forest."},
    )
    min_samples_split: List[int] = field(
        default_factory=lambda: [2, 5, 10],
        metadata={"help": "Minimum samples split for random forest."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FORREST


@dataclass
class ClusterConfig:
    num_replicas: int = field(
        default=1,
        metadata={"help": "Number of replicas."},
    )
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    global_scheduler_config: BaseGlobalSchedulerConfig = field(
        default_factory=RoundRobinGlobalSchedulerConfig,
        metadata={"help": "Global scheduler config."},
    )
    replica_scheduler_config: BaseReplicaSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig,
        metadata={"help": "Replica scheduler config."},
    )


@dataclass
class SimulationConfig(ABC):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},
    )
    time_limit: int = field(
        default=0,  # in seconds, 0 is no limit
        metadata={"help": "Time limit for simulation in seconds. 0 means no limit."},
    )
    cluster_config: ClusterConfig = field(
        default_factory=ClusterConfig,
        metadata={"help": "Cluster config."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "Request generator config."},
    )
    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForrestExecutionTimePredictorConfig,
        metadata={"help": "Execution time predictor config."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics config."},
    )

    def __post_init__(self):
        self.write_config_to_file()

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(f"{self.metrics_config.output_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
