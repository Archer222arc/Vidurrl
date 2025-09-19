"""
Modular PPO-based global scheduler implementation.

This implementation uses modular components from src/rl_components
following the project naming conventions and architecture standards.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from vidur.entities import Replica, Request
from vidur.config import SimulationConfig
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler_with_state import (
    build_global_state,
    debug_dump_replica_state,
)

# Import modular components from src/
from src.rl_components import (
    ActorCritic,
    RunningNormalizer,
    PPOTrainer,
    RewardCalculator,
    RolloutBuffer,
    StateBuilder,
    TensorBoardMonitor,
    PPOTrainingDetector,
    CheckpointManager,
    InferenceMode,
    MetricsExporter,
    TemperatureController,
)

logger = init_logger(__name__)


class PPOGlobalSchedulerModular(BaseGlobalScheduler):
    """
    Modular PPO-based global scheduler with component separation.

    Uses reinforcement learning components from src/rl_components
    for maintainable and testable PPO implementation.
    """
    TYPE = "ppo_modular"

    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]) -> None:
        super().__init__(config, replicas)
        self._replica_ids: List[int] = sorted(list(replicas.keys()))

        # Extract configuration parameters - direct access, no fallback allowed
        gcfg = config.cluster_config.global_scheduler_config

        # PPO hyperparameters - direct access from config
        self._hidden_size    = int(gcfg.hidden_size)
        self._layer_N        = int(gcfg.layer_N)
        self._gru_layers     = int(gcfg.gru_layers)
        self._lr             = float(gcfg.lr)
        self._gamma          = float(gcfg.gamma)
        self._gae_lambda     = float(gcfg.gae_lambda)
        self._clip_ratio     = float(gcfg.clip_ratio)
        self._entropy_coef   = float(gcfg.entropy_coef)
        self._value_coef     = float(gcfg.value_coef)
        self._epochs         = int(gcfg.epochs)
        self._rollout_len    = int(gcfg.rollout_len)
        self._minibatch_size = int(gcfg.minibatch_size)
        self._max_grad_norm  = float(gcfg.max_grad_norm)

        # Reward calculation parameters - direct access
        self._reward_latency_weight   = float(gcfg.reward_latency_weight)
        self._balance_penalty_weight  = float(gcfg.balance_penalty_weight)
        self._max_queue_requests      = int(gcfg.max_queue_requests_per_replica)
        self._debug_dump              = bool(gcfg.debug_dump_global_state)

        # Reward mode configuration - direct access
        mode = gcfg.reward_mode
        if hasattr(mode, "value"):  # Handle Enum types
            mode = mode.value
        self._reward_mode = str(mode).lower()
        # Direct validation - no fallback allowed per CLAUDE.md regulations
        assert self._reward_mode in ("delta", "instant", "hybrid"), f"Invalid reward_mode: {self._reward_mode}"

        self._device = "cpu"

        # Initialize modular components
        self._norm = RunningNormalizer(eps=1e-6, clip=5.0)

        # Enhanced StateBuilder configuration - direct access
        self._enable_enhanced_features = bool(gcfg.enable_enhanced_features)
        self._state_history_window = int(gcfg.state_history_window)
        self._qps_window = int(gcfg.qps_window)

        self._state_builder = StateBuilder(
            max_queue_requests=self._max_queue_requests,
            history_window=self._state_history_window,
            qps_window=self._qps_window,
            enable_enhanced_features=self._enable_enhanced_features
        )

        # Enhanced reward calculation parameters - direct access, no fallback allowed
        self._latency_threshold = float(gcfg.latency_threshold)
        self._latency_penalty_scale = float(gcfg.latency_penalty_scale)
        self._load_balance_penalty = float(gcfg.load_balance_penalty)

        # New restructured reward parameters
        self._throughput_target = float(gcfg.throughput_target)
        self._absolute_weight = float(gcfg.absolute_weight)
        self._delta_weight = float(gcfg.delta_weight)
        self._alpha = float(gcfg.alpha)
        self._beta = float(gcfg.beta)
        self._gamma = float(gcfg.gamma)
        self._kappa = float(gcfg.kappa)
        self._sigma = float(gcfg.sigma)
        self._ema_alpha = float(gcfg.ema_alpha)

        # Enhanced exploration and regularization parameters
        self._target_kl = float(gcfg.target_kl)
        self._entropy_min = float(gcfg.entropy_min)
        self._kl_coef = float(gcfg.kl_coef)

        self._reward_calc = RewardCalculator(
            mode=self._reward_mode,
            latency_weight=self._reward_latency_weight,
            balance_penalty_weight=self._balance_penalty_weight,
            latency_threshold=self._latency_threshold,
            latency_penalty_scale=self._latency_penalty_scale,
            load_balance_penalty=self._load_balance_penalty,
            # New structured reward parameters
            throughput_target=self._throughput_target,
            absolute_weight=self._absolute_weight,
            delta_weight=self._delta_weight,
            alpha=self._alpha,
            beta=self._beta,
            gamma=self._gamma,
            kappa=self._kappa,
            sigma=self._sigma,
            ema_alpha=self._ema_alpha,
        )

        # Verify critical configuration parameters
        logger.info(f"🎯 PPO重构奖励参数 - throughput_target={self._throughput_target:.1f}, "
                   f"abs_weight={self._absolute_weight:.1f}, delta_weight={self._delta_weight:.1f}, "
                   f"alpha={self._alpha:.1f}, kappa={self._kappa:.1f}")

        # Dynamic temperature control - direct access, no fallback allowed
        self._enable_dynamic_temperature = bool(gcfg.enable_dynamic_temperature)
        self._base_temperature = float(gcfg.base_temperature)
        self._min_temperature = float(gcfg.min_temperature)
        self._max_temperature = float(gcfg.max_temperature)
        self._qps_sensitivity = float(gcfg.qps_sensitivity)
        self._latency_sensitivity = float(gcfg.latency_sensitivity)

        self._temperature_controller = TemperatureController(
            base_temperature=self._base_temperature,
            min_temperature=self._min_temperature,
            max_temperature=self._max_temperature,
            qps_sensitivity=self._qps_sensitivity,
            latency_sensitivity=self._latency_sensitivity,
        ) if self._enable_dynamic_temperature else None

        # Determine state and action dimensions
        s0 = self._state_builder.build_global_state(
            self._replicas, self.get_replica_scheduler, 0.0, None
        )
        self._norm.update(s0)
        s0n = self._norm.normalize(s0)
        state_dim = int(s0n.shape[0])
        action_dim = len(self._replica_ids)

        # Enhanced Actor-Critic architecture parameters
        self._enable_decoupled_ac = bool(gcfg.enable_decoupled_ac)
        self._feature_projection_dim = int(gcfg.feature_projection_dim) if gcfg.feature_projection_dim > 0 else None

        # Initialize Enhanced Actor-Critic network
        self._ac = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self._hidden_size,
            layer_N=self._layer_N,
            gru_layers=self._gru_layers,
            use_orthogonal=True,
            enable_decoupled=self._enable_decoupled_ac,
            feature_projection_dim=self._feature_projection_dim,
        ).to(self._device)

        # Initialize PPO trainer
        self._ppo = PPOTrainer(
            self._ac,
            lr=self._lr,
            clip_ratio=self._clip_ratio,
            entropy_coef=self._entropy_coef,
            value_coef=self._value_coef,
            epochs=self._epochs,
            minibatch_size=self._minibatch_size,
            max_grad_norm=self._max_grad_norm,
            device=self._device,
            target_kl=self._target_kl,
            entropy_min=self._entropy_min,
            kl_coef=self._kl_coef,
        )

        # Initialize rollout buffer
        self._buf = RolloutBuffer(
            state_dim=state_dim,
            rollout_len=self._rollout_len,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            device=self._device,
        )

        # GRU hidden state initialization
        # Initialize hidden states for Actor-Critic
        if self._enable_decoupled_ac:
            # Separate hidden states for actor and critic
            hxs_actor = torch.zeros(self._gru_layers, 1, self._hidden_size, device=self._device)
            hxs_critic = torch.zeros(1, 1, self._hidden_size, device=self._device)  # Critic has 1 GRU layer
            self._hxs = (hxs_actor, hxs_critic)
        else:
            # Single shared hidden state
            self._hxs = torch.zeros(self._gru_layers, 1, self._hidden_size, device=self._device)

        # Training state
        self._step: int = 0

        # Normalizer reinitialization flag for inference mode
        self._needs_norm_reinit: bool = False

        # TensorBoard monitoring - direct access
        self._enable_tensorboard = bool(gcfg.enable_tensorboard)
        self._tensorboard_log_dir = str(gcfg.tensorboard_log_dir)
        self._tensorboard_auto_start = bool(gcfg.tensorboard_auto_start)
        self._tensorboard_port = int(gcfg.tensorboard_port)
        self._tensorboard_start_retries = int(gcfg.tensorboard_start_retries)
        self._tensorboard_retry_delay = float(gcfg.tensorboard_retry_delay)
        self._tensorboard_force_kill = bool(gcfg.tensorboard_force_kill)

        self._tb_monitor = TensorBoardMonitor(
            log_dir=self._tensorboard_log_dir,
            enabled=self._enable_tensorboard,
            auto_start=self._tensorboard_auto_start,
            port=self._tensorboard_port,
            start_retries=self._tensorboard_start_retries,
            retry_delay=self._tensorboard_retry_delay,
            force_kill=self._tensorboard_force_kill,
        )
        self._training_detector = PPOTrainingDetector()

        # Metrics export configuration - direct access
        self._metrics_export_enabled = bool(gcfg.metrics_export_enabled)
        self._metrics_export_format = str(gcfg.metrics_export_format)
        self._metrics_export_path = str(gcfg.metrics_export_path)
        self._metrics_export_interval = int(gcfg.metrics_export_interval)

        self._metrics_exporter = MetricsExporter(
            export_path=self._metrics_export_path,
            export_format=self._metrics_export_format,
            flush_interval=self._metrics_export_interval,
            enabled=self._metrics_export_enabled,
        )

        # Checkpoint management - direct access
        self._enable_checkpoints = bool(gcfg.enable_checkpoints)
        self._checkpoint_dir = str(gcfg.checkpoint_dir)
        self._checkpoint_interval = int(gcfg.checkpoint_interval)
        self._max_checkpoints = int(gcfg.max_checkpoints)
        self._load_checkpoint = str(gcfg.load_checkpoint)
        self._inference_only = bool(gcfg.inference_only)

        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=self._checkpoint_dir,
            save_interval=self._checkpoint_interval,
            max_checkpoints=self._max_checkpoints,
            auto_save=self._enable_checkpoints and not self._inference_only,
        )

        # Handle checkpoint loading and inference mode
        if self._load_checkpoint:
            self._load_from_checkpoint()
        elif self._inference_only:
            raise ValueError("inference_only=True requires load_checkpoint to be specified")

        # Set correct training/inference mode for Actor-Critic
        if self._inference_only:
            self._ac.eval()
        else:
            self._ac.train()

        # Log hyperparameters
        if self._tb_monitor.is_active():
            hparams = {
                "hidden_size": self._hidden_size,
                "layer_N": self._layer_N,
                "gru_layers": self._gru_layers,
                "lr": self._lr,
                "gamma": self._gamma,
                "gae_lambda": self._gae_lambda,
                "clip_ratio": self._clip_ratio,
                "entropy_coef": self._entropy_coef,
                "value_coef": self._value_coef,
                "epochs": self._epochs,
                "rollout_len": self._rollout_len,
                "minibatch_size": self._minibatch_size,
                "max_grad_norm": self._max_grad_norm,
                "reward_latency_weight": self._reward_latency_weight,
                "balance_penalty_weight": self._balance_penalty_weight,
                "reward_mode": self._reward_mode,
            }
            self._tb_monitor.log_hyperparameters(hparams)

        logger.info(
            "[PPO:init] mode=%s | h=%d L=%d GRU=%d | lr=%.2e gamma=%.3f lam=%.3f "
            "clip=%.2f ent=%.3f vf=%.3f epochs=%d roll=%d mb=%d gnorm=%.2f | lat_w=%.3f bal_w=%.3f",
            self._reward_mode, self._hidden_size, self._layer_N, self._gru_layers,
            self._lr, self._gamma, self._gae_lambda, self._clip_ratio,
            self._entropy_coef, self._value_coef, self._epochs, self._rollout_len,
            self._minibatch_size, self._max_grad_norm, self._reward_latency_weight, self._balance_penalty_weight
        )
        logger.info(
            "[PPO:init] state_dim=%d sample(min/mean/max)=%.4f/%.4f/%.4f",
            s0.shape[0], float(np.min(s0)), float(np.mean(s0)), float(np.max(s0))
        )

    def _load_from_checkpoint(self) -> None:
        """Load model and training state from checkpoint."""
        try:
            checkpoint_data = self._checkpoint_manager.load_checkpoint(self._load_checkpoint)

            # In inference mode, rebuild components from checkpoint config to ensure compatibility
            if self._inference_only:
                logger.info("[PPO:checkpoint] Loading complete configuration from checkpoint for inference compatibility")

                metadata = checkpoint_data.get("metadata", {})

                # Load and apply StateBuilder config
                state_builder_config = metadata.get("state_builder_config")
                if state_builder_config:
                    logger.info("[PPO:checkpoint] Applying StateBuilder config from checkpoint")
                    from src.rl_components.state_builder import StateBuilder
                    self._state_builder = StateBuilder(
                        max_queue_requests=state_builder_config["max_queue_requests"],
                        history_window=state_builder_config["history_window"],
                        qps_window=state_builder_config["qps_window"],
                        enable_enhanced_features=state_builder_config["enable_enhanced_features"]
                    )
                    # Update internal config variables
                    self._max_queue_requests = state_builder_config["max_queue_requests"]
                    self._state_history_window = state_builder_config["history_window"]
                    self._qps_window = state_builder_config["qps_window"]
                    self._enable_enhanced_features = state_builder_config["enable_enhanced_features"]
                    self._needs_norm_reinit = True
                    logger.info("[PPO:checkpoint] StateBuilder config loaded: %s", state_builder_config)

                # Load and apply reward config
                reward_config = metadata.get("reward_config")
                if reward_config:
                    logger.info("[PPO:checkpoint] Applying reward config from checkpoint")
                    self._latency_threshold = reward_config.get("latency_threshold", self._latency_threshold)
                    self._latency_penalty_scale = reward_config.get("latency_penalty_scale", self._latency_penalty_scale)
                    self._load_balance_penalty = reward_config.get("load_balance_penalty", self._load_balance_penalty)
                    self._throughput_target = reward_config.get("throughput_target", self._throughput_target)
                    self._absolute_weight = reward_config.get("absolute_weight", self._absolute_weight)
                    self._delta_weight = reward_config.get("delta_weight", self._delta_weight)
                    self._alpha = reward_config.get("alpha", self._alpha)
                    self._kappa = reward_config.get("kappa", self._kappa)
                    # Recreate reward calculator with loaded config
                    self._reward_calc = RewardCalculator(
                        mode=self._reward_mode,
                        latency_weight=self._reward_latency_weight,
                        balance_penalty_weight=self._balance_penalty_weight,
                        latency_threshold=self._latency_threshold,
                        latency_penalty_scale=self._latency_penalty_scale,
                        load_balance_penalty=self._load_balance_penalty,
                        throughput_target=self._throughput_target,
                        absolute_weight=self._absolute_weight,
                        delta_weight=self._delta_weight,
                        alpha=self._alpha,
                        kappa=self._kappa,
                        sigma=self._sigma,
                        ema_alpha=self._ema_alpha,
                    )

                # Load and apply temperature config
                temperature_config = metadata.get("temperature_config")
                if temperature_config:
                    logger.info("[PPO:checkpoint] Applying temperature config from checkpoint")
                    self._enable_dynamic_temperature = temperature_config.get("enable_dynamic_temperature", self._enable_dynamic_temperature)
                    self._base_temperature = temperature_config.get("base_temperature", self._base_temperature)
                    self._min_temperature = temperature_config.get("min_temperature", self._min_temperature)
                    self._max_temperature = temperature_config.get("max_temperature", self._max_temperature)
                    self._qps_sensitivity = temperature_config.get("qps_sensitivity", self._qps_sensitivity)
                    self._latency_sensitivity = temperature_config.get("latency_sensitivity", self._latency_sensitivity)
                    # Recreate temperature controller with loaded config
                    if self._enable_dynamic_temperature:
                        self._temperature_controller = TemperatureController(
                            base_temperature=self._base_temperature,
                            min_temperature=self._min_temperature,
                            max_temperature=self._max_temperature,
                            qps_sensitivity=self._qps_sensitivity,
                            latency_sensitivity=self._latency_sensitivity,
                        )
                    else:
                        self._temperature_controller = None

                # Rebuild Actor-Critic from checkpoint config
                self._ac = self._checkpoint_manager.create_actor_critic_from_checkpoint(
                    checkpoint_data, device=self._device
                )

                if not state_builder_config:
                    logger.warning("[PPO:checkpoint] No StateBuilder config found in checkpoint, using existing configuration")
                    self._needs_norm_reinit = False
                    # Load normalizer state for inference mode with existing StateBuilder
                    normalizer_state = checkpoint_data["normalizer_state"]
                    self._norm.eps = normalizer_state["eps"]
                    self._norm.clip = normalizer_state["clip"]
                    self._norm.count = normalizer_state["count"]
                    if normalizer_state["mean"] is not None:
                        self._norm.mean = np.array(normalizer_state["mean"])
                    if normalizer_state["m2"] is not None:
                        self._norm.m2 = np.array(normalizer_state["m2"])
            else:
                # Load model state into existing network (training mode)
                self._ac.load_state_dict(checkpoint_data["model_state_dict"])

                # Load normalizer state for training mode
                normalizer_state = checkpoint_data["normalizer_state"]
                self._norm.eps = normalizer_state["eps"]
                self._norm.clip = normalizer_state["clip"]
                self._norm.count = normalizer_state["count"]
                if normalizer_state["mean"] is not None:
                    self._norm.mean = np.array(normalizer_state["mean"])
                if normalizer_state["m2"] is not None:
                    self._norm.m2 = np.array(normalizer_state["m2"])

            # Load training state
            training_state = checkpoint_data.get("training_state", {})
            self._step = training_state.get("step", 0)

            # Reset reward calculator state if loading
            self._reward_calc.reset_state()

            # Ensure correct training/inference mode after loading
            if self._inference_only:
                self._ac.eval()
            else:
                self._ac.train()

            logger.info(
                "[PPO:checkpoint] Loaded from step %d, inference_only=%s",
                checkpoint_data["step"], self._inference_only
            )

        except Exception as e:
            logger.error("[PPO:checkpoint] Failed to load checkpoint: %s", e)
            if self._inference_only:
                raise RuntimeError("Cannot start inference mode without valid checkpoint")
            else:
                logger.warning("[PPO:checkpoint] Continuing with fresh training state")

    def schedule(self) -> List[Tuple[int, Request]]:
        """
        Schedule requests using PPO-based decision making.

        Returns:
            List of (replica_id, request) tuples for scheduling
        """
        if not self._request_queue:
            return []

        if self._debug_dump:
            for rid in self._replica_ids:
                rs = self.get_replica_scheduler(rid)
                debug_dump_replica_state(rid, rs)

        # Handle normalizer reinitialization for inference mode after StateBuilder rebuild
        if self._needs_norm_reinit:
            logger.info("[PPO:schedule] Reinitializing normalizer to match rebuilt StateBuilder")
            from src.rl_components.normalizers import RunningNormalizer
            self._norm = RunningNormalizer()
            self._needs_norm_reinit = False

        # 1) Build and normalize state
        s_np = self._state_builder.build_global_state(
            self._replicas,
            self.get_replica_scheduler,
            self._current_time,
            self._metric_store,
        )
        self._norm.update(s_np)
        s_norm = self._norm.normalize(s_np)
        s = torch.from_numpy(s_norm).float().unsqueeze(0).to(self._device)  # (1, D)

        if self._debug_dump:
            self._debug_state_details(s_np)

        # 2) Calculate reward using modular reward calculator with validity check
        # First check if we should skip this update due to invalid metrics
        raw_throughput = float(self._metric_store.get_throughput(self._current_time))
        raw_latency = float(self._metric_store.get_average_latency())

        if not self._reward_calc.is_valid_update(raw_throughput, raw_latency):
            # Skip this step - use previous reward and don't update training
            if hasattr(self, '_last_reward'):
                r = self._last_reward
                reward_info = self._last_reward_info.copy()
                reward_info["skipped_invalid_metrics"] = True
                reward_info["raw_throughput"] = raw_throughput
                reward_info["raw_latency"] = raw_latency
            else:
                # Very first step - use neutral reward
                r = 0.0
                reward_info = {"skipped_invalid_metrics": True, "first_step": True}
        else:
            # Normal reward calculation
            r, reward_info = self._reward_calc.calculate_reward(
                self._metric_store,
                self._current_time,
                self._replica_ids,
                self.get_replica_scheduler,
            )
            # Store for potential reuse if next step has invalid metrics
            self._last_reward = r
            self._last_reward_info = reward_info

        # Log reward metrics to TensorBoard
        if self._tb_monitor.is_active():
            self._tb_monitor.log_reward_metrics(reward_info, r, step=self._step)

        # 3) Determine episode continuation mask
        idle = 1 if (len(self._request_queue) == 0) else 0
        mask = torch.tensor([1.0 - float(idle)], dtype=torch.float32, device=self._device).unsqueeze(-1)

        # 4) Forward pass: sample action and get value
        if self._inference_only:
            # Inference mode: use stochastic sampling like training for consistency
            with torch.inference_mode():
                # Compute dynamic temperature even in inference for consistency
                temperature = 1.0
                if self._temperature_controller:
                    # Calculate current metrics for temperature control
                    current_latency = reward_info["latency"] if reward_info and "latency" in reward_info else self._latency_threshold
                    current_qps = self._metric_store.get_throughput(self._current_time) if self._metric_store else 2.0
                    target_qps = current_qps * 1.1
                    target_latency = self._latency_threshold

                    # Calculate load balance from current replica utilizations
                    utilizations = []
                    for replica_id in self._replica_ids:
                        scheduler = self.get_replica_scheduler(replica_id)
                        num_alloc = scheduler._num_allocated_blocks
                        num_blocks = scheduler._config.num_blocks
                        utilizations.append(num_alloc / num_blocks)

                    system_load_balance = 1.0 - (max(utilizations) - min(utilizations)) if utilizations else 1.0

                    # Use delta values from reward_info if available
                    delta_throughput = reward_info.get("delta_throughput", 0.0) if reward_info else 0.0
                    delta_latency = reward_info.get("delta_latency", 0.0) if reward_info else 0.0

                    temperature = self._temperature_controller.compute_temperature(
                        current_qps=current_qps,
                        target_qps=target_qps,
                        current_latency=current_latency,
                        target_latency=target_latency,
                        system_load_balance=system_load_balance,
                        delta_throughput=delta_throughput,
                        delta_latency=delta_latency,
                    )

                # Use the same stochastic sampling as training
                a, _, _, self._hxs = self._ac.act_value(s, self._hxs, mask, temperature=temperature)
                a_i = int(a.item())
                rid = self._replica_ids[a_i]

                # Skip all training-related operations in inference mode
                if self._debug_dump:
                    logger.info("[PPO:inference] step=%d act=%d rid=%d temp=%.3f", self._step, a_i, rid, temperature)

        else:
            # Training mode: full PPO pipeline (gradients enabled)

            # Update buffer with previous step's reward
            if self._buf.ptr > 0 and self._buf.ptr <= self._rollout_len:
                self._buf.r[-1] = float(r)
                self._buf.masks[-1] = float(mask.item())

            # Compute dynamic temperature for exploration control
            temperature = 1.0
            if self._temperature_controller and not self._inference_only:
                # Get real metrics from reward_info (no fallback allowed)
                current_latency = reward_info["latency"]  # Direct access - fail if missing
                current_qps = self._metric_store.get_throughput(self._current_time)
                target_qps = current_qps * 1.1  # Target 10% improvement
                target_latency = self._latency_threshold

                # Calculate load balance from current replica utilizations
                utilizations = []
                for replica_id in self._replica_ids:
                    scheduler = self.get_replica_scheduler(replica_id)
                    num_alloc = scheduler._num_allocated_blocks
                    num_blocks = scheduler._config.num_blocks
                    utilizations.append(num_alloc / num_blocks)

                # Load balance score: 1.0 for perfect balance, lower for imbalance
                system_load_balance = 1.0 - (max(utilizations) - min(utilizations))

                # Get delta values for stagnation detection
                delta_throughput = reward_info.get("delta_throughput", 0.0)
                delta_latency = reward_info.get("delta_latency", 0.0)

                temperature = self._temperature_controller.compute_temperature(
                    current_qps=current_qps,
                    target_qps=target_qps,
                    current_latency=current_latency,
                    target_latency=target_latency,
                    system_load_balance=system_load_balance,
                    delta_throughput=delta_throughput,
                    delta_latency=delta_latency,
                )

            # Sample action and get value (with dynamic temperature for training)
            a, logp, v, self._hxs = self._ac.act_value(s, self._hxs, mask, temperature)
            a_i = int(a.item())
            rid = self._replica_ids[a_i]

            # Store experience in buffer
            self._buf.add_step(
                s.squeeze(0).detach(),           # state: (D,) - detached for storage
                a.detach(),                      # action
                logp.detach(),                   # log probability
                v.detach(),                      # value estimate
                0.0,                             # reward placeholder
                mask.squeeze(-1).detach(),       # mask: (1,)
            )

            # Trigger PPO update when buffer is full (gradients enabled)
            if self._buf.is_full():
                self._perform_ppo_update(s, mask)

        # 5) Schedule request to selected replica
        self.sort_requests()
        req = self._request_queue.pop(0)
        mapping = [(rid, req)]

        # 6) Debug and progress logging
        if self._debug_dump:
            self._log_step_info(a_i, reward_info, r)

        if (self._step % 10) == 0:
            self._log_progress()

        # 7) Update TensorBoard and monitoring (only in training mode)
        if not self._inference_only:
            # Export metrics data
            if self._metrics_exporter.enabled:
                # Gather current metrics for export
                system_metrics = {
                    "queue_length": len(self._request_queue),
                    "selected_replica": rid,
                    "action_value": a_i,
                    "reward": float(r),
                    "value_estimate": float(v.item()),
                    "log_probability": float(logp.item()),
                    "temperature": temperature,
                }

                # Preserve the true reward value before merging reward_info
                true_reward = system_metrics["reward"]

                # Add reward breakdown if available
                if isinstance(reward_info, dict):
                    system_metrics.update(reward_info)

                # Restore the true reward value (prevent overwriting by total_reward from reward_info)
                system_metrics["reward"] = true_reward

                # Add detailed reward breakdown from reward calculator
                reward_breakdown = self._reward_calc.get_reward_breakdown()
                for key, value in reward_breakdown.items():
                    # Avoid overwriting the true reward field
                    if key != "total_reward" or f"reward_{key}" not in system_metrics:
                        system_metrics[f"reward_{key}"] = value

                # Add temperature controller metrics if available
                if self._temperature_controller:
                    temp_metrics = self._temperature_controller.get_pressure_metrics()
                    for key, value in temp_metrics.items():
                        system_metrics[f"temp_{key}"] = value

                self._metrics_exporter.append_training_metrics(
                    step=self._step,
                    metrics=system_metrics,
                    metadata={
                        "replica_id": rid,
                        "action": a_i,
                        "queue_size": len(self._request_queue),
                        "has_temperature_control": self._temperature_controller is not None,
                    }
                )

                # Log system metrics
                system_metrics = {
                    "QueueLength": len(self._request_queue),
                    "BufferProgress": self._buf.ptr / self._rollout_len,
                    "SelectedReplica": a_i,
                    "EpisodeMask": float(mask.item()),
                }
                self._tb_monitor.log_system_metrics(system_metrics, step=self._step)
                self._tb_monitor.flush()

            # Update training activity detector
            self._training_detector.update_training_activity(is_training_update=False)

        self._step += 1
        return mapping

    def _perform_ppo_update(self, s: torch.Tensor, mask: torch.Tensor) -> None:
        """Perform PPO update when rollout buffer is full."""
        with torch.no_grad():
            # Bootstrap value using current state - use act_value for proper decoupled handling
            _, _, last_v, _ = self._ac.act_value(s, self._hxs, mask)
            last_v = last_v.detach()

        # Compute GAE advantages and returns
        s_t, a_t, logp_t, v_t, ret_t, adv_t = self._buf.compute_gae(last_v)
        m_tensor = torch.tensor(self._buf.masks, dtype=torch.float32, device=self._device)

        # Update policy using PPO
        # Handle detach for both single tensor and tuple hidden states
        if isinstance(self._hxs, tuple):
            hxs_detached = tuple(h.detach() for h in self._hxs)
        else:
            hxs_detached = self._hxs.detach()

        stats = self._ppo.update(
            s_t, a_t, logp_t, v_t, ret_t, adv_t,
            masks=m_tensor,
            hxs_init=hxs_detached,
        )

        # Log training statistics
        self._log_training_stats(stats, ret_t, adv_t, m_tensor, a_t)

        # Update training detector and compute rollout statistics for monitoring/export
        self._training_detector.update_training_activity(is_training_update=True)

        # Compute rollout statistics (needed for both TensorBoard and metrics export)
        rollout_stats = {}
        act_hist = []

        if self._tb_monitor.is_active() or self._metrics_exporter.enabled:
            with torch.no_grad():
                r_hist = np.asarray(self._buf.r, dtype=np.float32)
                rollout_stats = {
                    "reward_mean": float(np.mean(r_hist)),
                    "reward_std": float(np.std(r_hist)),
                    "reward_min": float(np.min(r_hist)),
                    "reward_max": float(np.max(r_hist)),
                    "value_mean": float(v_t.mean().item()),
                    "return_mean": float(ret_t.mean().item()),
                    "advantage_std": float(adv_t.std(unbiased=False).item()),
                }

                # Action distribution
                a_np = a_t.view(-1).to(torch.int64).cpu().numpy()
                act_hist = np.bincount(a_np, minlength=len(self._replica_ids)).tolist()

        # Log to TensorBoard if active
        if self._tb_monitor.is_active():
            self._tb_monitor.log_training_metrics(stats, step=self._step)
            self._tb_monitor.log_rollout_metrics(
                rollout_stats,
                act_hist,
                len(self._buf.r),
                self._rollout_len,
                step=self._step,
            )

        # Export metrics data for offline analysis
        if self._metrics_exporter.enabled:
            # Export PPO training metrics
            self._metrics_exporter.append_training_metrics(
                step=self._step,
                metrics=stats,
                metadata={
                    "data_type": "ppo_update",
                    "rollout_length": self._rollout_len,
                    "buffer_size": len(self._buf.r),
                }
            )

            # Export rollout statistics
            buffer_progress = len(self._buf.r) / self._rollout_len if self._rollout_len > 0 else 0.0
            self._metrics_exporter.append_rollout_metrics(
                step=self._step,
                rollout_stats=rollout_stats,
                action_distribution=act_hist,
                buffer_progress=buffer_progress
            )

        # Save checkpoint if needed
        if self._checkpoint_manager.should_save(self._step):
            training_state = {
                "step": self._step,
                "rollouts_completed": self._step // self._rollout_len,
            }

            metadata = {
                "reward_mode": self._reward_mode,
                "total_steps": self._step,
                "last_update_stats": stats,
                "state_builder_config": {
                    "max_queue_requests": self._max_queue_requests,
                    "history_window": self._state_history_window,
                    "qps_window": self._qps_window,
                    "enable_enhanced_features": self._enable_enhanced_features,
                },
                "ppo_config": {
                    "lr": self._lr,
                    "gamma": self._gamma,
                    "gae_lambda": self._gae_lambda,
                    "clip_ratio": self._clip_ratio,
                    "entropy_coef": self._entropy_coef,
                    "value_coef": self._value_coef,
                    "epochs": self._epochs,
                    "rollout_len": self._rollout_len,
                    "minibatch_size": self._minibatch_size,
                    "max_grad_norm": self._max_grad_norm,
                    "target_kl": self._target_kl,
                    "entropy_min": self._entropy_min,
                    "kl_coef": self._kl_coef,
                },
                "reward_config": {
                    "reward_latency_weight": self._reward_latency_weight,
                    "balance_penalty_weight": self._balance_penalty_weight,
                    "latency_threshold": self._latency_threshold,
                    "latency_penalty_scale": self._latency_penalty_scale,
                    "load_balance_penalty": self._load_balance_penalty,
                    "throughput_target": self._throughput_target,
                    "absolute_weight": self._absolute_weight,
                    "delta_weight": self._delta_weight,
                    "alpha": self._alpha,
                    "beta": self._beta,
                    "gamma": self._gamma,
                    "kappa": self._kappa,
                    "sigma": self._sigma,
                    "ema_alpha": self._ema_alpha,
                },
                "temperature_config": {
                    "enable_dynamic_temperature": self._enable_dynamic_temperature,
                    "base_temperature": self._base_temperature,
                    "min_temperature": self._min_temperature,
                    "max_temperature": self._max_temperature,
                    "qps_sensitivity": self._qps_sensitivity,
                    "latency_sensitivity": self._latency_sensitivity,
                },
                "network_config": {
                    "hidden_size": self._hidden_size,
                    "layer_N": self._layer_N,
                    "gru_layers": self._gru_layers,
                    "enable_decoupled_ac": self._enable_decoupled_ac,
                    "feature_projection_dim": self._feature_projection_dim,
                },
            }

            self._checkpoint_manager.save_checkpoint(
                step=self._step,
                actor_critic=self._ac,
                normalizer=self._norm,
                training_state=training_state,
                metadata=metadata,
            )

        # Reset buffer for next rollout
        self._buf.reset(state_dim=s.shape[-1])

    def _debug_state_details(self, s_np: np.ndarray) -> None:
        """Debug logging for state details."""
        try:
            RAW = s_np
            BASE_LEN = 11
            REQ_FEATS = 7
            K = int(self._max_queue_requests)
            stride = BASE_LEN + K * REQ_FEATS

            for rid_idx, rid in enumerate(self._replica_ids):
                base_start = rid_idx * stride
                base_slice = RAW[base_start : base_start + BASE_LEN]
                logger.info(
                    "[STATE:rid=%d] base(%d)=%s",
                    rid, BASE_LEN,
                    np.array2string(base_slice, formatter={"float_kind": lambda x: f"{x: .3e}"})
                )

                # Log first 2 requests in queue
                rs = self.get_replica_scheduler(rid)
                rq = getattr(rs, "_request_queue", [])
                for j, req in enumerate(rq[: min(2, K)]):
                    off = base_start + BASE_LEN + j * REQ_FEATS
                    req_slice = RAW[off : off + REQ_FEATS]
                    prio_actual = float(getattr(req, "priority", 0.0) or 0.0)
                    logger.info(
                        "[STATE:rid=%d req#%d id=%s] "
                        "[age,prefill,processed,prefill_done,completed,decode,priority]=%s | priority(req)=%.3f",
                        rid, j, getattr(req, "id", None),
                        np.array2string(req_slice, formatter={"float_kind": lambda x: f"{x: .3e}"}),
                        prio_actual,
                    )
        except Exception:
            pass

    def _log_step_info(self, action: int, reward_info: Dict, reward: float) -> None:
        """Log step information for debugging."""
        if reward_info["mode"] == "delta":
            logger.info(
                "[PPO:step=%d] mode=delta act=%d d_tp=%.5f d_lat=%.5f bal=%.5f reward=%.6f",
                self._step, action, reward_info.get("delta_throughput", 0.0),
                reward_info.get("delta_latency", 0.0), reward_info["balance_penalty"], reward
            )
        else:
            logger.info(
                "[PPO:step=%d] mode=instant act=%d tp=%.5f lat=%.5f bal=%.5f reward=%.6f",
                self._step, action, reward_info["throughput"],
                reward_info["latency"], reward_info["balance_penalty"], reward
            )

    def _log_training_stats(
        self,
        stats: Dict[str, float],
        ret_t: torch.Tensor,
        adv_t: torch.Tensor,
        m_tensor: torch.Tensor,
        a_t: torch.Tensor
    ) -> None:
        """Log detailed training statistics."""
        with torch.no_grad():
            r_hist = np.asarray(self._buf.r, dtype=np.float32)
            T = r_hist.shape[0]
            R_mean, R_std = float(np.mean(r_hist)), float(np.std(r_hist))
            R_min, R_max = float(np.min(r_hist)), float(np.max(r_hist))
            Adv_std = float(adv_t.std(unbiased=False).item())
            V_mean = float(ret_t.mean().item() - adv_t.mean().item())  # v_t approximation
            Ret_mean = float(ret_t.mean().item())
            mask_mean = float(m_tensor.mean().item())

            # Action distribution
            a_np = a_t.view(-1).to(torch.int64).cpu().numpy()
            act_hist = np.bincount(a_np, minlength=len(self._replica_ids)).tolist()

        logger.info(
            "[PPO:update] step=%d len=%d "
            "pi=%.6f vf=%.6f ent=%.4f kl=%.5f clip=%.3f ev=%.3f gnorm=%.3f lr=%.2e "
            "R(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f Adv_std=%.4f V/Ret=%.4f/%.4f mask=%.3f act=%s",
            self._step, T,
            stats["pi_loss"], stats["vf_loss"], stats["entropy"],
            stats["approx_kl"], stats["clipfrac"], stats["explained_var"],
            stats["pg_grad_norm"], stats["lr"],
            R_mean, R_std, R_min, R_max, Adv_std, V_mean, Ret_mean, mask_mean,
            act_hist,
        )

    def _log_progress(self) -> None:
        """Log rollout progress."""
        try:
            last_r = float(self._buf.r[-1]) if len(self._buf.r) > 0 else float("nan")
            logger.info(
                "[PPO:rollout] step=%d buf_ptr=%d/%d last_r=%.6f",
                self._step, self._buf.ptr, self._rollout_len, last_r
            )
        except Exception:
            pass


# Note: Scheduler registration is handled in global_scheduler_registry.py
# Both PPOONLINE and PPO_MODULAR point to this modular implementation
