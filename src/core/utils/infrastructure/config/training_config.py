#!/usr/bin/env python3
"""
Training configuration builder for PPO warmstart training.
Converts JSON config to command line arguments.
"""

import json
import sys
from typing import Dict, List


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def build_ppo_args(config: Dict, output_dir: str) -> List[str]:
    """Build PPO command line arguments from config."""
    args = []

    # Basic configuration
    args.extend([
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", str(config["training"]["num_replicas"]),
        "--synthetic_request_generator_config_num_requests", str(config["training"]["ppo_requests"]),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(config["training"]["qps"]),
        "--metrics_config_subsamples", str(config["monitoring"]["metrics_subsamples"])
    ])

    # PPO configuration
    ppo_cfg = config["ppo_config"]
    ppo_prefix = "--p_p_o_global_scheduler_modular_config_"
    args.extend([
        f"{ppo_prefix}lr", str(ppo_cfg["lr"]),
        f"{ppo_prefix}gamma", str(ppo_cfg["gamma"]),
        f"{ppo_prefix}gae_lambda", str(ppo_cfg["gae_lambda"]),
        f"{ppo_prefix}clip_ratio", str(ppo_cfg["clip_ratio"]),
        f"{ppo_prefix}epochs", str(ppo_cfg["epochs"]),
        f"{ppo_prefix}rollout_len", str(ppo_cfg["rollout_len"]),
        f"{ppo_prefix}minibatch_size", str(ppo_cfg["minibatch_size"]),
        f"{ppo_prefix}entropy_coef", str(ppo_cfg["entropy_coef"]),
        f"{ppo_prefix}entropy_warmup_coef", str(ppo_cfg["entropy_warmup_coef"]),
        f"{ppo_prefix}stabilization_steps", str(ppo_cfg["stabilization_steps"])
    ])

    # Entropy schedule configuration (NEW)
    if "entropy_schedule" in ppo_cfg and ppo_cfg["entropy_schedule"]["enable"]:
        entropy_sched = ppo_cfg["entropy_schedule"]
        args.extend([
            f"{ppo_prefix}entropy_schedule_enable",
            f"{ppo_prefix}entropy_initial", str(entropy_sched["initial"]),
            f"{ppo_prefix}entropy_final", str(entropy_sched["final"]),
            f"{ppo_prefix}entropy_decay_steps", str(entropy_sched["decay_steps"])
        ])

    # Curriculum learning configuration (NEW)
    if "curriculum_learning" in config and config["curriculum_learning"]["enable"]:
        curriculum_cfg = config["curriculum_learning"]
        args.append(f"{ppo_prefix}enable_curriculum_learning")

        # Pass curriculum stages as base64-encoded JSON to avoid shell parsing issues
        import json
        import base64
        stages_json = json.dumps(curriculum_cfg["stages"])
        # Encode JSON as base64 to avoid all shell special characters
        encoded_stages = base64.b64encode(stages_json.encode('utf-8')).decode('ascii')
        args.extend([
            f"{ppo_prefix}curriculum_stages_json_base64", encoded_stages
        ])

    # Tail latency monitoring configuration (NEW)
    if "monitoring" in config and "tail_latency_tracking" in config["monitoring"]:
        if config["monitoring"]["tail_latency_tracking"]["enable"]:
            tail_cfg = config["monitoring"]["tail_latency_tracking"]
            args.extend([
                f"{ppo_prefix}tail_latency_tracking_enable",
                f"{ppo_prefix}tail_latency_alert_threshold_p99", str(tail_cfg["alert_threshold_p99"]),
                f"{ppo_prefix}tail_latency_window_size", str(tail_cfg["window_size"])
            ])

    # State builder enhanced features configuration (NEW)
    if "state_builder_enhanced" in config:
        state_builder_cfg = config["state_builder_enhanced"]
        if state_builder_cfg.get("enable_queue_delay_features", False):
            args.append(f"{ppo_prefix}enable_queue_delay_features")
            args.extend([
                f"{ppo_prefix}queue_delay_max_wait_time", str(state_builder_cfg["queue_delay_normalization"]["max_wait_time_seconds"]),
                f"{ppo_prefix}queue_delay_urgency_scale", str(state_builder_cfg["queue_delay_normalization"]["urgency_scale"]),
                f"{ppo_prefix}queue_delay_priority_weight", str(state_builder_cfg["queue_delay_normalization"]["priority_weight"])
            ])

    # Network architecture configuration (NEW)
    if "actor_critic_architecture" in config:
        network_cfg = config["actor_critic_architecture"]

        # Cross-replica attention configuration
        if network_cfg.get("enable_cross_replica_attention", False):
            cross_replica_cfg = network_cfg.get("cross_replica_attention", {})
            args.extend([
                f"{ppo_prefix}enable_cross_replica_attention",
                f"{ppo_prefix}cross_replica_attention_heads", str(network_cfg.get("attention_heads", 4)),
                f"{ppo_prefix}cross_replica_num_replicas", str(network_cfg.get("num_replicas", 4))
            ])

        # Actor architecture
        if "actor" in network_cfg:
            actor_cfg = network_cfg["actor"]
            if "hidden_size" in actor_cfg:
                args.extend([f"{ppo_prefix}actor_hidden_size", str(actor_cfg["hidden_size"])])
            if "gru_layers" in actor_cfg:
                args.extend([f"{ppo_prefix}actor_gru_layers", str(actor_cfg["gru_layers"])])
            if "temperature_scaling" in actor_cfg:
                if actor_cfg["temperature_scaling"]:
                    args.append(f"{ppo_prefix}enable_temperature_scaling")

        # Critic architecture
        if "critic" in network_cfg:
            critic_cfg = network_cfg["critic"]
            if "hidden_size" in critic_cfg:
                args.extend([f"{ppo_prefix}critic_hidden_size", str(critic_cfg["hidden_size"])])
            if "gru_layers" in critic_cfg:
                args.extend([f"{ppo_prefix}critic_gru_layers", str(critic_cfg["gru_layers"])])

        # Temporal LSTM configuration
        if "temporal_lstm" in network_cfg:
            temporal_cfg = network_cfg["temporal_lstm"]
            if temporal_cfg.get("enable", False):
                args.append(f"{ppo_prefix}enable_temporal_lstm")
                args.extend([
                    f"{ppo_prefix}temporal_lstm_feature_chunks", str(temporal_cfg.get("feature_chunks", 4)),
                    f"{ppo_prefix}temporal_lstm_hidden_ratio", str(temporal_cfg.get("hidden_size_ratio", 0.25))
                ])
                # Bidirectional is True by default, only add flag if it's True (which it is)
                if temporal_cfg.get("bidirectional", True):
                    args.append(f"{ppo_prefix}temporal_lstm_bidirectional")

    # Reward configuration
    reward_cfg = config["reward_config"]
    args.extend([
        f"{ppo_prefix}reward_latency_weight", str(reward_cfg["latency_weight"]),
        f"{ppo_prefix}balance_penalty_weight", str(reward_cfg["balance_penalty_weight"]),
        f"{ppo_prefix}latency_threshold", str(reward_cfg["latency_threshold"]),
        f"{ppo_prefix}latency_penalty_scale", str(reward_cfg["latency_penalty_scale"]),
        f"{ppo_prefix}load_balance_penalty", str(reward_cfg["load_balance_penalty"]),
        f"{ppo_prefix}throughput_target", str(reward_cfg["throughput_target"]),
        f"{ppo_prefix}absolute_weight", str(reward_cfg["absolute_weight"]),
        f"{ppo_prefix}delta_weight", str(reward_cfg["delta_weight"]),
        f"{ppo_prefix}alpha", str(reward_cfg["alpha"]),
        f"{ppo_prefix}beta", str(reward_cfg.get("beta", 0.4)),
        f"{ppo_prefix}gamma", str(reward_cfg.get("gamma", 0.3)),
        f"{ppo_prefix}kappa", str(reward_cfg["kappa"]),
        f"{ppo_prefix}sigma", str(reward_cfg["sigma"]),
        f"{ppo_prefix}ema_alpha", str(reward_cfg.get("ema_alpha", 0.1))
    ])

    # KL regularization
    kl_cfg = config["kl_regularization"]
    args.extend([
        f"{ppo_prefix}target_kl", str(kl_cfg["target_kl"]),
        f"{ppo_prefix}entropy_min", str(kl_cfg["entropy_min"]),
        f"{ppo_prefix}entropy_penalty_coef", str(kl_cfg["entropy_penalty_coef"]),
        f"{ppo_prefix}kl_coef", str(kl_cfg["kl_coef"]),
        f"{ppo_prefix}kl_ref_coef_initial", str(kl_cfg["kl_ref_coef_initial"]),
        f"{ppo_prefix}kl_ref_coef_final", str(kl_cfg["kl_ref_coef_final"]),
        f"{ppo_prefix}kl_ref_decay_steps", str(kl_cfg["kl_ref_decay_steps"]),
        f"{ppo_prefix}warmup_steps", str(kl_cfg["warmup_steps"])
    ])

    # Entropy threshold penalty (NEW)
    if "entropy_threshold_penalty" in kl_cfg and kl_cfg["entropy_threshold_penalty"]["enable"]:
        entropy_threshold_cfg = kl_cfg["entropy_threshold_penalty"]
        args.extend([
            f"{ppo_prefix}entropy_threshold_penalty_enable",
            f"{ppo_prefix}entropy_threshold", str(entropy_threshold_cfg["threshold"]),
            f"{ppo_prefix}entropy_threshold_penalty_coef", str(entropy_threshold_cfg["penalty_coef"])
        ])

    # Temperature control
    temp_cfg = config["temperature_control"]
    args.extend([
        f"{ppo_prefix}base_temperature", str(temp_cfg["base_temperature"]),
        f"{ppo_prefix}min_temperature", str(temp_cfg["min_temperature"]),
        f"{ppo_prefix}max_temperature", str(temp_cfg["max_temperature"]),
        f"{ppo_prefix}qps_sensitivity", str(temp_cfg["qps_sensitivity"]),
        f"{ppo_prefix}latency_sensitivity", str(temp_cfg["latency_sensitivity"])
    ])

    # Temperature pulse control (NEW)
    if "temperature_pulse" in temp_cfg:
        if temp_cfg["temperature_pulse"]["enable"]:
            # Temperature pulse is enabled by default, no need to pass extra args
            pass
        else:
            args.append(f"{ppo_prefix}disable_temperature_pulse")

    # Enhanced features
    if config["enhanced_features"]["enable_enhanced_features"]:
        enhanced_cfg = config["enhanced_features"]
        args.extend([
            f"{ppo_prefix}enable_enhanced_features",
            f"{ppo_prefix}state_history_window", str(enhanced_cfg["state_history_window"]),
            f"{ppo_prefix}qps_window", str(enhanced_cfg["qps_window"])
        ])

    # Dynamic temperature
    if temp_cfg["enable_dynamic_temperature"]:
        args.append(f"{ppo_prefix}enable_dynamic_temperature")

    # Checkpointing
    if config["checkpointing"]["enable_checkpoints"]:
        ckpt_cfg = config["checkpointing"]
        args.extend([
            f"{ppo_prefix}enable_checkpoints",
            f"{ppo_prefix}checkpoint_dir", ckpt_cfg["checkpoint_dir"],
            f"{ppo_prefix}checkpoint_interval", str(ckpt_cfg["checkpoint_interval"]),
            f"{ppo_prefix}max_checkpoints", str(ckpt_cfg["max_checkpoints"])
        ])

        # Advanced checkpoint options (currently not supported by vidur.main)
        # Note: save_optimizer_state and incremental_checkpoints are not available in vidur.main
        # if ckpt_cfg.get("save_optimizer_state", False):
        #     args.append(f"{ppo_prefix}save_optimizer_state")
        # if ckpt_cfg.get("incremental_checkpoints", False):
        #     args.append(f"{ppo_prefix}incremental_checkpoints")

    # Monitoring
    monitoring_cfg = config["monitoring"]
    args.extend([
        f"{ppo_prefix}tensorboard_log_dir", f"{output_dir}/tensorboard",
        f"{ppo_prefix}tensorboard_force_kill",
        f"{ppo_prefix}tensorboard_port", str(monitoring_cfg["tensorboard_port"]),
        f"{ppo_prefix}tensorboard_auto_start",
        f"{ppo_prefix}metrics_export_enabled",
        f"{ppo_prefix}metrics_export_format", monitoring_cfg["metrics_export_format"],
        f"{ppo_prefix}metrics_export_path", f"{output_dir}/metrics",
        f"{ppo_prefix}metrics_export_interval", str(monitoring_cfg["metrics_export_interval"])
    ])

    return args


def main():
    """CLI entry point for config builder."""
    if len(sys.argv) != 3:
        print("Usage: python training_config.py <config_path> <output_dir>")
        sys.exit(1)

    config_path = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        config = load_config(config_path)
        args = build_ppo_args(config, output_dir)
        print(" ".join(args))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
