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

    # Revolutionary PPO features (GPPO + CHAIN)
    if "revolutionary_features" in ppo_cfg:
        rev_cfg = ppo_cfg["revolutionary_features"]
        if rev_cfg.get("use_gradient_preserving", False):
            args.append(f"{ppo_prefix}use_gradient_preserving")
        if rev_cfg.get("use_chain_bias_reduction", False):
            args.append(f"{ppo_prefix}use_chain_bias_reduction")
        args.extend([
            f"{ppo_prefix}churn_reduction_factor", str(rev_cfg.get("churn_reduction_factor", 0.9)),
            f"{ppo_prefix}trust_region_coef", str(rev_cfg.get("trust_region_coef", 0.01))
        ])

    # Additional PPO stabilization parameters
    if "clip_range_vf" in ppo_cfg:
        args.extend([f"{ppo_prefix}clip_range_vf", str(ppo_cfg["clip_range_vf"])])
    if "max_grad_norm" in ppo_cfg:
        args.extend([f"{ppo_prefix}max_grad_norm", str(ppo_cfg["max_grad_norm"])])
    if "value_coef" in ppo_cfg:
        args.extend([f"{ppo_prefix}value_coef", str(ppo_cfg["value_coef"])])

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

    # Network architecture configuration (NEW + Revolutionary)
    if "actor_critic_architecture" in config:
        network_cfg = config["actor_critic_architecture"]

        # Revolutionary stabilization features
        if "stabilization" in network_cfg:
            stab_cfg = network_cfg["stabilization"]
            if stab_cfg.get("enable_layer_normalization", False):
                args.append(f"{ppo_prefix}enable_layer_normalization")
            if stab_cfg.get("enable_hyperspherical_normalization", False):
                args.append(f"{ppo_prefix}enable_hyperspherical_normalization")
            args.extend([
                f"{ppo_prefix}input_normalization_momentum", str(stab_cfg.get("input_normalization_momentum", 0.99)),
                f"{ppo_prefix}observation_clip", str(stab_cfg.get("observation_clip", 10.0)),
                f"{ppo_prefix}reward_clip", str(stab_cfg.get("reward_clip", 10.0)),
                f"{ppo_prefix}running_mean_std_momentum", str(stab_cfg.get("running_mean_std_momentum", 0.99)),
                f"{ppo_prefix}norm_epsilon", str(stab_cfg.get("norm_epsilon", 1e-8))
            ])

        # Stabilized GRU configuration
        if network_cfg.get("use_stabilized_gru", False):
            args.append(f"{ppo_prefix}use_stabilized_gru")

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

    # Reward configuration with asymmetric penalties
    reward_cfg = config["reward_config"]
    args.extend([
        f"{ppo_prefix}reward_mode", reward_cfg.get("mode", "hybrid"),
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
        f"{ppo_prefix}reward_gamma", str(reward_cfg.get("reward_gamma", 0.3)),
        f"{ppo_prefix}kappa", str(reward_cfg["kappa"]),
        f"{ppo_prefix}sigma", str(reward_cfg["sigma"]),
        f"{ppo_prefix}ema_alpha", str(reward_cfg.get("ema_alpha", 0.1))
    ])

    # Revolutionary asymmetric penalty features
    if "asymmetric_penalties" in reward_cfg:
        asym_cfg = reward_cfg["asymmetric_penalties"]
        if asym_cfg.get("use_asymmetric_penalties", False):
            args.append(f"{ppo_prefix}use_asymmetric_penalties")
        args.extend([
            f"{ppo_prefix}false_positive_penalty", str(asym_cfg.get("false_positive_penalty", 5.0)),
            f"{ppo_prefix}over_provision_factor", str(asym_cfg.get("over_provision_factor", 0.1))
        ])
        if asym_cfg.get("beta_exploration_enable", False):
            args.append(f"{ppo_prefix}beta_exploration_enable")
        if asym_cfg.get("temporal_tracking_enable", False):
            args.append(f"{ppo_prefix}temporal_tracking_enable")

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

    # Advanced stabilization features
    if "advanced_stabilization" in config:
        adv_cfg = config["advanced_stabilization"]

        # Early stopping configuration
        if "early_stopping" in adv_cfg:
            early_cfg = adv_cfg["early_stopping"]
            if early_cfg.get("early_stop_epochs", False):
                args.append(f"{ppo_prefix}early_stop_epochs")
            args.extend([
                f"{ppo_prefix}min_epochs", str(early_cfg.get("min_epochs", 2))
            ])

        # Intrinsic motivation
        if "intrinsic_motivation" in adv_cfg:
            intrinsic_cfg = adv_cfg["intrinsic_motivation"]
            if intrinsic_cfg.get("use_intrinsic_motivation", False):
                args.append(f"{ppo_prefix}use_intrinsic_motivation")
            args.extend([
                f"{ppo_prefix}intrinsic_reward_coef", str(intrinsic_cfg.get("intrinsic_reward_coef", 0.1)),
                f"{ppo_prefix}curiosity_decay", str(intrinsic_cfg.get("curiosity_decay", 0.999)),
                f"{ppo_prefix}exploration_anneal_steps", str(intrinsic_cfg.get("exploration_anneal_steps", 500000))
            ])

        # Gradient monitoring
        if "gradient_monitoring" in adv_cfg:
            grad_cfg = adv_cfg["gradient_monitoring"]
            if grad_cfg.get("log_gradient_norms", False):
                args.append(f"{ppo_prefix}log_gradient_norms")
            if grad_cfg.get("log_entropy", False):
                args.append(f"{ppo_prefix}log_entropy")
            if grad_cfg.get("log_kl_divergence", False):
                args.append(f"{ppo_prefix}log_kl_divergence")
            if grad_cfg.get("abort_on_nan", False):
                args.append(f"{ppo_prefix}abort_on_nan")
            args.extend([
                f"{ppo_prefix}nan_check_frequency", str(grad_cfg.get("nan_check_frequency", 100))
            ])

    # Enhanced features
    if config["enhanced_features"]["enable_enhanced_features"]:
        enhanced_cfg = config["enhanced_features"]
        args.extend([
            f"{ppo_prefix}enable_enhanced_features",
            f"{ppo_prefix}state_history_window", str(enhanced_cfg["state_history_window"]),
            f"{ppo_prefix}qps_window", str(enhanced_cfg["qps_window"])
        ])

    # Action Balance Reward (Revolutionary Feature)
    if "cluster_config" in config and "global_scheduler_config" in config["cluster_config"]:
        scheduler_cfg = config["cluster_config"]["global_scheduler_config"]

        if scheduler_cfg.get("action_balance_enable", False):
            args.extend([
                f"{ppo_prefix}action_balance_enable",
                f"{ppo_prefix}action_balance_weight", str(scheduler_cfg.get("action_balance_weight", 0.1)),
                f"{ppo_prefix}action_balance_window", str(scheduler_cfg.get("action_balance_window", 50))
            ])

    # Context-aware entropy regulation (Revolutionary Feature)
    if "cluster_config" in config and "global_scheduler_config" in config["cluster_config"]:
        scheduler_cfg = config["cluster_config"]["global_scheduler_config"]

        if scheduler_cfg.get("context_aware_entropy_enable", False):
            args.extend([
                f"{ppo_prefix}context_aware_entropy_enable",
                f"{ppo_prefix}context_entropy_min", str(scheduler_cfg.get("context_entropy_min", 0.01)),
                f"{ppo_prefix}context_entropy_max", str(scheduler_cfg.get("context_entropy_max", 0.5)),
                f"{ppo_prefix}context_target_entropy_ratio", str(scheduler_cfg.get("context_target_entropy_ratio", 0.6)),
                f"{ppo_prefix}context_mode_collapse_threshold", str(scheduler_cfg.get("context_mode_collapse_threshold", 0.7)),
                f"{ppo_prefix}context_min_action_freq_threshold", str(scheduler_cfg.get("context_min_action_freq_threshold", 0.08)),
                f"{ppo_prefix}context_sensitivity_threshold", str(scheduler_cfg.get("context_sensitivity_threshold", 0.1)),
                f"{ppo_prefix}context_performance_decline_threshold", str(scheduler_cfg.get("context_performance_decline_threshold", -0.05)),
                f"{ppo_prefix}context_emergency_boost_factor", str(scheduler_cfg.get("context_emergency_boost_factor", 2.0)),
                f"{ppo_prefix}context_gentle_adjustment_rate", str(scheduler_cfg.get("context_gentle_adjustment_rate", 0.02)),
                f"{ppo_prefix}context_intervention_cooldown", str(scheduler_cfg.get("context_intervention_cooldown", 50)),
                f"{ppo_prefix}context_min_samples_for_analysis", str(scheduler_cfg.get("context_min_samples_for_analysis", 100)),
                f"{ppo_prefix}context_analysis_window", str(scheduler_cfg.get("context_analysis_window", 500)),
                f"{ppo_prefix}context_state_discretization_bins", str(scheduler_cfg.get("context_state_discretization_bins", 10))
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
