"""
PPO (Proximal Policy Optimization) trainer implementation.

This module implements the PPO algorithm for policy optimization
with clipped objective and value function learning.
"""

import math
from typing import Dict, Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.actor_critic import ActorCritic
from .stabilized_ppo_components import (
    gradient_preserving_clip,
    chain_dual_bias_reduction,
    ProductionLoadBalancingReward
)
from .context_aware_entropy import ContextAwareEntropyRegulator


class PPOTrainer:
    """
    PPO trainer with clipped policy objective and value function learning.

    Implements the PPO algorithm with policy clipping, value function clipping,
    and entropy regularization for stable policy optimization.
    """

    def __init__(
        self,
        policy: ActorCritic,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        # NEW: Adaptive entropy scheduling parameters
        entropy_schedule_enable: bool = False,
        entropy_initial: float = 0.02,
        entropy_final: float = 0.0,
        entropy_decay_steps: int = 40000,
        value_coef: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        target_kl: float = 0.01,
        entropy_min: float = 0.5,
        entropy_penalty_coef: float = 0.1,
        entropy_threshold_penalty_enable: bool = False,
        entropy_threshold: float = 1.1,
        entropy_threshold_penalty_coef: float = 0.05,
        kl_coef: float = 0.2,
        # Warm start and KL regularization parameters
        kl_ref_coef_initial: float = 0.5,
        kl_ref_coef_final: float = 0.0,
        kl_ref_decay_steps: int = 1000,
        warmup_steps: int = 500,
        entropy_warmup_coef: float = 0.5,
        # GPPO and CHAIN parameters
        use_gradient_preserving: bool = True,
        use_chain_bias_reduction: bool = True,
        churn_reduction_factor: float = 0.9,
        trust_region_coef: float = 0.01,
        # Advanced stabilization parameters
        clip_range_vf: float = 0.2,
        early_stop_epochs: bool = False,
        min_epochs: int = 2,
        # Intrinsic motivation parameters
        use_intrinsic_motivation: bool = False,
        intrinsic_reward_coef: float = 0.1,
        curiosity_decay: float = 0.999,
        exploration_anneal_steps: int = 500000,
        # Gradient monitoring parameters
        log_gradient_norms: bool = False,
        log_entropy: bool = False,
        log_kl_divergence: bool = False,
        abort_on_nan: bool = False,
        nan_check_frequency: int = 100,
    ):
        """
        Initialize PPO trainer.

        Args:
            policy: Actor-critic policy network
            lr: Learning rate for optimizer
            clip_ratio: PPO clipping ratio for policy objective
            entropy_coef: Coefficient for entropy regularization
            value_coef: Coefficient for value function loss
            epochs: Number of training epochs per update
            minibatch_size: Size of minibatches for training
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for computations
            target_kl: Target KL divergence for early stopping
            entropy_min: Minimum entropy threshold to maintain exploration
            kl_coef: Coefficient for KL regularization loss
        """
        self.policy = policy.to(device)
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        # NEW: Adaptive entropy scheduling
        self.entropy_schedule_enable = entropy_schedule_enable
        self.entropy_initial = entropy_initial
        self.entropy_final = entropy_final
        self.entropy_decay_steps = entropy_decay_steps
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.entropy_min = entropy_min
        self.entropy_penalty_coef = entropy_penalty_coef
        self.entropy_threshold_penalty_enable = entropy_threshold_penalty_enable
        self.entropy_threshold = entropy_threshold
        self.entropy_threshold_penalty_coef = entropy_threshold_penalty_coef
        self.kl_coef = kl_coef

        # Warm start and KL regularization parameters
        self.kl_ref_coef_initial = kl_ref_coef_initial
        self.kl_ref_coef_final = kl_ref_coef_final
        self.kl_ref_decay_steps = kl_ref_decay_steps
        self.warmup_steps = warmup_steps
        self.entropy_warmup_coef = entropy_warmup_coef

        # GPPO and CHAIN parameters
        self.use_gradient_preserving = use_gradient_preserving
        self.use_chain_bias_reduction = use_chain_bias_reduction
        self.churn_reduction_factor = churn_reduction_factor
        self.trust_region_coef = trust_region_coef

        # Advanced stabilization parameters
        self.clip_range_vf = clip_range_vf
        self.early_stop_epochs = early_stop_epochs
        self.min_epochs = min_epochs

        # Intrinsic motivation parameters
        self.use_intrinsic_motivation = use_intrinsic_motivation
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.curiosity_decay = curiosity_decay
        self.exploration_anneal_steps = exploration_anneal_steps

        # Gradient monitoring parameters
        self.log_gradient_norms = log_gradient_norms
        self.log_entropy = log_entropy
        self.log_kl_divergence = log_kl_divergence
        self.abort_on_nan = abort_on_nan
        self.nan_check_frequency = nan_check_frequency

        # Reference policy for KL regularization (will be set during warm start)
        self.reference_policy = None
        self.current_step = 0

        # Initialize production reward calculator
        self.production_reward = ProductionLoadBalancingReward()

        # Initialize context-aware entropy regulator with configurable parameters
        self.context_entropy_regulator = None  # Will be initialized when context parameters are available

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def initialize_context_entropy_regulator(
        self,
        state_dim: int,
        num_actions: int,
        entropy_min: float = 0.01,
        entropy_max: float = 0.5,
        target_entropy_ratio: float = 0.6,
        mode_collapse_threshold: float = 0.7,
        context_sensitivity_threshold: float = 0.1,
        performance_decline_threshold: float = -0.05,
        emergency_boost_factor: float = 2.0,
        gentle_adjustment_rate: float = 0.02,
        intervention_cooldown: int = 50,
        min_samples_for_analysis: int = 100,
        analysis_window: int = 500,
        state_discretization_bins: int = 10  # This will be passed to StateActionAnalyzer
    ) -> None:
        """
        Initialize context-aware entropy regulator with specific parameters.

        Args:
            state_dim: Dimension of state space
            num_actions: Number of possible actions
            entropy_min: Minimum entropy coefficient
            entropy_max: Maximum entropy coefficient
            target_entropy_ratio: Target entropy as fraction of max entropy
            mode_collapse_threshold: Threshold for detecting mode collapse
            context_sensitivity_threshold: Min mutual info for context sensitivity
            performance_decline_threshold: Threshold for performance decline
            emergency_boost_factor: Factor to boost entropy in emergency
            gentle_adjustment_rate: Rate for gentle adjustments
            intervention_cooldown: Steps between interventions
            min_samples_for_analysis: Minimum samples for analysis
            analysis_window: Window size for analysis
            state_discretization_bins: Bins for state discretization
        """
        self.context_entropy_regulator = ContextAwareEntropyRegulator(
            state_dim=state_dim,
            num_actions=num_actions,
            device=self.device,
            entropy_min=entropy_min,
            entropy_max=entropy_max,
            target_entropy_ratio=target_entropy_ratio,
            mode_collapse_threshold=mode_collapse_threshold,
            context_sensitivity_threshold=context_sensitivity_threshold,
            performance_decline_threshold=performance_decline_threshold,
            emergency_boost_factor=emergency_boost_factor,
            gentle_adjustment_rate=gentle_adjustment_rate,
            intervention_cooldown=intervention_cooldown,
            min_samples_for_analysis=min_samples_for_analysis,
            analysis_window=analysis_window
        )

        # Set state_discretization_bins on the StateActionAnalyzer if needed
        if hasattr(self.context_entropy_regulator.state_action_analyzer, 'state_discretization_bins'):
            self.context_entropy_regulator.state_action_analyzer.state_discretization_bins = state_discretization_bins

    def set_reference_policy(self, reference_policy: ActorCritic) -> None:
        """
        Set reference policy for KL regularization.

        Args:
            reference_policy: Reference policy (frozen copy of current policy)
        """
        self.reference_policy = reference_policy
        # Freeze reference policy parameters
        for param in self.reference_policy.parameters():
            param.requires_grad = False
        self.reference_policy.eval()

    def get_current_kl_ref_coef(self) -> float:
        """
        Get current KL reference coefficient based on training step.

        Returns:
            Current KL reference coefficient
        """
        if self.reference_policy is None or self.current_step >= self.kl_ref_decay_steps:
            return 0.0

        # Linear decay from initial to final
        progress = min(1.0, self.current_step / self.kl_ref_decay_steps)
        return self.kl_ref_coef_initial * (1 - progress) + self.kl_ref_coef_final * progress

    def get_current_entropy_coef(self) -> float:
        """
        Get current entropy coefficient using context-aware regulation.

        The context-aware entropy regulator intelligently adjusts the entropy coefficient
        based on state-action relationships, performance trends, and mode collapse detection.
        This replaces the simple scheduling approaches with intelligent adaptive control.

        Returns:
            Current entropy coefficient from context-aware regulator
        """
        return self.context_entropy_regulator.get_entropy_coef()

    def compute_kl_reference_loss(
        self,
        s: torch.Tensor,
        hxs: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss w.r.t. reference policy.

        Args:
            s: States
            hxs: Hidden states
            masks: Masks

        Returns:
            KL divergence loss
        """
        if self.reference_policy is None:
            return torch.tensor(0.0, device=self.device)

        # Get current policy logits
        current_logits, _, _ = self.policy.get_action_logits_values(s, hxs, masks)

        # Get reference policy logits
        with torch.no_grad():
            ref_logits, _, _ = self.reference_policy.get_action_logits_values(s, hxs, masks)

        # Compute KL divergence: KL(current || reference)
        current_probs = torch.softmax(current_logits, dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        current_log_probs = torch.log_softmax(current_logits, dim=-1)

        kl_div = torch.sum(current_probs * (current_log_probs - ref_log_probs), dim=-1)
        return torch.mean(kl_div)

    def update(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        logp_old: torch.Tensor,
        v_old: torch.Tensor,
        ret: torch.Tensor,
        adv: torch.Tensor,
        masks: torch.Tensor,
        hxs_init: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Update policy using PPO objective.

        Args:
            s: States tensor (N, state_dim)
            a: Actions tensor (N,)
            logp_old: Old log probabilities (N,)
            v_old: Old value estimates (N,)
            ret: Returns (N,)
            adv: Advantages (N,)
            masks: Episode masks (N,)
            hxs_init: Initial hidden states (tensor for non-decoupled, tuple for decoupled)

        Returns:
            Dictionary of training statistics
        """
        N = s.shape[0]
        idx = np.arange(N)

        # Training statistics collectors
        pi_losses, vf_losses, entropies = [], [], []
        kls, clipfracs, gradnorms, evs = [], [], [], []

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for i0 in range(0, N, self.minibatch_size):
                j = idx[i0 : i0 + self.minibatch_size]
                bs = s[j]
                ba = a[j].view(-1).to(torch.long)  # Ensure 1D integer
                blogp = logp_old[j]
                bret = ret[j]
                badv = adv[j]
                bm = masks[j]

                # Initialize hidden states for minibatch
                # Simplified: each sample starts from same initial hxs
                with torch.no_grad():
                    if isinstance(hxs_init, tuple):
                        hxs = tuple(h.clone().detach() for h in hxs_init)
                    else:
                        hxs = hxs_init.clone().detach()

                # Forward pass through policy
                new_logp, entropy, v_pred, _ = self.policy.evaluate_actions(
                    bs, hxs, bm.unsqueeze(-1), ba
                )

                # PPO policy loss with gradient-preserving clipping
                ratio = torch.exp(new_logp - blogp)

                if self.use_gradient_preserving:
                    # Use GPPO gradient-preserving clipping
                    clipped_obj = gradient_preserving_clip(ratio, badv, self.clip_ratio)
                    pi_loss = -clipped_obj.mean()
                else:
                    # Traditional PPO clipping
                    surr1 = ratio * badv
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * badv
                    pi_loss = -torch.min(surr1, surr2).mean()

                # Value function loss with clipping
                v_clipped = v_old[j] + (v_pred - v_old[j]).clamp(
                    -self.clip_ratio, self.clip_ratio
                )
                vf_loss1 = (v_pred - bret).pow(2)
                vf_loss2 = (v_clipped - bret).pow(2)
                vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                # KL divergence regularization
                kl_div = torch.mean(blogp - new_logp).clamp(min=0)
                kl_penalty = self.kl_coef * kl_div

                # Update context-aware entropy regulator if initialized
                # Extract rewards from advantages (approximate)
                approx_rewards = badv + v_pred.detach() - bret.detach()  # r_t ≈ A_t + V(s_t) - R_t

                entropy_analysis = self.context_entropy_regulator.update(
                    states=bs,
                    actions=ba,
                    rewards=approx_rewards,
                    value_loss=vf_loss.item(),
                    policy_loss=pi_loss.item(),
                    kl_divergence=kl_div.item(),
                    current_entropy=entropy.item()
                )

                # Use context-aware entropy coefficient
                current_entropy_coef = self.get_current_entropy_coef()
                entropy_bonus = current_entropy_coef * entropy
                # Additional penalty if entropy drops below minimum
                entropy_penalty = 0.0
                if entropy.item() < self.entropy_min:
                    entropy_penalty = self.entropy_penalty_coef * (self.entropy_min - entropy.item())

                # Additional penalty if entropy exceeds threshold (prevents uniform distribution lock-in)
                entropy_threshold_penalty = 0.0
                if self.entropy_threshold_penalty_enable and entropy.item() > self.entropy_threshold:
                    entropy_threshold_penalty = self.entropy_threshold_penalty_coef * (entropy.item() - self.entropy_threshold)

                # KL regularization w.r.t. reference policy
                kl_ref_coef = self.get_current_kl_ref_coef()
                kl_ref_loss = 0.0
                if kl_ref_coef > 0.0:
                    kl_ref_loss = kl_ref_coef * self.compute_kl_reference_loss(bs, hxs, bm)

                # Apply CHAIN dual bias reduction if enabled
                if self.use_chain_bias_reduction:
                    pi_loss, vf_loss = chain_dual_bias_reduction(
                        pi_loss, vf_loss,
                        self.churn_reduction_factor,
                        self.trust_region_coef
                    )

                # Total loss with enhanced regularization
                loss = (pi_loss + self.value_coef * vf_loss + kl_penalty + kl_ref_loss
                       - entropy_bonus + entropy_penalty + entropy_threshold_penalty)

                # Update current step for warm-up and KL decay
                self.current_step += 1


                # Backward pass and optimization
                self.opt.zero_grad()
                loss.backward()

                # Compute gradient norm before clipping
                total_sq = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        g = p.grad.data
                        total_sq += float(torch.sum(g * g))
                grad_norm = float(math.sqrt(total_sq)) if total_sq > 0 else 0.0

                # Gradient clipping and optimization step
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

                # Collect training statistics
                with torch.no_grad():
                    approx_kl = torch.mean(blogp - new_logp).clamp(min=0).item()
                    clipfrac = torch.mean(
                        (torch.abs(ratio - 1.0) > self.clip_ratio).float()
                    ).item()
                    var_y = torch.var(bret, unbiased=False)
                    ev = (1.0 - torch.var(bret - v_pred, unbiased=False) / (var_y + 1e-8)).item()

                pi_losses.append(pi_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(entropy.item())
                kls.append(approx_kl)
                clipfracs.append(clipfrac)
                gradnorms.append(grad_norm)
                evs.append(ev)

        # Return training statistics with context-aware entropy information
        lr = self.opt.param_groups[0]["lr"]

        stats = {
            "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "vf_loss": float(np.mean(vf_losses)) if vf_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "pg_grad_norm": float(np.mean(gradnorms)) if gradnorms else 0.0,
            "explained_var": float(np.mean(evs)) if evs else 0.0,
            "lr": float(lr),
        }

        # Add context-aware entropy regulation stats if available
        entropy_diagnostics = self.context_entropy_regulator.get_diagnostic_info()
        context_stats = {
            "entropy_coef": entropy_diagnostics['current_entropy_coef'],
            "entropy_emergency_mode": entropy_diagnostics['emergency_mode'],
            "entropy_interventions": entropy_diagnostics['intervention_count'],
            "entropy_steps_since_intervention": entropy_diagnostics['steps_since_intervention'],
        }
        print(f"[DEBUG] About to return stats from PPO trainer update method")
        print(f"[DEBUG] Adding context stats to main stats: {context_stats}")
        stats.update(context_stats)
        print(f"[DEBUG] Final stats keys: {list(stats.keys())}")

        return stats