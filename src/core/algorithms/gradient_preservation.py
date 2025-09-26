"""
Advanced Gradient Preservation and Recovery Mechanisms for PPO.

This module implements cutting-edge gradient preservation techniques to prevent
the gradient vanishing that leads to entropy collapse (entropy = 0.0000).

Key techniques implemented:
1. Adaptive gradient scaling with warmup
2. Spectral normalization for weight matrices
3. Emergency gradient injection during collapse
4. Gradient flow monitoring and restoration
5. Dynamic learning rate adjustment based on gradient health

Based on the observation that entropy went to 0.0000 for extended periods,
indicating complete gradient flow breakdown.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class AdaptiveGradientScaler:
    """
    Adaptive gradient scaling to maintain healthy gradient flow.

    Prevents both gradient vanishing (entropy = 0) and explosion that
    can lead to policy collapse in load balancing tasks.
    """

    def __init__(
        self,
        min_gradient_norm: float = 1e-6,
        max_gradient_norm: float = 10.0,
        target_gradient_norm: float = 1.0,
        adaptation_rate: float = 0.01,
        warmup_steps: int = 1000,
        emergency_boost_factor: float = 100.0,
    ):
        """
        Initialize adaptive gradient scaler.

        Args:
            min_gradient_norm: Minimum gradient norm (emergency intervention below this)
            max_gradient_norm: Maximum gradient norm for clipping
            target_gradient_norm: Target gradient norm for scaling
            adaptation_rate: Rate of adaptation to target norm
            warmup_steps: Steps for gradient warmup
            emergency_boost_factor: Boost factor during gradient emergency
        """
        self.min_gradient_norm = min_gradient_norm
        self.max_gradient_norm = max_gradient_norm
        self.target_gradient_norm = target_gradient_norm
        self.adaptation_rate = adaptation_rate
        self.warmup_steps = warmup_steps
        self.emergency_boost_factor = emergency_boost_factor

        self.current_scale = 1.0
        self.step_count = 0
        self.gradient_norm_history = []
        self.emergency_interventions = 0

    def scale_gradients(self, model: nn.Module, force_emergency: bool = False) -> Dict[str, float]:
        """
        Scale gradients adaptively to maintain healthy flow.

        Args:
            model: The model whose gradients to scale
            force_emergency: Force emergency gradient boost

        Returns:
            Dictionary with scaling statistics
        """
        self.step_count += 1

        # Calculate current gradient norm
        total_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        total_norm = total_norm ** 0.5 if param_count > 0 else 0.0
        self.gradient_norm_history.append(total_norm)

        # Keep only recent history
        if len(self.gradient_norm_history) > 100:
            self.gradient_norm_history = self.gradient_norm_history[-100:]

        # Determine scaling strategy
        scaling_strategy = "normal"
        scale_factor = 1.0

        # Emergency intervention for vanishing gradients
        if force_emergency or (total_norm < self.min_gradient_norm and self.step_count > 50):
            scale_factor = self.emergency_boost_factor
            scaling_strategy = "emergency"
            self.emergency_interventions += 1
            logger.warning(f"[GradientScaler] Emergency gradient boost: {total_norm:.2e} -> "
                          f"{total_norm * scale_factor:.2e} (intervention #{self.emergency_interventions})")

        # Warmup phase: gradually increase gradients
        elif self.step_count < self.warmup_steps:
            warmup_progress = self.step_count / self.warmup_steps
            scale_factor = 0.1 + 0.9 * warmup_progress  # Scale from 0.1 to 1.0
            scaling_strategy = "warmup"

        # Adaptive scaling to target norm
        elif total_norm > 0:
            target_scale = self.target_gradient_norm / (total_norm + 1e-8)
            # Smooth adaptation
            scale_factor = (1 - self.adaptation_rate) * self.current_scale + \
                          self.adaptation_rate * target_scale
            scale_factor = np.clip(scale_factor, 0.1, 10.0)  # Reasonable bounds
            scaling_strategy = "adaptive"

        # Apply scaling
        if scale_factor != 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale_factor)

        self.current_scale = scale_factor
        scaled_norm = total_norm * scale_factor

        # Apply final clipping to prevent explosion
        if scaled_norm > self.max_gradient_norm:
            clip_factor = self.max_gradient_norm / (scaled_norm + 1e-8)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_factor)
            scaled_norm = self.max_gradient_norm
            scaling_strategy = f"{scaling_strategy}_clipped"

        return {
            'original_norm': total_norm,
            'scaled_norm': scaled_norm,
            'scale_factor': scale_factor,
            'strategy': scaling_strategy,
            'emergency_count': self.emergency_interventions,
            'step': self.step_count
        }


class SpectralNormalization(nn.Module):
    """
    Spectral normalization for weight matrices to prevent gradient issues.

    Maintains Lipschitz continuity of the network to ensure stable gradients.
    """

    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        """
        Initialize spectral normalization.

        Args:
            module: Module to apply spectral normalization to
            name: Name of the weight parameter
            n_power_iterations: Number of power iterations for spectral norm estimation
        """
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations

        if not hasattr(module, name):
            raise ValueError(f"Module does not have parameter '{name}'")

        weight = getattr(module, name)
        height = weight.size(0)
        width = weight.view(height, -1).size(1)

        u = nn.Parameter(torch.randn(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(torch.randn(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)

        self.register_parameter(name + "_u", u)
        self.register_parameter(name + "_v", v)
        self.register_parameter(name + "_orig", nn.Parameter(weight.data))

        del module._parameters[name]

    def _update_u_v(self):
        """Update u and v vectors using power iteration."""
        u = getattr(self, self.name + "_u")
        v = getattr(self, self.name + "_v")
        w = getattr(self, self.name + "_orig")

        height = w.size(0)
        w_mat = w.view(height, -1)

        for _ in range(self.n_power_iterations):
            v.data = F.normalize(torch.mv(w_mat.t(), u), dim=0, eps=1e-12)
            u.data = F.normalize(torch.mv(w_mat, v), dim=0, eps=1e-12)

    def forward(self, *args, **kwargs):
        """Forward pass with spectral normalization."""
        if self.training:
            self._update_u_v()

        u = getattr(self, self.name + "_u")
        v = getattr(self, self.name + "_v")
        w = getattr(self, self.name + "_orig")

        height = w.size(0)
        w_mat = w.view(height, -1)
        sigma = torch.dot(u, torch.mv(w_mat, v))

        # Apply spectral normalization
        w_norm = w / sigma

        setattr(self.module, self.name, w_norm)
        return self.module(*args, **kwargs)


class GradientFlowMonitor:
    """
    Monitor gradient flow through the network and detect flow issues.

    Tracks gradient statistics at different layers to identify bottlenecks.
    """

    def __init__(
        self,
        monitor_frequency: int = 20,
        flow_threshold: float = 1e-7,
        layer_names: Optional[list[str]] = None
    ):
        """
        Initialize gradient flow monitor.

        Args:
            monitor_frequency: Frequency of monitoring (every N steps)
            flow_threshold: Threshold below which gradients are considered too small
            layer_names: Names of layers to monitor (None = all layers)
        """
        self.monitor_frequency = monitor_frequency
        self.flow_threshold = flow_threshold
        self.layer_names = layer_names or []

        self.step_count = 0
        self.flow_statistics = {}
        self.bottleneck_layers = []

    def monitor_flow(self, model: nn.Module) -> Dict[str, Any]:
        """
        Monitor gradient flow through the model.

        Args:
            model: Model to monitor

        Returns:
            Dictionary with flow statistics
        """
        self.step_count += 1

        if self.step_count % self.monitor_frequency != 0:
            return {}

        flow_stats = {}
        bottlenecks = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item()
                relative_grad = grad_norm / (param_norm + 1e-8)

                flow_stats[name] = {
                    'grad_norm': grad_norm,
                    'param_norm': param_norm,
                    'relative_grad': relative_grad
                }

                # Check for bottlenecks
                if grad_norm < self.flow_threshold:
                    bottlenecks.append(name)

        self.flow_statistics = flow_stats
        self.bottleneck_layers = bottlenecks

        # Log bottlenecks
        if bottlenecks:
            logger.warning(f"[GradientFlow] Bottleneck layers detected: {bottlenecks}")

        return {
            'bottleneck_count': len(bottlenecks),
            'bottleneck_layers': bottlenecks,
            'total_layers': len(flow_stats),
            'step': self.step_count
        }

    def get_flow_health_score(self) -> float:
        """
        Calculate overall gradient flow health score.

        Returns:
            Score between 0 (unhealthy) and 1 (healthy)
        """
        if not self.flow_statistics:
            return 1.0  # No data = assume healthy

        total_layers = len(self.flow_statistics)
        healthy_layers = total_layers - len(self.bottleneck_layers)

        return healthy_layers / total_layers if total_layers > 0 else 1.0


class EmergencyGradientInjection:
    """
    Emergency gradient injection system for severe collapse cases.

    Injects artificial gradients when natural gradients vanish completely.
    """

    def __init__(
        self,
        injection_strength: float = 0.01,
        noise_scale: float = 0.001,
        target_layers: Optional[list[str]] = None
    ):
        """
        Initialize emergency gradient injection.

        Args:
            injection_strength: Strength of injected gradients
            noise_scale: Scale of gradient noise injection
            target_layers: Specific layers to target (None = all layers)
        """
        self.injection_strength = injection_strength
        self.noise_scale = noise_scale
        self.target_layers = target_layers or []

        self.injection_count = 0

    def inject_gradients(self, model: nn.Module, force_injection: bool = False) -> Dict[str, int]:
        """
        Inject emergency gradients into the model.

        Args:
            model: Model to inject gradients into
            force_injection: Force injection regardless of gradient health

        Returns:
            Dictionary with injection statistics
        """
        injected_params = 0
        total_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            total_params += 1

            # Check if this parameter needs injection
            needs_injection = force_injection
            if param.grad is not None and not force_injection:
                grad_norm = param.grad.data.norm(2).item()
                needs_injection = grad_norm < 1e-8  # Very small gradients

            if needs_injection:
                # Create synthetic gradient
                synthetic_grad = torch.randn_like(param.data) * self.noise_scale

                # Add directional component towards zero (regularization-like)
                synthetic_grad += -param.data * self.injection_strength

                # Apply the synthetic gradient
                if param.grad is None:
                    param.grad = synthetic_grad
                else:
                    param.grad.data += synthetic_grad

                injected_params += 1

        if injected_params > 0:
            self.injection_count += 1
            logger.warning(f"[EmergencyGradient] Injected gradients into {injected_params}/{total_params} "
                          f"parameters (injection #{self.injection_count})")

        return {
            'injected_params': injected_params,
            'total_params': total_params,
            'injection_count': self.injection_count
        }


class DynamicLearningRateScheduler:
    """
    Dynamic learning rate adjustment based on gradient health.

    Adjusts learning rate to maintain optimal gradient magnitudes.
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        adaptation_rate: float = 0.1,
        gradient_target: float = 1e-3,
        emergency_lr_boost: float = 10.0
    ):
        """
        Initialize dynamic learning rate scheduler.

        Args:
            base_lr: Base learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            adaptation_rate: Rate of learning rate adaptation
            gradient_target: Target gradient magnitude
            emergency_lr_boost: LR boost factor during emergencies
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adaptation_rate = adaptation_rate
        self.gradient_target = gradient_target
        self.emergency_lr_boost = emergency_lr_boost

        self.current_lr = base_lr
        self.emergency_count = 0

    def adjust_lr(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_norm: float,
        emergency: bool = False
    ) -> Dict[str, float]:
        """
        Adjust learning rate based on gradient health.

        Args:
            optimizer: Optimizer to adjust
            gradient_norm: Current gradient norm
            emergency: Whether this is an emergency adjustment

        Returns:
            Dictionary with adjustment statistics
        """
        old_lr = self.current_lr

        if emergency:
            # Emergency boost
            new_lr = min(self.max_lr, self.current_lr * self.emergency_lr_boost)
            self.emergency_count += 1
            adjustment_type = "emergency"
        else:
            # Normal adaptation
            if gradient_norm > 0:
                target_multiplier = self.gradient_target / (gradient_norm + 1e-8)
                target_lr = self.current_lr * (1 + self.adaptation_rate * (target_multiplier - 1))
                new_lr = np.clip(target_lr, self.min_lr, self.max_lr)
                adjustment_type = "adaptive"
            else:
                new_lr = self.current_lr
                adjustment_type = "unchanged"

        # Apply new learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        self.current_lr = new_lr

        return {
            'old_lr': old_lr,
            'new_lr': new_lr,
            'adjustment_type': adjustment_type,
            'gradient_norm': gradient_norm,
            'emergency_count': self.emergency_count
        }


def apply_spectral_normalization(module: nn.Module, layer_types: Tuple = (nn.Linear, nn.Conv2d)) -> nn.Module:
    """
    Apply spectral normalization to specific layer types in a module.

    Args:
        module: Module to apply spectral normalization to
        layer_types: Types of layers to normalize

    Returns:
        Module with spectral normalization applied
    """
    for name, child in module.named_children():
        if isinstance(child, layer_types):
            # Apply spectral normalization to this layer
            spectral_child = SpectralNormalization(child)
            setattr(module, name, spectral_child)
        else:
            # Recursively apply to children
            apply_spectral_normalization(child, layer_types)

    return module


class GradientPreservationSuite:
    """
    Complete gradient preservation suite combining all techniques.

    This is the main interface for applying all gradient preservation methods.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[Dict] = None
    ):
        """
        Initialize complete gradient preservation suite.

        Args:
            model: Model to apply preservation to
            optimizer: Optimizer for the model
            config: Configuration dictionary for preservation techniques
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}

        # Initialize components
        self.gradient_scaler = AdaptiveGradientScaler(
            min_gradient_norm=self.config.get('min_gradient_norm', 1e-6),
            max_gradient_norm=self.config.get('max_gradient_norm', 10.0),
            emergency_boost_factor=self.config.get('emergency_boost_factor', 100.0)
        )

        self.flow_monitor = GradientFlowMonitor(
            monitor_frequency=self.config.get('monitor_frequency', 20),
            flow_threshold=self.config.get('flow_threshold', 1e-7)
        )

        self.emergency_injection = EmergencyGradientInjection(
            injection_strength=self.config.get('injection_strength', 0.01),
            noise_scale=self.config.get('noise_scale', 0.001)
        )

        self.lr_scheduler = DynamicLearningRateScheduler(
            base_lr=self.config.get('base_lr', 3e-4),
            emergency_lr_boost=self.config.get('emergency_lr_boost', 10.0)
        )

        # Apply spectral normalization if requested
        if self.config.get('use_spectral_norm', False):
            apply_spectral_normalization(self.model)

        logger.info("[GradientPreservation] Initialized complete preservation suite")

    def preserve_gradients(self, force_emergency: bool = False) -> Dict[str, Any]:
        """
        Apply all gradient preservation techniques.

        Args:
            force_emergency: Force emergency interventions

        Returns:
            Dictionary with preservation statistics
        """
        # 1. Scale gradients adaptively
        scaling_stats = self.gradient_scaler.scale_gradients(
            self.model, force_emergency=force_emergency
        )

        # 2. Monitor gradient flow
        flow_stats = self.flow_monitor.monitor_flow(self.model)

        # 3. Emergency injection if needed
        emergency_needed = (force_emergency or
                          scaling_stats['original_norm'] < 1e-8 or
                          flow_stats.get('bottleneck_count', 0) > 0)

        injection_stats = {}
        if emergency_needed:
            injection_stats = self.emergency_injection.inject_gradients(
                self.model, force_injection=force_emergency
            )

        # 4. Adjust learning rate
        lr_stats = self.lr_scheduler.adjust_lr(
            self.optimizer,
            scaling_stats['original_norm'],
            emergency=emergency_needed
        )

        # 5. Calculate overall health score
        gradient_health = self.flow_monitor.get_flow_health_score()

        return {
            'gradient_health_score': gradient_health,
            'scaling_stats': scaling_stats,
            'flow_stats': flow_stats,
            'injection_stats': injection_stats,
            'lr_stats': lr_stats,
            'emergency_applied': emergency_needed,
            'step': scaling_stats.get('step', 0)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all preservation components."""
        return {
            'gradient_scaler': {
                'current_scale': self.gradient_scaler.current_scale,
                'emergency_interventions': self.gradient_scaler.emergency_interventions,
                'step_count': self.gradient_scaler.step_count
            },
            'flow_monitor': {
                'bottleneck_layers': self.flow_monitor.bottleneck_layers,
                'health_score': self.flow_monitor.get_flow_health_score()
            },
            'emergency_injection': {
                'injection_count': self.emergency_injection.injection_count
            },
            'lr_scheduler': {
                'current_lr': self.lr_scheduler.current_lr,
                'emergency_count': self.lr_scheduler.emergency_count
            }
        }