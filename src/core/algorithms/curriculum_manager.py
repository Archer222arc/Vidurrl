"""
Curriculum Learning Manager for PPO Training.

This module implements curriculum learning to progressively increase
training difficulty for better policy convergence and stability.
"""

from typing import Dict, List, Any, Optional
import json


class CurriculumStage:
    """
    Represents a single curriculum learning stage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum stage from configuration.

        Args:
            config: Stage configuration dictionary
        """
        self.name = config["name"]
        self.duration_requests = config["duration_requests"]
        self.qps_scale = config["qps_scale"]
        self.latency_threshold_scale = config["latency_threshold_scale"]
        self.reward_penalty_scale = config["reward_penalty_scale"]

    def __repr__(self) -> str:
        return f"CurriculumStage({self.name}, duration={self.duration_requests})"


class CurriculumManager:
    """
    Manages curriculum learning progression for PPO training.

    Implements progressive difficulty adjustment based on request count
    and performance metrics to avoid local optima and improve convergence.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum manager.

        Args:
            config: Curriculum learning configuration
        """
        self.enabled = config.get("enable", False)
        self.stages = []
        self.current_stage_index = 0
        self.current_stage_requests = 0
        self.total_requests = 0

        if self.enabled:
            for stage_config in config.get("stages", []):
                self.stages.append(CurriculumStage(stage_config))

        self.current_stage = self.stages[0] if self.stages else None

    def update(self, num_requests: int) -> bool:
        """
        Update curriculum state based on processed requests.

        Args:
            num_requests: Number of new requests processed

        Returns:
            bool: True if stage changed, False otherwise
        """
        if not self.enabled or not self.stages:
            return False

        self.current_stage_requests += num_requests
        self.total_requests += num_requests

        # Check if current stage is complete
        if (self.current_stage and
            self.current_stage_requests >= self.current_stage.duration_requests):

            # Move to next stage if available
            if self.current_stage_index < len(self.stages) - 1:
                self.current_stage_index += 1
                self.current_stage = self.stages[self.current_stage_index]
                self.current_stage_requests = 0
                return True

        return False

    def get_current_parameters(self) -> Dict[str, float]:
        """
        Get current curriculum parameters for environment scaling.

        Returns:
            Dictionary with scaling factors for QPS, latency, and rewards
        """
        if not self.enabled or not self.current_stage:
            return {
                "qps_scale": 1.0,
                "latency_threshold_scale": 1.0,
                "reward_penalty_scale": 1.0
            }

        return {
            "qps_scale": self.current_stage.qps_scale,
            "latency_threshold_scale": self.current_stage.latency_threshold_scale,
            "reward_penalty_scale": self.current_stage.reward_penalty_scale
        }

    def get_stage_info(self) -> Dict[str, Any]:
        """
        Get information about current curriculum stage.

        Returns:
            Dictionary with stage name, progress, and parameters
        """
        if not self.enabled or not self.current_stage:
            return {
                "enabled": False,
                "stage_name": "disabled",
                "stage_progress": 0.0,
                "total_stages": 0
            }

        progress = min(1.0, self.current_stage_requests / self.current_stage.duration_requests)

        return {
            "enabled": True,
            "stage_name": self.current_stage.name,
            "stage_index": self.current_stage_index,
            "stage_progress": progress,
            "total_stages": len(self.stages),
            "stage_requests": self.current_stage_requests,
            "stage_duration": self.current_stage.duration_requests,
            "total_requests": self.total_requests,
            "parameters": self.get_current_parameters()
        }

    def is_complete(self) -> bool:
        """
        Check if curriculum learning is complete.

        Returns:
            bool: True if all stages completed or disabled
        """
        if not self.enabled:
            return True

        return (self.current_stage_index >= len(self.stages) - 1 and
                self.current_stage_requests >= self.current_stage.duration_requests)

    def reset(self) -> None:
        """Reset curriculum to first stage."""
        self.current_stage_index = 0
        self.current_stage_requests = 0
        self.total_requests = 0
        self.current_stage = self.stages[0] if self.stages else None


def create_curriculum_manager(config_path: str) -> CurriculumManager:
    """
    Create curriculum manager from configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        CurriculumManager instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    curriculum_config = config.get("curriculum_learning", {})
    return CurriculumManager(curriculum_config)