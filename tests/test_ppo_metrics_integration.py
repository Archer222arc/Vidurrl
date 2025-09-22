"""
Integration tests for MetricsExporter with PPO scheduler.

This module tests the end-to-end integration of metrics export
functionality within the PPO scheduling system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Mock the vidur imports to avoid dependency issues in tests
with patch.dict('sys.modules', {
    'vidur.config.config': Mock(),
    'vidur.scheduler.global_scheduler.base_global_scheduler': Mock(),
    'vidur.metrics': Mock(),
}):
    from src.core.utils.monitoring.metrics_exporter import MetricsExporter


class TestPPOMetricsIntegration:
    """Integration tests for PPO scheduler with metrics export."""

    def test_metrics_export_configuration(self):
        """Test that metrics export can be configured properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test configuration scenarios
            configs = [
                {
                    "enabled": True,
                    "format": "csv",
                    "path": temp_dir,
                    "interval": 50
                },
                {
                    "enabled": False,
                    "format": "csv",
                    "path": temp_dir,
                    "interval": 50
                },
            ]

            for config in configs:
                exporter = MetricsExporter(
                    export_path=config["path"],
                    export_format=config["format"],
                    flush_interval=config["interval"],
                    enabled=config["enabled"]
                )

                assert exporter.enabled == config["enabled"]
                assert exporter.export_format == config["format"]
                assert exporter.flush_interval == config["interval"]

    def test_step_by_step_metrics_simulation(self):
        """Simulate step-by-step metrics collection like in PPO scheduler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=10,
                enabled=True
            )

            # Simulate multiple training steps
            for step in range(25):
                # Simulate step-level metrics (schedule method)
                step_metrics = {
                    "queue_length": 5 + (step % 3),
                    "selected_replica": step % 4,
                    "action_value": step % 4,
                    "reward": step * 0.1 + (step % 2) * 0.05,
                    "value_estimate": step * 0.2,
                    "log_probability": -(step * 0.01 + 0.5),
                }

                reward_breakdown = {
                    "throughput": step * 0.05,
                    "latency": -(step * 0.02),
                    "balance_penalty": -(step % 5) * 0.01,
                }
                step_metrics.update(reward_breakdown)

                exporter.append_training_metrics(
                    step=step,
                    metrics=step_metrics,
                    metadata={
                        "replica_id": f"replica_{step % 4}",
                        "action": step % 4,
                        "queue_size": 5 + (step % 3),
                    }
                )

                # Simulate PPO update every 10 steps (rollout completion)
                if step > 0 and step % 10 == 0:
                    # PPO training metrics
                    ppo_metrics = {
                        "pi_loss": 0.1 + step * 0.001,
                        "vf_loss": 0.2 + step * 0.002,
                        "entropy": 0.5 - step * 0.001,
                        "approx_kl": step * 0.0001,
                        "clipfrac": 0.1 + (step % 5) * 0.02,
                        "explained_var": 0.8 - step * 0.001,
                    }

                    exporter.append_training_metrics(
                        step=step,
                        metrics=ppo_metrics,
                        metadata={
                            "data_type": "ppo_update",
                            "rollout_length": 10,
                            "buffer_size": 10,
                        }
                    )

                    # Rollout statistics
                    rollout_stats = {
                        "reward_mean": step * 0.05,
                        "reward_std": 0.1 + step * 0.001,
                        "reward_min": step * 0.02,
                        "reward_max": step * 0.08,
                        "value_mean": step * 0.1,
                        "return_mean": step * 0.12,
                        "advantage_std": 0.2 + step * 0.001,
                    }

                    action_distribution = [
                        step // 4, (step + 1) // 4,
                        (step + 2) // 4, (step + 3) // 4
                    ]

                    exporter.append_rollout_metrics(
                        step=step,
                        rollout_stats=rollout_stats,
                        action_distribution=action_distribution,
                        buffer_progress=1.0  # Buffer full at update
                    )

            # Verify export happened
            exporter.close()

            csv_files = list(Path(temp_dir).glob("*.csv"))
            assert len(csv_files) >= 1

            # Verify content diversity
            with open(csv_files[0], 'r') as f:
                content = f.read()

                # Should contain step-level data
                assert "queue_length" in content
                assert "reward" in content
                assert "value_estimate" in content

                # Should contain PPO update data
                assert "pi_loss" in content
                assert "vf_loss" in content
                assert "entropy" in content

                # Should contain rollout data
                assert "rollout" in content
                assert "action_replica_" in content

    def test_error_handling_in_export(self):
        """Test graceful error handling during export operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=2,
                enabled=True
            )

            # Add valid metrics
            exporter.append_training_metrics(
                step=1,
                metrics={"loss": 0.1},
                metadata={"action": 1}
            )

            # Simulate filesystem error during flush
            with patch('builtins.open', side_effect=IOError("Disk full")):
                # This should not crash, just print error
                exporter.append_training_metrics(
                    step=2,
                    metrics={"loss": 0.2},
                    metadata={"action": 2}
                )

            # Exporter should still be functional
            assert exporter.enabled
            assert len(exporter.buffer) >= 1

    def test_concurrent_tensorboard_and_export(self):
        """Test metrics export working alongside TensorBoard monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=5,
                enabled=True
            )

            # Simulate scenario where both TensorBoard and export are active
            for step in range(12):
                # Step metrics (always collected)
                exporter.append_training_metrics(
                    step=step,
                    metrics={"reward": step * 0.1, "action": step % 3}
                )

                # PPO update metrics (every 5 steps)
                if step % 5 == 0 and step > 0:
                    # This simulates the condition in PPO scheduler where
                    # rollout stats are computed when either TensorBoard OR
                    # metrics export is enabled

                    exporter.append_training_metrics(
                        step=step,
                        metrics={"pi_loss": 0.1, "vf_loss": 0.2},
                        metadata={"data_type": "ppo_update"}
                    )

                    exporter.append_rollout_metrics(
                        step=step,
                        rollout_stats={"reward_mean": step * 0.05},
                        action_distribution=[step, step+1, step+2],
                        buffer_progress=1.0
                    )

            exporter.close()

            # Verify export worked
            csv_files = list(Path(temp_dir).glob("*.csv"))
            assert len(csv_files) >= 1

    def test_export_stats_monitoring(self):
        """Test export statistics for monitoring export health."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=3,
                enabled=True
            )

            # Initial stats
            stats = exporter.get_export_stats()
            assert stats["buffer_size"] == 0
            assert stats["flush_count"] == 0
            assert stats["enabled"] is True

            # Add some data
            for i in range(5):
                exporter.append_training_metrics(i, {"loss": i * 0.1})

            # Check intermediate stats
            stats = exporter.get_export_stats()
            assert stats["buffer_size"] == 2  # 5 items, 3 flushed, 2 remaining
            assert stats["flush_count"] == 1   # One flush occurred at item 3

            # Final flush
            exporter.close()

            stats = exporter.get_export_stats()
            assert stats["buffer_size"] == 0
            assert stats["flush_count"] == 2  # Initial flush + close flush

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test very small flush interval
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=1,
                enabled=True
            )

            # Every metric should trigger flush
            for i in range(3):
                exporter.append_training_metrics(i, {"loss": i})
                # Buffer should be empty after each append
                assert len(exporter.buffer) == 0

            assert exporter.flush_count == 3

            # Test large flush interval
            exporter2 = MetricsExporter(
                export_path=temp_dir,
                flush_interval=1000,
                enabled=True
            )

            # Add many metrics without flushing
            for i in range(50):
                exporter2.append_training_metrics(i, {"loss": i})

            assert len(exporter2.buffer) == 50
            assert exporter2.flush_count == 0

            # Manual flush
            exporter2.flush()
            assert len(exporter2.buffer) == 0
            assert exporter2.flush_count == 1

    @pytest.mark.skipif(
        not pytest.importorskip("pandas", reason="pandas not available"),
        reason="Parquet tests require pandas"
    )
    def test_parquet_export_integration(self):
        """Test Parquet export in realistic scenario."""
        pytest.importorskip("pyarrow")

        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                export_format="parquet",
                flush_interval=10,
                enabled=True
            )

            # Add diverse metrics
            for step in range(15):
                exporter.append_training_metrics(
                    step=step,
                    metrics={
                        "reward": step * 0.1,
                        "value": step * 0.2,
                        "action": step % 4
                    },
                    metadata={"replica": f"r_{step % 3}"}
                )

            exporter.close()

            # Verify Parquet file creation
            parquet_files = list(Path(temp_dir).glob("*.parquet"))
            assert len(parquet_files) >= 1

            # Verify we can read the Parquet file
            import pandas as pd
            df = pd.read_parquet(parquet_files[0])

            assert len(df) == 15
            assert "step" in df.columns
            assert "reward" in df.columns
            assert "data_type" in df.columns