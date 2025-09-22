"""
Unit tests for MetricsExporter functionality.

This module tests the CSV and Parquet export capabilities of the
MetricsExporter class used in PPO training.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.rl_components.metrics_exporter import (
    MetricsExporter,
    check_parquet_support,
    get_missing_dependencies,
)


class TestMetricsExporter:
    """Test suite for MetricsExporter class."""

    def test_initialization_disabled(self):
        """Test exporter initialization when disabled."""
        exporter = MetricsExporter(enabled=False)
        assert not exporter.enabled
        assert exporter.buffer == []
        assert exporter.flush_count == 0

    def test_initialization_csv_enabled(self):
        """Test CSV exporter initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                export_format="csv",
                enabled=True
            )
            assert exporter.enabled
            assert exporter.export_format == "csv"
            assert Path(temp_dir).exists()

    @pytest.mark.skipif(not check_parquet_support(), reason="Parquet support not available")
    def test_initialization_parquet_enabled(self):
        """Test Parquet exporter initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                export_format="parquet",
                enabled=True
            )
            assert exporter.enabled
            assert exporter.export_format == "parquet"

    def test_invalid_export_format(self):
        """Test invalid export format raises error."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            MetricsExporter(export_format="invalid", enabled=True)

    def test_key_normalization(self):
        """Test dictionary key normalization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(export_path=temp_dir, enabled=True)

            test_data = {
                "Training/Loss": 0.5,
                "Reward-Signal": 1.0,
                "Buffer Size": 100
            }

            normalized = exporter._normalize_keys(test_data)
            expected = {
                "training_loss": 0.5,
                "reward_signal": 1.0,
                "buffer_size": 100
            }

            assert normalized == expected

    def test_append_training_metrics_disabled(self):
        """Test appending metrics when exporter is disabled."""
        exporter = MetricsExporter(enabled=False)
        exporter.append_training_metrics(1, {"loss": 0.5})
        assert len(exporter.buffer) == 0

    def test_append_training_metrics_enabled(self):
        """Test appending training metrics when enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=10,
                enabled=True
            )

            metrics = {"pi_loss": 0.1, "vf_loss": 0.2}
            metadata = {"action": 1, "reward": 5.0}

            exporter.append_training_metrics(
                step=1,
                metrics=metrics,
                metadata=metadata
            )

            assert len(exporter.buffer) == 1
            row = exporter.buffer[0]

            assert row["step"] == 1
            assert row["data_type"] == "training"
            assert row["pi_loss"] == 0.1
            assert row["vf_loss"] == 0.2
            assert row["action"] == 1
            assert row["reward"] == 5.0
            assert "timestamp" in row
            assert "datetime" in row

    def test_append_rollout_metrics(self):
        """Test appending rollout metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(export_path=temp_dir, enabled=True)

            rollout_stats = {
                "reward_mean": 2.5,
                "reward_std": 1.0,
                "value_mean": 3.0
            }
            action_distribution = [10, 5, 8]
            buffer_progress = 0.75

            exporter.append_rollout_metrics(
                step=10,
                rollout_stats=rollout_stats,
                action_distribution=action_distribution,
                buffer_progress=buffer_progress
            )

            assert len(exporter.buffer) == 1
            row = exporter.buffer[0]

            assert row["step"] == 10
            assert row["data_type"] == "rollout"
            assert row["reward_mean"] == 2.5
            assert row["buffer_progress"] == 0.75
            assert row["action_replica_0"] == 10
            assert row["action_replica_1"] == 5
            assert row["action_replica_2"] == 8

    def test_append_system_metrics(self):
        """Test appending system metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(export_path=temp_dir, enabled=True)

            system_info = {
                "QueueLength": 15,
                "BufferProgress": 0.8,
                "SelectedReplica": 2
            }

            exporter.append_system_metrics(step=5, system_info=system_info)

            assert len(exporter.buffer) == 1
            row = exporter.buffer[0]

            assert row["step"] == 5
            assert row["data_type"] == "system"
            assert row["queuelength"] == 15
            assert row["bufferprogress"] == 0.8
            assert row["selectedreplica"] == 2

    def test_auto_flush_on_interval(self):
        """Test automatic flushing when buffer reaches flush interval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=2,
                enabled=True
            )

            # Add first metric - should not flush
            exporter.append_training_metrics(1, {"loss": 0.1})
            assert len(exporter.buffer) == 1
            assert exporter.flush_count == 0

            # Add second metric - should trigger flush
            exporter.append_training_metrics(2, {"loss": 0.2})
            assert len(exporter.buffer) == 0  # Buffer cleared after flush
            assert exporter.flush_count == 1

    def test_csv_flush(self):
        """Test CSV file creation and content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                export_format="csv",
                flush_interval=1,
                enabled=True
            )

            exporter.append_training_metrics(
                step=1,
                metrics={"loss": 0.5},
                metadata={"action": 2}
            )

            # Check that CSV file was created
            csv_files = list(Path(temp_dir).glob("*.csv"))
            assert len(csv_files) == 1

            # Check CSV content
            with open(csv_files[0], 'r') as f:
                content = f.read()
                assert "step" in content
                assert "loss" in content
                assert "action" in content
                assert "training" in content

    @pytest.mark.skipif(not check_parquet_support(), reason="Parquet support not available")
    def test_parquet_flush(self):
        """Test Parquet file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                export_format="parquet",
                flush_interval=1,
                enabled=True
            )

            exporter.append_training_metrics(
                step=1,
                metrics={"loss": 0.5},
                metadata={"action": 2}
            )

            # Check that Parquet file was created
            parquet_files = list(Path(temp_dir).glob("*.parquet"))
            assert len(parquet_files) == 1

    def test_file_rotation(self):
        """Test file rotation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=1,
                enabled=True
            )

            # Add data and flush
            exporter.append_training_metrics(1, {"loss": 0.1})
            first_file = exporter.current_file

            # Rotate files
            exporter.rotate_files()

            # Add more data - should create new file
            exporter.append_training_metrics(2, {"loss": 0.2})
            second_file = exporter.current_file

            assert first_file != second_file
            assert Path(first_file).exists()
            assert Path(second_file).exists()

    def test_export_stats(self):
        """Test export statistics retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                export_format="csv",
                flush_interval=5,
                enabled=True
            )

            # Add some data
            exporter.append_training_metrics(1, {"loss": 0.1})
            exporter.append_training_metrics(2, {"loss": 0.2})

            stats = exporter.get_export_stats()

            assert stats["enabled"] is True
            assert stats["format"] == "csv"
            assert stats["export_path"] == str(temp_dir)
            assert stats["buffer_size"] == 2
            assert stats["flush_count"] == 0  # No flush yet

    def test_close_functionality(self):
        """Test exporter close and cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=10,
                enabled=True
            )

            # Add data that hasn't been flushed
            exporter.append_training_metrics(1, {"loss": 0.1})
            exporter.append_training_metrics(2, {"loss": 0.2})

            assert len(exporter.buffer) == 2
            assert exporter.flush_count == 0

            # Close should flush remaining data
            exporter.close()

            assert len(exporter.buffer) == 0
            assert exporter.flush_count == 1
            assert not exporter.enabled

    def test_missing_dependencies(self):
        """Test dependency checking functions."""
        # Test dependency detection
        missing_csv = get_missing_dependencies("csv")
        assert missing_csv == []  # CSV should always be available

        missing_parquet = get_missing_dependencies("parquet")
        if check_parquet_support():
            assert missing_parquet == []
        else:
            assert len(missing_parquet) > 0

    @patch('src.rl_components.metrics_exporter.PANDAS_AVAILABLE', False)
    def test_parquet_without_pandas(self):
        """Test Parquet export fails gracefully without pandas."""
        with pytest.raises(ImportError, match="pandas"):
            MetricsExporter(export_format="parquet", enabled=True)

    @patch('src.rl_components.metrics_exporter.PARQUET_AVAILABLE', False)
    def test_parquet_without_pyarrow(self):
        """Test Parquet export fails gracefully without pyarrow."""
        with pytest.raises(ImportError, match="pyarrow"):
            MetricsExporter(export_format="parquet", enabled=True)

    def test_filename_generation(self):
        """Test timestamped filename generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(export_path=temp_dir, enabled=True)

            filename1 = exporter._generate_filename("test")
            time.sleep(0.1)
            filename2 = exporter._generate_filename("test")

            assert filename1 != filename2
            assert filename1.startswith("ppo_test_")
            assert filename1.endswith(".csv")


class TestMetricsExporterIntegration:
    """Integration tests for MetricsExporter with realistic workloads."""

    def test_mixed_metric_types(self):
        """Test exporting mixed metric types in sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=10,
                enabled=True
            )

            # Simulate a training sequence
            for step in range(5):
                # Step-level metrics
                exporter.append_training_metrics(
                    step=step,
                    metrics={"reward": step * 0.1, "value": step * 0.2},
                    metadata={"action": step % 3}
                )

                # System metrics every few steps
                if step % 2 == 0:
                    exporter.append_system_metrics(
                        step=step,
                        system_info={"queue_length": 10 - step, "buffer_progress": step / 10.0}
                    )

                # Rollout metrics occasionally
                if step % 3 == 0:
                    exporter.append_rollout_metrics(
                        step=step,
                        rollout_stats={"reward_mean": step * 0.05},
                        action_distribution=[step, step + 1, step + 2],
                        buffer_progress=step / 5.0
                    )

            # Verify all data is buffered
            assert len(exporter.buffer) > 0

            # Flush and verify file creation
            exporter.flush()

            csv_files = list(Path(temp_dir).glob("*.csv"))
            assert len(csv_files) == 1

            # Verify file contains all data types
            with open(csv_files[0], 'r') as f:
                content = f.read()
                assert "training" in content
                assert "system" in content
                assert "rollout" in content

    def test_high_volume_export(self):
        """Test performance with high-volume metric export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MetricsExporter(
                export_path=temp_dir,
                flush_interval=100,
                enabled=True
            )

            # Simulate high-volume training
            start_time = time.time()

            for step in range(500):
                exporter.append_training_metrics(
                    step=step,
                    metrics={
                        "pi_loss": step * 0.001,
                        "vf_loss": step * 0.002,
                        "entropy": step * 0.0005,
                        "approx_kl": step * 0.0001,
                    }
                )

            export_time = time.time() - start_time

            # Should complete reasonably quickly (< 1 second for 500 metrics)
            assert export_time < 1.0

            # Verify some data is buffered and some has been flushed
            assert exporter.flush_count >= 4  # 500 / 100 = 5 flushes

            # Final flush
            exporter.close()

            csv_files = list(Path(temp_dir).glob("*.csv"))
            assert len(csv_files) >= 1