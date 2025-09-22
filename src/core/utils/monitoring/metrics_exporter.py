"""
Metrics export utility for PPO training.

This module provides structured data export capabilities for training metrics,
supporting both CSV and Parquet formats for downstream analysis.
"""

import atexit
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Check for pandas/pyarrow availability
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class MetricsExporter:
    """
    Metrics exporter for PPO training data.

    Buffers training metrics and exports them to CSV or Parquet files
    for analysis and visualization. Supports configurable flush intervals
    and automatic file rotation.
    """

    def __init__(
        self,
        export_path: str = "./outputs/runs/ppo_training/exports",
        export_format: str = "csv",
        flush_interval: int = 50,
        enabled: bool = True,
    ):
        """
        Initialize metrics exporter.

        Args:
            export_path: Directory for exported files
            export_format: Export format ('csv' or 'parquet')
            flush_interval: Number of rows to buffer before flushing
            enabled: Whether export is enabled
        """
        self.export_path = Path(export_path)
        self.export_format = export_format.lower()
        self.flush_interval = flush_interval
        self.enabled = enabled

        # Data buffer
        self.buffer: List[Dict[str, Any]] = []
        self.flush_count = 0

        # File management
        self.current_file: Optional[str] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        self.csv_file_handle = None

        # Validation
        if self.enabled:
            self._validate_configuration()
            self._setup_export_directory()

        # Register cleanup
        atexit.register(self.close)

    def _validate_configuration(self) -> None:
        """Validate export configuration and dependencies."""
        if self.export_format not in ["csv", "parquet"]:
            raise ValueError(f"Unsupported export format: {self.export_format}")

        if self.export_format == "parquet":
            if not PANDAS_AVAILABLE:
                raise ImportError(
                    "Parquet export requires pandas. Install with: pip install pandas"
                )
            if not PARQUET_AVAILABLE:
                raise ImportError(
                    "Parquet export requires pyarrow. Install with: pip install pyarrow"
                )

        print(f"📊 指标导出已启用: 格式={self.export_format}, 路径={self.export_path}")

    def _setup_export_directory(self) -> None:
        """Create export directory if it doesn't exist."""
        self.export_path.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, data_type: str = "metrics") -> str:
        """Generate timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ppo_{data_type}_{timestamp}.{self.export_format}"

    def _normalize_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dictionary keys for consistent column names."""
        normalized = {}
        for key, value in data.items():
            # Replace special characters and standardize naming
            clean_key = key.replace("/", "_").replace("-", "_").replace(" ", "_")
            clean_key = clean_key.lower()
            normalized[clean_key] = value
        return normalized

    def append_training_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append training metrics to buffer.

        Args:
            step: Training step number
            metrics: Training metrics dictionary
            metadata: Optional metadata (action, reward, etc.)
        """
        if not self.enabled:
            return

        # Create base row
        row = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "step": step,
            "data_type": "training",
        }

        # Add normalized metrics
        normalized_metrics = self._normalize_keys(metrics)
        row.update(normalized_metrics)

        # Add metadata if provided
        if metadata:
            normalized_metadata = self._normalize_keys(metadata)
            row.update(normalized_metadata)

        self.buffer.append(row)

        # Check if we need to flush
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def append_rollout_metrics(
        self,
        step: int,
        rollout_stats: Dict[str, float],
        action_distribution: List[int],
        buffer_progress: float,
    ) -> None:
        """
        Append rollout metrics to buffer.

        Args:
            step: Training step number
            rollout_stats: Rollout statistics
            action_distribution: Action distribution counts
            buffer_progress: Buffer fill progress (0.0 to 1.0)
        """
        if not self.enabled:
            return

        # Create rollout row
        row = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "step": step,
            "data_type": "rollout",
            "buffer_progress": buffer_progress,
        }

        # Add normalized rollout stats
        normalized_stats = self._normalize_keys(rollout_stats)
        row.update(normalized_stats)

        # Add action distribution
        for i, count in enumerate(action_distribution):
            row[f"action_replica_{i}"] = count

        self.buffer.append(row)

    def append_system_metrics(
        self,
        step: int,
        system_info: Dict[str, float],
    ) -> None:
        """
        Append system metrics to buffer.

        Args:
            step: Training step number
            system_info: System performance metrics
        """
        if not self.enabled:
            return

        # Create system row
        row = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "step": step,
            "data_type": "system",
        }

        # Add normalized system info
        normalized_system = self._normalize_keys(system_info)
        row.update(normalized_system)

        self.buffer.append(row)

    def flush(self) -> None:
        """Flush buffered data to file."""
        if not self.enabled or not self.buffer:
            return

        try:
            if self.export_format == "csv":
                self._flush_csv()
            elif self.export_format == "parquet":
                self._flush_parquet()

            self.flush_count += 1
            rows_flushed = len(self.buffer)
            self.buffer.clear()

            print(f"📤 指标导出: {rows_flushed} 行已写入 ({self.export_format})")

        except Exception as e:
            print(f"❌ 指标导出失败: {e}")

    def _flush_csv(self) -> None:
        """Flush data to CSV file."""
        if not self.buffer:
            return

        # Get all unique fieldnames from buffer
        all_fieldnames = set()
        for row in self.buffer:
            all_fieldnames.update(row.keys())
        fieldnames = sorted(all_fieldnames)

        # Generate filename if needed
        if not self.current_file:
            self.current_file = self.export_path / self._generate_filename()

        # Check if we need to write header
        write_header = not self.current_file.exists()

        # Open file and write data
        with open(self.current_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            for row in self.buffer:
                # Fill missing keys with None
                complete_row = {field: row.get(field) for field in fieldnames}
                writer.writerow(complete_row)

    def _flush_parquet(self) -> None:
        """Flush data to Parquet file."""
        if not self.buffer:
            return

        # Convert buffer to DataFrame
        df = pd.DataFrame(self.buffer)

        # Generate filename
        filename = self.export_path / self._generate_filename()

        # Write to Parquet
        df.to_parquet(filename, engine='pyarrow', index=False)

        # Update current file reference
        self.current_file = filename

    def rotate_files(self) -> None:
        """Rotate to a new file (useful for long training runs)."""
        if not self.enabled:
            return

        # Flush any pending data
        self.flush()

        # Reset file reference to force new file creation
        self.current_file = None
        print(f"🔄 指标文件轮换: 下一次导出将创建新文件")

    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics."""
        stats = {
            "enabled": self.enabled,
            "format": self.export_format,
            "export_path": str(self.export_path),
            "buffer_size": len(self.buffer),
            "flush_count": self.flush_count,
            "current_file": str(self.current_file) if self.current_file else None,
        }
        return stats

    def close(self) -> None:
        """Close exporter and flush any remaining data."""
        if not self.enabled:
            return

        try:
            # Flush remaining data
            if self.buffer:
                self.flush()

            # Close CSV file handle if open
            if self.csv_file_handle:
                self.csv_file_handle.close()
                self.csv_file_handle = None

            print(f"📊 指标导出已关闭: 总计 {self.flush_count} 次写入")

        except Exception as e:
            print(f"⚠️  指标导出关闭时发生错误: {e}")

        self.enabled = False


# Availability check functions
def check_parquet_support() -> bool:
    """Check if Parquet export is supported."""
    return PANDAS_AVAILABLE and PARQUET_AVAILABLE


def get_missing_dependencies(export_format: str) -> List[str]:
    """Get list of missing dependencies for given export format."""
    missing = []

    if export_format == "parquet":
        if not PANDAS_AVAILABLE:
            missing.append("pandas")
        if not PARQUET_AVAILABLE:
            missing.append("pyarrow")

    return missing