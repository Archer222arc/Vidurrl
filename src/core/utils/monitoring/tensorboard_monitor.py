"""
TensorBoard monitoring for PPO training.

This module provides real-time monitoring and visualization of PPO training
metrics through TensorBoard integration.
"""

import math
import os
import signal
import subprocess
import time
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardMonitor:
    """
    TensorBoard monitor for PPO training metrics.

    Automatically logs training progress, performance metrics, and system
    statistics to TensorBoard for real-time visualization.
    """

    def __init__(
        self,
        log_dir: str = "./outputs/runs/ppo_training",
        enabled: bool = True,
        auto_start: bool = True,
        port: int = 6006,
        host: str = "localhost",
        start_retries: int = 3,
        retry_delay: float = 5.0,
        force_kill: bool = False,
    ):
        """
        Initialize TensorBoard monitor.

        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether monitoring is enabled
            auto_start: Whether to auto-start TensorBoard server
            port: Port for TensorBoard server
            host: Host for TensorBoard server
            start_retries: Number of retries for TensorBoard startup
            retry_delay: Delay in seconds between retries
            force_kill: Whether to terminate existing TensorBoard process on the port before starting
        """
        self.log_dir = log_dir
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.auto_start = auto_start
        self.port = port
        self.host = host
        self.start_retries = start_retries
        self.retry_delay = retry_delay
        self.force_kill = force_kill
        self.writer: Optional[SummaryWriter] = None
        self.step = 0
        self._tb_process = None
        self._auto_start_disabled = False  # Flag to disable auto-start after repeated failures

        if self.enabled:
            self._initialize_writer()
            if self.auto_start:
                self._start_tensorboard_server()
            else:
                print(f"📊 TensorBoard日志记录启用: {self.log_dir}")
                print(f"💡 手动启动TensorBoard: tensorboard --logdir {self.log_dir} --port {self.port} --host {self.host}")

    def _initialize_writer(self) -> None:
        """Initialize TensorBoard SummaryWriter."""
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"📊 TensorBoard监控已启动: {self.log_dir}")

    def _start_tensorboard_server(self) -> None:
        """Auto-start TensorBoard server with retry mechanism."""
        if not self.enabled or self._auto_start_disabled:
            return

        # Normalize URL for user-facing logs
        display_host = "127.0.0.1" if self.host in ["*", "0.0.0.0"] else self.host
        url = f"http://{display_host}:{self.port}"

        if self.force_kill:
            self._force_kill_existing_tensorboard(url)

        print(f"🌐 TensorBoard服务器启动中... 访问: {url}")

        for attempt in range(1, self.start_retries + 1):
            try:
                # Start TensorBoard process
                self._tb_process = subprocess.Popen([
                    "tensorboard", "--logdir", self.log_dir,
                    "--port", str(self.port), "--host", self.host
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Wait and check if process is still running
                time.sleep(self.retry_delay)

                if self._tb_process.poll() is None:
                    # Process is still running - success!
                    if attempt > 1:
                        print(f"✅ TensorBoard启动重试 {attempt} 成功: {url}")
                    else:
                        print(f"✅ TensorBoard服务器启动成功: {url}")
                    return
                else:
                    # Process exited - failure
                    exit_code = self._tb_process.returncode
                    if attempt < self.start_retries:
                        print(f"⚠️  TensorBoard启动尝试 {attempt} 失败 (退出码: {exit_code})，{self.retry_delay}秒后重试...")
                        time.sleep(self.retry_delay)
                    else:
                        print(f"❌ TensorBoard启动尝试 {attempt} 失败 (退出码: {exit_code})")

            except FileNotFoundError:
                if attempt < self.start_retries:
                    print(f"⚠️  TensorBoard命令未找到，尝试 {attempt}/{self.start_retries}，{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"❌ TensorBoard命令未找到，请安装: pip install tensorboard")

            except Exception as e:
                if attempt < self.start_retries:
                    print(f"⚠️  TensorBoard启动异常 (尝试 {attempt}/{self.start_retries}): {e}")
                    print(f"   {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"❌ TensorBoard启动异常: {e}")

        # All retries failed
        print(f"💥 TensorBoard启动失败，已尝试 {self.start_retries} 次")
        print(f"💡 手动启动命令: tensorboard --logdir {self.log_dir} --port {self.port} --host {self.host}")

        # Disable auto-start for the rest of the run to avoid noise
        self._auto_start_disabled = True
        print("🔇 已禁用后续自动启动尝试，避免重复错误信息")

    def _force_kill_existing_tensorboard(self, url: str) -> None:
        """Force terminate any existing process listening on the TensorBoard port."""
        try:
            output = subprocess.check_output(
                [
                    "lsof",
                    "-nP",
                    f"-iTCP:{self.port}",
                    "-sTCP:LISTEN",
                    "-Fp",
                ],
                text=True,
            )
        except FileNotFoundError:
            print("⚠️  无法强制终止: 系统缺少 lsof 命令")
            return
        except subprocess.CalledProcessError:
            # No process currently listening on the port
            return
        except Exception as exc:
            print(f"⚠️  查询端口占用时出错: {exc}")
            return

        pids = []
        for line in output.splitlines():
            if line.startswith("p"):
                try:
                    pids.append(int(line[1:]))
                except ValueError:
                    continue

        if not pids:
            return

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"🛑 已终止占用 TensorBoard 端口的进程 PID={pid}")
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                print(f"⚠️  无权限终止进程 PID={pid}: {exc}")
            except Exception as exc:
                print(f"⚠️  终止进程 PID={pid} 时出错: {exc}")

        # Give the OS a moment to release the port
        time.sleep(1.0)
        print(f"🔁 端口 {self.port} 已清理，将尝试重新启动 TensorBoard: {url}")

    def _resolve_step(self, step: Optional[int]) -> int:
        """Resolve and update the writer's global step."""
        if step is None:
            return self.step

        try:
            self.step = int(step)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid TensorBoard step: {step!r}") from None

        return self.step

    @staticmethod
    def _sanitize_value(value: Any) -> float:
        """Convert values to finite floats accepted by TensorBoard."""
        if value is None:
            return 0.0

        if isinstance(value, bool):
            # bool is an int subclass but make intent explicit
            return 1.0 if value else 0.0

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                return 0.0

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if math.isnan(numeric) or math.isinf(numeric):
            return 0.0

        return numeric

    def _log_scalar(self, tag: str, value: Any, step: Optional[int]) -> None:
        """Log a single scalar value after sanitizing input."""
        if not self.enabled or not self.writer:
            return

        resolved_step = self._resolve_step(step)
        safe_value = self._sanitize_value(value)
        self.writer.add_scalar(tag, safe_value, resolved_step)

    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log PPO training metrics.

        Args:
            metrics: Dictionary of training metrics
            step: Optional explicit global step
        """
        metric_step = self._resolve_step(step)
        if not self.enabled or not self.writer:
            return

        # PPO specific metrics
        ppo_metrics = {
            "Training/PolicyLoss": metrics.get("pi_loss", 0.0),
            "Training/ValueLoss": metrics.get("vf_loss", 0.0),
            "Training/Entropy": metrics.get("entropy", 0.0),
            "Training/ApproxKL": metrics.get("approx_kl", 0.0),
            "Training/ClipFraction": metrics.get("clipfrac", 0.0),
            "Training/ExplainedVariance": metrics.get("explained_var", 0.0),
            "Training/GradientNorm": metrics.get("pg_grad_norm", 0.0),
            "Training/LearningRate": metrics.get("lr", 0.0),
        }

        for metric_name, value in ppo_metrics.items():
            self.writer.add_scalar(metric_name, self._sanitize_value(value), metric_step)

    def log_reward_metrics(
        self,
        reward_info: Dict[str, float],
        reward: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log reward and environment metrics.

        Args:
            reward_info: Dictionary of reward computation details
            reward: Final reward value
            step: Optional explicit global step
        """
        metric_step = self._resolve_step(step)
        if not self.enabled or not self.writer:
            return

        # Reward components
        self.writer.add_scalar("Reward/Total", self._sanitize_value(reward), metric_step)
        self.writer.add_scalar(
            "Reward/Throughput",
            self._sanitize_value(reward_info.get("throughput", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Reward/Latency",
            self._sanitize_value(reward_info.get("latency", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Reward/BalancePenalty",
            self._sanitize_value(reward_info.get("balance_penalty", 0.0)),
            metric_step,
        )

        # Enhanced debugging metrics for reward analysis
        if "raw_reward" in reward_info:
            self.writer.add_scalar(
                "Debug/RawReward",
                self._sanitize_value(reward_info["raw_reward"]),
                metric_step,
            )
        if "reward_clipped" in reward_info:
            self.writer.add_scalar(
                "Debug/RewardClipped",
                1.0 if reward_info["reward_clipped"] else 0.0,
                metric_step,
            )
        if "raw_throughput" in reward_info:
            self.writer.add_scalar(
                "Debug/RawThroughput",
                self._sanitize_value(reward_info["raw_throughput"]),
                metric_step,
            )
        if "raw_latency" in reward_info:
            self.writer.add_scalar(
                "Debug/RawLatency",
                self._sanitize_value(reward_info["raw_latency"]),
                metric_step,
            )
        if "skipped_invalid_metrics" in reward_info:
            self.writer.add_scalar(
                "Debug/SkippedInvalidMetrics",
                1.0 if reward_info["skipped_invalid_metrics"] else 0.0,
                metric_step,
            )

        # Delta mode specific metrics
        if reward_info.get("mode") == "delta":
            self.writer.add_scalar(
                "Reward/DeltaThroughput",
                self._sanitize_value(reward_info.get("delta_throughput", 0.0)),
                metric_step,
            )
            self.writer.add_scalar(
                "Reward/DeltaLatency",
                self._sanitize_value(reward_info.get("delta_latency", 0.0)),
                metric_step,
            )

    def log_rollout_metrics(
        self,
        rollout_stats: Dict[str, float],
        action_distribution: list,
        buffer_size: int,
        max_buffer_size: int,
        step: Optional[int] = None,
    ) -> None:
        """
        Log rollout and buffer statistics.

        Args:
            rollout_stats: Statistics from rollout buffer
            action_distribution: Distribution of actions taken
            buffer_size: Current buffer size
            max_buffer_size: Maximum buffer size
            step: Optional explicit global step
        """
        metric_step = self._resolve_step(step)
        if not self.enabled or not self.writer:
            return

        # Rollout statistics
        self.writer.add_scalar(
            "Rollout/RewardMean",
            self._sanitize_value(rollout_stats.get("reward_mean", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Rollout/RewardStd",
            self._sanitize_value(rollout_stats.get("reward_std", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Rollout/RewardMin",
            self._sanitize_value(rollout_stats.get("reward_min", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Rollout/RewardMax",
            self._sanitize_value(rollout_stats.get("reward_max", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Rollout/ValueMean",
            self._sanitize_value(rollout_stats.get("value_mean", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Rollout/ReturnMean",
            self._sanitize_value(rollout_stats.get("return_mean", 0.0)),
            metric_step,
        )
        self.writer.add_scalar(
            "Rollout/AdvantageStd",
            self._sanitize_value(rollout_stats.get("advantage_std", 0.0)),
            metric_step,
        )

        # Buffer progress
        buffer_progress = buffer_size / max_buffer_size if max_buffer_size > 0 else 0.0
        self.writer.add_scalar(
            "System/BufferProgress", self._sanitize_value(buffer_progress), metric_step
        )

        # Action distribution
        for i, count in enumerate(action_distribution):
            self.writer.add_scalar(
                f"Actions/Replica_{i}", self._sanitize_value(count), metric_step
            )

    def log_system_metrics(
        self,
        system_info: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log system-level metrics.

        Args:
            system_info: System performance metrics
            step: Optional explicit global step
        """
        metric_step = self._resolve_step(step)
        if not self.enabled or not self.writer:
            return

        # System metrics
        for metric_name, value in system_info.items():
            self.writer.add_scalar(
                f"System/{metric_name}", self._sanitize_value(value), metric_step
            )

    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters for experiment tracking.

        Args:
            hparams: Dictionary of hyperparameters
        """
        if not self.enabled or not self.writer:
            return

        # Filter out non-scalar values for TensorBoard
        scalar_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                scalar_hparams[key] = value

        self.writer.add_hparams(scalar_hparams, {})

    def increment_step(self) -> None:
        """Increment the global step counter."""
        self.step += 1

    def set_step(self, step: int) -> None:
        """Set the global step counter."""
        self._resolve_step(step)

    def flush(self) -> None:
        """Flush pending logs to disk."""
        if self.enabled and self.writer:
            self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer and terminate process."""
        if self.enabled and self.writer:
            print("📊 TensorBoard监控已关闭")
            self.writer.close()
            self.writer = None

        # Terminate TensorBoard process if running
        if self._tb_process and self._tb_process.poll() is None:
            try:
                self._tb_process.terminate()
                self._tb_process.wait(timeout=5)
                print("🔌 TensorBoard服务器已停止")
            except Exception:
                # Force kill if terminate doesn't work
                try:
                    self._tb_process.kill()
                    print("🔌 TensorBoard服务器已强制停止")
                except Exception:
                    pass  # Best effort cleanup

    def is_active(self) -> bool:
        """Check if monitoring is active."""
        return self.enabled and self.writer is not None

    def add_custom_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """
        Add a custom scalar metric.

        Args:
            tag: Metric name/tag
            value: Metric value
            step: Optional explicit global step
        """
        self._log_scalar(tag, value, step)


class PPOTrainingDetector:
    """
    Automatically detect if PPO is in training mode.

    Monitors training indicators to enable/disable TensorBoard logging.
    """

    def __init__(self):
        self.training_indicators = {
            "buffer_updates": 0,
            "policy_updates": 0,
            "last_update_step": -1,
        }

    def update_training_activity(self, is_training_update: bool = False) -> None:
        """
        Update training activity indicators.

        Args:
            is_training_update: Whether this is a training update
        """
        if is_training_update:
            self.training_indicators["policy_updates"] += 1
            self.training_indicators["last_update_step"] = self.training_indicators["buffer_updates"]

        self.training_indicators["buffer_updates"] += 1

    def is_training_active(self, recent_window: int = 100) -> bool:
        """
        Determine if training is currently active.

        Args:
            recent_window: Window size for checking recent activity

        Returns:
            True if training appears to be active
        """
        recent_updates = (
            self.training_indicators["buffer_updates"] -
            self.training_indicators["last_update_step"]
        )

        # Consider training active if there were recent policy updates
        return recent_updates <= recent_window and self.training_indicators["policy_updates"] > 0
