"""
TensorBoard monitoring for toy model simulation.

Provides real-time visualization of queue metrics, replica utilization,
and scheduling performance.
"""

import os
import subprocess
import time
from typing import Dict, List, Optional, Any

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ToyModelTensorBoardMonitor:
    """TensorBoard monitor for toy model M/M/1 queue simulation."""

    def __init__(
        self,
        log_dir: str = "outputs/toymodel/tensorboard",
        enabled: bool = True,
        auto_start: bool = True,
        port: int = 6006,
        clean_previous_runs: bool = True,
    ):
        """
        Initialize TensorBoard monitor.

        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether monitoring is enabled
            auto_start: Whether to auto-start TensorBoard server
            port: Port for TensorBoard server
            clean_previous_runs: Whether to clean previous run logs
        """
        self.log_dir = log_dir
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.auto_start = auto_start
        self.port = port
        self.writer: Optional[SummaryWriter] = None
        self.step = 0
        self._tb_process = None

        if self.enabled:
            # Clean previous runs if requested
            if clean_previous_runs:
                import shutil
                if os.path.exists(self.log_dir):
                    shutil.rmtree(self.log_dir)
                    print(f"🧹 已清理旧日志: {self.log_dir}")

            self._initialize_writer()
            if self.auto_start:
                self._start_tensorboard_server()

    def _initialize_writer(self) -> None:
        """Initialize TensorBoard SummaryWriter."""
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"📊 TensorBoard监控已启动: {self.log_dir}")

    def _kill_existing_tensorboard(self) -> None:
        """Kill existing TensorBoard processes on the same port."""
        try:
            # Find process using the port
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        print(f"🛑 已终止旧TensorBoard进程 PID={pid}")
                    except subprocess.CalledProcessError:
                        pass
                time.sleep(1)  # Wait for port to be released

        except FileNotFoundError:
            # lsof not available, try alternative method
            try:
                result = subprocess.run(
                    ["ps", "aux"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if f'tensorboard' in line and f'--port {self.port}' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            try:
                                subprocess.run(["kill", "-9", pid], check=True)
                                print(f"🛑 已终止旧TensorBoard进程 PID={pid}")
                            except subprocess.CalledProcessError:
                                pass
            except Exception:
                pass

    def _start_tensorboard_server(self) -> None:
        """Auto-start TensorBoard server."""
        if not self.enabled:
            return

        url = f"http://localhost:{self.port}"

        # Kill existing TensorBoard on this port
        self._kill_existing_tensorboard()

        print(f"🌐 TensorBoard服务器启动中... 访问: {url}")

        try:
            # Start TensorBoard in background (detached from parent)
            self._tb_process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir", self.log_dir,
                    "--port", str(self.port),
                    "--reload_interval", "5",  # Reload data every 5 seconds
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )
            time.sleep(2)

            if self._tb_process.poll() is None:
                print(f"✅ TensorBoard服务器启动成功（后台运行）: {url}")
                print(f"💡 停止服务器: lsof -ti :{self.port} | xargs kill -9")
            else:
                print(f"❌ TensorBoard启动失败")
                print(f"💡 手动启动: tensorboard --logdir {self.log_dir} --port {self.port}")

        except FileNotFoundError:
            print(f"❌ TensorBoard命令未找到，请安装: pip install tensorboard")
        except Exception as e:
            print(f"❌ TensorBoard启动异常: {e}")

    def log_queue_metrics(
        self,
        replica_id: int,
        queue_length: int,
        utilization: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log queue metrics for a replica.

        Args:
            replica_id: Replica ID
            queue_length: Current queue length
            utilization: Replica utilization (0-1)
            step: Optional step counter
        """
        if not self.enabled or not self.writer:
            return

        current_step = step if step is not None else self.step

        self.writer.add_scalar(
            f"Queue/Replica_{replica_id}_Length",
            queue_length,
            current_step
        )
        self.writer.add_scalar(
            f"Queue/Replica_{replica_id}_Utilization",
            utilization,
            current_step
        )

    def log_request_metrics(
        self,
        request_type: int,
        assigned_replica: int,
        queue_time: float,
        service_time: float,
        total_time: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics for a completed request.

        Args:
            request_type: Request type (0 or 1)
            assigned_replica: Assigned replica ID
            queue_time: Time spent in queue
            service_time: Time spent in service
            total_time: Total time in system
            step: Optional step counter
        """
        if not self.enabled or not self.writer:
            return

        current_step = step if step is not None else self.step

        self.writer.add_scalar(
            f"Request/Type_{request_type}_QueueTime",
            queue_time,
            current_step
        )
        self.writer.add_scalar(
            f"Request/Type_{request_type}_ServiceTime",
            service_time,
            current_step
        )
        self.writer.add_scalar(
            f"Request/Type_{request_type}_TotalTime",
            total_time,
            current_step
        )


    def log_system_state(
        self,
        current_time: float,
        total_requests_completed: int,
        total_requests_in_system: int,
        step: Optional[int] = None,
    ) -> None:
        """
        Log overall system state.

        Args:
            current_time: Current simulation time
            total_requests_completed: Total completed requests
            total_requests_in_system: Total requests currently in system
            step: Optional step counter
        """
        if not self.enabled or not self.writer:
            return

        current_step = step if step is not None else self.step

        self.writer.add_scalar("System/Time", current_time, current_step)
        self.writer.add_scalar(
            "System/CompletedRequests",
            total_requests_completed,
            current_step
        )
        self.writer.add_scalar(
            "System/RequestsInSystem",
            total_requests_in_system,
            current_step
        )

    def log_aggregate_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log aggregate performance metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step counter
        """
        if not self.enabled or not self.writer:
            return

        current_step = step if step is not None else self.step

        for metric_name, value in metrics.items():
            self.writer.add_scalar(f"Metrics/{metric_name}", value, current_step)

    def log_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Log experiment configuration.

        Args:
            config_dict: Configuration dictionary
        """
        if not self.enabled or not self.writer:
            return

        # Log as text
        config_text = "\n".join([f"{k}: {v}" for k, v in config_dict.items()])
        self.writer.add_text("Config/Experiment", config_text, 0)

    def increment_step(self) -> None:
        """Increment the global step counter."""
        self.step += 1

    def set_step(self, step: int) -> None:
        """Set the global step counter."""
        self.step = step

    def flush(self) -> None:
        """Flush pending logs to disk."""
        if self.enabled and self.writer:
            self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer (but keep server running)."""
        if self.enabled and self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            print("📊 TensorBoard日志已保存")
            print(f"💡 TensorBoard继续运行在后台: http://localhost:{self.port}")

        # Do NOT terminate the background TensorBoard server
        # It will be killed and restarted on next run

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
