#!/usr/bin/env python3
"""
TensorBoard重试机制测试

测试TensorBoard启动重试功能，包括：
- 模拟启动失败和重试成功
- 验证重试次数和延迟
- 测试最终失败后的处理
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from src.rl_components import TensorBoardMonitor


class TestTensorBoardRetry(unittest.TestCase):
    """TensorBoard重试机制测试类"""

    def setUp(self):
        """测试准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "tensorboard_logs")

    def tearDown(self):
        """测试清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_retry_success_on_third_attempt(self, mock_sleep, mock_popen):
        """测试第三次重试成功的情况"""

        # 模拟前两次失败，第三次成功
        processes = []
        for i in range(3):
            mock_proc = Mock()
            if i < 2:
                # 前两次：进程立即退出 (失败)
                mock_proc.poll.return_value = 1  # 退出码1
                mock_proc.returncode = 1
            else:
                # 第三次：进程持续运行 (成功)
                mock_proc.poll.return_value = None  # 仍在运行
            processes.append(mock_proc)

        mock_popen.side_effect = processes

        # 创建TensorBoard监控器，启用自动启动
        monitor = TensorBoardMonitor(
            log_dir=self.log_dir,
            enabled=True,
            auto_start=True,
            port=6006,
            start_retries=3,
            retry_delay=1.0,
        )

        # 验证subprocess.Popen被调用了3次
        self.assertEqual(mock_popen.call_count, 3)

        # 验证sleep被调用了正确次数
        # 3次检查延迟 + 2次重试间隔 = 5次
        self.assertEqual(mock_sleep.call_count, 5)

        # 验证最终进程是第三个（成功的那个）
        self.assertEqual(monitor._tb_process, processes[2])

        monitor.close()

    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_all_retries_fail(self, mock_sleep, mock_popen):
        """测试所有重试都失败的情况"""

        # 模拟所有尝试都失败
        mock_proc = Mock()
        mock_proc.poll.return_value = 1  # 始终退出
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        # 创建TensorBoard监控器
        monitor = TensorBoardMonitor(
            log_dir=self.log_dir,
            enabled=True,
            auto_start=True,
            port=6006,
            start_retries=3,
            retry_delay=0.5,
        )

        # 验证所有3次尝试都被执行
        self.assertEqual(mock_popen.call_count, 3)

        # 验证自动启动被禁用
        self.assertTrue(monitor._auto_start_disabled)

        monitor.close()

    @patch('subprocess.Popen')
    def test_file_not_found_error(self, mock_popen):
        """测试TensorBoard命令未找到的情况"""

        # 模拟FileNotFoundError
        mock_popen.side_effect = FileNotFoundError("tensorboard command not found")

        # 创建TensorBoard监控器
        monitor = TensorBoardMonitor(
            log_dir=self.log_dir,
            enabled=True,
            auto_start=True,
            port=6006,
            start_retries=2,
            retry_delay=0.1,
        )

        # 验证重试次数正确
        self.assertEqual(mock_popen.call_count, 2)

        # 验证自动启动被禁用
        self.assertTrue(monitor._auto_start_disabled)

        monitor.close()

    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_immediate_success(self, mock_sleep, mock_popen):
        """测试首次启动就成功的情况"""

        # 模拟首次启动成功
        mock_proc = Mock()
        mock_proc.poll.return_value = None  # 进程正在运行
        mock_popen.return_value = mock_proc

        # 创建TensorBoard监控器
        monitor = TensorBoardMonitor(
            log_dir=self.log_dir,
            enabled=True,
            auto_start=True,
            port=6006,
            start_retries=3,
            retry_delay=1.0,
        )

        # 验证只调用了一次Popen
        self.assertEqual(mock_popen.call_count, 1)

        # 验证sleep只调用了一次（检查延迟）
        self.assertEqual(mock_sleep.call_count, 1)

        # 验证自动启动未被禁用
        self.assertFalse(monitor._auto_start_disabled)

        monitor.close()

    def test_auto_start_disabled(self):
        """测试自动启动被禁用时不执行启动"""

        with patch('subprocess.Popen') as mock_popen:
            # 创建监控器，但关闭自动启动
            monitor = TensorBoardMonitor(
                log_dir=self.log_dir,
                enabled=True,
                auto_start=False,  # 关闭自动启动
                port=6006,
                start_retries=3,
                retry_delay=1.0,
            )

            # 验证Popen未被调用
            mock_popen.assert_not_called()

            monitor.close()

    def test_url_normalization(self):
        """测试URL规范化功能"""

        with patch('subprocess.Popen') as mock_popen:
            mock_proc = Mock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            # 测试通配符host的规范化
            monitor = TensorBoardMonitor(
                log_dir=self.log_dir,
                enabled=True,
                auto_start=True,
                port=6006,
                host="*",  # 通配符host
                start_retries=1,
                retry_delay=0.1,
            )

            # URL应该被规范化为127.0.0.1
            # 这里我们无法直接验证print输出，但可以确保没有异常

            monitor.close()

    @patch('subprocess.Popen')
    def test_process_cleanup_on_close(self, mock_popen):
        """测试关闭时进程清理"""

        # 模拟成功启动
        mock_proc = Mock()
        mock_proc.poll.return_value = None  # 进程正在运行
        mock_popen.return_value = mock_proc

        monitor = TensorBoardMonitor(
            log_dir=self.log_dir,
            enabled=True,
            auto_start=True,
            port=6006,
            start_retries=1,
            retry_delay=0.1,
        )

        # 关闭监控器
        monitor.close()

        # 验证进程被正确终止
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)


def run_integration_test():
    """集成测试：在真实环境中测试端口冲突"""

    print("🧪 TensorBoard重试机制集成测试")
    print("=" * 50)

    import socket
    import threading
    import time

    # 创建一个虚拟服务器占用端口
    def dummy_server(port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('localhost', port))
            sock.listen(1)
            print(f"🔒 虚拟服务器占用端口 {port}")
            time.sleep(10)  # 保持10秒
            sock.close()
            print(f"🔓 虚拟服务器释放端口 {port}")
        except Exception as e:
            print(f"虚拟服务器错误: {e}")

    # 在6010端口启动虚拟服务器
    server_thread = threading.Thread(target=dummy_server, args=(6010,), daemon=True)
    server_thread.start()

    time.sleep(1)  # 确保服务器启动

    # 尝试在相同端口启动TensorBoard
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = TensorBoardMonitor(
            log_dir=temp_dir,
            enabled=True,
            auto_start=True,
            port=6010,  # 使用被占用的端口
            start_retries=2,
            retry_delay=2.0,
        )

        print("✅ 集成测试完成 - 应该看到端口冲突和重试信息")
        monitor.close()


if __name__ == "__main__":
    # 运行单元测试
    print("🚀 运行TensorBoard重试机制单元测试")
    unittest.main(argv=[''], exit=False, verbosity=2)

    print("\n" + "="*50)

    # 运行集成测试
    run_integration_test()