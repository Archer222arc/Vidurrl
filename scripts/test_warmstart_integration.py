#!/usr/bin/env python3
"""
Warmstart集成测试

直接测试PPO调度器是否能正确识别和处理warmstart参数。
"""

import sys
import os
sys.path.insert(0, '.')

import argparse
import tempfile
from pathlib import Path


def simulate_warmstart_args():
    """模拟warmstart训练的命令行参数"""

    # 基础参数
    base_args = [
        "--global_scheduler_config_type", "ppo_modular",
        "--cluster_config_num_replicas", "4",
        "--synthetic_request_generator_config_num_requests", "1000",  # 小数量测试
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "2.0",

        # PPO基础配置
        "--p_p_o_global_scheduler_modular_config_lr", "0.0003",
        "--p_p_o_global_scheduler_modular_config_gamma", "0.95",
        "--p_p_o_global_scheduler_modular_config_clip_ratio", "0.15",
        "--p_p_o_global_scheduler_modular_config_entropy_coef", "0.01",
        "--p_p_o_global_scheduler_modular_config_epochs", "2",  # 小epochs测试
        "--p_p_o_global_scheduler_modular_config_rollout_len", "32",  # 小rollout测试
        "--p_p_o_global_scheduler_modular_config_minibatch_size", "16",

        # 关键的warmstart参数
        "--p_p_o_global_scheduler_modular_config_enable_warm_start",

        # Tensorboard和指标
        "--p_p_o_global_scheduler_modular_config_tensorboard_port", "6006",
        "--p_p_o_global_scheduler_modular_config_tensorboard_auto_start",
        "--p_p_o_global_scheduler_modular_config_metrics_export_enabled",
        "--p_p_o_global_scheduler_modular_config_metrics_export_format", "csv",
    ]

    return base_args


def test_ppo_scheduler_initialization():
    """测试PPO调度器初始化"""
    print("🧪 测试PPO调度器初始化与warmstart参数...")

    try:
        # 模拟参数
        args_list = simulate_warmstart_args()

        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 添加输出目录相关参数
            args_list.extend([
                "--p_p_o_global_scheduler_modular_config_tensorboard_log_dir", f"{temp_dir}/tensorboard",
                "--p_p_o_global_scheduler_modular_config_metrics_export_path", f"{temp_dir}/metrics"
            ])

            print(f"📁 使用临时目录: {temp_dir}")

            # 构造预训练模型路径（为测试创建一个虚拟的）
            fake_pretrain_path = Path(temp_dir) / "fake_pretrain.pt"

            # 创建一个虚拟的预训练模型文件
            import torch
            fake_model_data = {
                'state_dict': {
                    'actor.0.weight': torch.randn(64, 210),
                    'actor.0.bias': torch.randn(64),
                }
            }
            torch.save(fake_model_data, fake_pretrain_path)
            print(f"✅ 创建虚拟预训练模型: {fake_pretrain_path}")

            # 添加预训练模型路径
            args_list.extend([
                "--p_p_o_global_scheduler_modular_config_pretrained_actor_path", str(fake_pretrain_path)
            ])

            print(f"🔧 总共 {len(args_list)} 个参数")

            # 尝试解析参数（不实际运行训练）
            try:
                from vidur.config.config import Config

                print("📋 解析配置参数...")
                config = Config.from_args(args_list)

                # 检查关键配置
                scheduler_config = config.global_scheduler_config
                print(f"📊 调度器类型: {scheduler_config.type}")

                if hasattr(scheduler_config, 'enable_warm_start'):
                    print(f"🔥 Warmstart启用: {scheduler_config.enable_warm_start}")
                else:
                    print("⚠️  Warmstart参数未找到")

                if hasattr(scheduler_config, 'pretrained_actor_path'):
                    print(f"🎭 预训练模型路径: {scheduler_config.pretrained_actor_path}")
                else:
                    print("⚠️  预训练模型路径未找到")

                print("✅ 参数解析成功")
                return True

            except Exception as e:
                print(f"❌ 参数解析失败: {e}")
                import traceback
                traceback.print_exc()
                return False

    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler_warmstart_detection():
    """测试调度器warmstart检测逻辑"""
    print("\n🔍 测试调度器warmstart检测逻辑...")

    try:
        # 检查PPO调度器模块是否能正确导入
        from vidur.scheduler.global_scheduler.ppo_scheduler_modular import PPOGlobalSchedulerModular
        print("✅ PPO调度器模块导入成功")

        # 检查warmstart相关方法是否存在
        required_methods = ['_apply_warm_start', '_load_from_checkpoint']

        for method_name in required_methods:
            if hasattr(PPOGlobalSchedulerModular, method_name):
                print(f"✅ 方法 {method_name} 存在")
            else:
                print(f"⚠️  方法 {method_name} 不存在")
                return False

        print("✅ 调度器warmstart检测通过")
        return True

    except Exception as e:
        print(f"❌ 调度器warmstart检测失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 Warmstart集成测试开始")
    print("=" * 50)

    tests = [
        ("PPO调度器初始化", test_ppo_scheduler_initialization),
        ("调度器Warmstart检测", test_scheduler_warmstart_detection),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🧪 运行测试: {test_name}")
        print("-" * 30)

        try:
            result = test_func()
            results[test_name] = result

            if result:
                print(f"✅ {test_name}: 通过")
            else:
                print(f"❌ {test_name}: 失败")

        except Exception as e:
            print(f"💥 {test_name}: 测试异常 - {e}")
            results[test_name] = False

    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")

    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")

    if passed == total:
        print("🎉 所有测试通过！Warmstart集成正常。")
        return 0
    else:
        print("⚠️  存在问题，需要进一步调试。")
        return 1


if __name__ == "__main__":
    exit(main())