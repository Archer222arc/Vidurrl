#!/usr/bin/env python3
"""
验证Warmstart配置传递脚本

检查JSON配置是否正确转换为命令行参数，并验证PPO调度器是否能正确接收参数。
"""

import sys
import json
import subprocess
from pathlib import Path


def check_config_to_args_conversion():
    """检查配置转换为命令行参数是否正确"""
    print("🔍 检查配置文件到命令行参数的转换...")

    config_file = "configs/ppo_warmstart.json"
    output_dir = "/tmp/test_output"

    try:
        # 调用training_config.py获取生成的参数
        result = subprocess.run([
            sys.executable,
            "src/core/utils/infrastructure/config/training_config.py",
            config_file,
            output_dir
        ], capture_output=True, text=True, cwd=".")

        if result.returncode != 0:
            print(f"❌ 配置转换失败: {result.stderr}")
            return False

        args = result.stdout.strip().split()
        print(f"✅ 成功生成 {len(args)} 个命令行参数")

        # 检查关键的warmstart相关参数
        warmstart_params = [
            "--p_p_o_global_scheduler_modular_config_lr",
            "--p_p_o_global_scheduler_modular_config_clip_ratio",
            "--p_p_o_global_scheduler_modular_config_entropy_coef",
            "--p_p_o_global_scheduler_modular_config_entropy_schedule_enable",
            "--p_p_o_global_scheduler_modular_config_enable_cross_replica_attention",
        ]

        found_params = []
        missing_params = []

        for param in warmstart_params:
            if param in args:
                found_params.append(param)
            else:
                missing_params.append(param)

        print(f"\n📊 参数检查结果:")
        print(f"  ✅ 找到参数: {len(found_params)}")
        for p in found_params:
            idx = args.index(p)
            value = args[idx + 1] if idx + 1 < len(args) else "无值"
            print(f"    {p} = {value}")

        if missing_params:
            print(f"  ⚠️  缺失参数: {len(missing_params)}")
            for p in missing_params:
                print(f"    {p}")

        return len(missing_params) == 0

    except Exception as e:
        print(f"❌ 检查过程出错: {e}")
        return False


def check_ppo_scheduler_integration():
    """检查PPO调度器配置集成"""
    print("\n🔍 检查PPO调度器配置集成...")

    # 读取配置文件
    try:
        with open("configs/ppo_warmstart.json", 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False

    print("📄 配置文件读取成功")

    # 检查关键配置项
    key_configs = {
        "PPO基础配置": ["ppo_config", "lr", "clip_ratio", "entropy_coef"],
        "网络架构": ["actor_critic_architecture", "enable_cross_replica_attention"],
        "Warmstart": ["state_dimension_compatibility", "requires_adaptation"],
        "KL正则化": ["kl_regularization", "target_kl"],
        "奖励配置": ["reward_config", "latency_weight"],
        "监控": ["monitoring", "tensorboard_port"],
    }

    all_ok = True

    for category, path in key_configs.items():
        current = config
        missing_path = []

        for key in path:
            if key in current:
                current = current[key]
            else:
                missing_path.append(key)
                break

        if missing_path:
            print(f"  ⚠️  {category}: 缺失路径 {' -> '.join(missing_path)}")
            all_ok = False
        else:
            print(f"  ✅ {category}: 配置存在")

    # 检查关键数值
    print(f"\n📊 关键配置值:")
    try:
        ppo_cfg = config.get("ppo_config", {})
        print(f"  学习率: {ppo_cfg.get('lr', '未设置')}")
        print(f"  熵系数: {ppo_cfg.get('entropy_coef', '未设置')}")
        print(f"  裁剪率: {ppo_cfg.get('clip_ratio', '未设置')}")

        arch_cfg = config.get("actor_critic_architecture", {})
        print(f"  交叉副本注意力: {arch_cfg.get('enable_cross_replica_attention', '未设置')}")

        state_cfg = config.get("state_dimension_compatibility", {})
        print(f"  状态维度兼容: {state_cfg.get('status', '未设置')}")

    except Exception as e:
        print(f"  ⚠️  读取配置值时出错: {e}")
        all_ok = False

    return all_ok


def verify_warmstart_script_logic():
    """验证warmstart脚本逻辑"""
    print("\n🔍 检查warmstart脚本逻辑...")

    script_path = "scripts/train_ppo_warmstart_optimized.sh"

    if not Path(script_path).exists():
        print(f"❌ 脚本文件不存在: {script_path}")
        return False

    with open(script_path, 'r') as f:
        script_content = f.read()

    # 检查关键逻辑片段
    critical_patterns = [
        "enable_warm_start",  # warmstart启用标志
        "pretrained_actor_path",  # 预训练模型路径
        "SKIP_WARMSTART",  # 跳过warmstart逻辑
        "CONFIG_FILE",  # 配置文件变量
        "python.*training_config.py",  # 配置转换调用
        "python.*vidur.main",  # 主程序调用
    ]

    found_patterns = []
    missing_patterns = []

    for pattern in critical_patterns:
        if pattern in script_content:
            found_patterns.append(pattern)
        else:
            missing_patterns.append(pattern)

    print(f"📊 脚本逻辑检查:")
    print(f"  ✅ 找到模式: {len(found_patterns)}")
    for p in found_patterns:
        print(f"    ✓ {p}")

    if missing_patterns:
        print(f"  ⚠️  缺失模式: {len(missing_patterns)}")
        for p in missing_patterns:
            print(f"    ✗ {p}")

    return len(missing_patterns) == 0


def main():
    """主验证流程"""
    print("🚀 开始验证Warmstart配置传递...")
    print("=" * 60)

    checks = [
        ("配置到参数转换", check_config_to_args_conversion),
        ("PPO调度器配置集成", check_ppo_scheduler_integration),
        ("Warmstart脚本逻辑", verify_warmstart_script_logic),
    ]

    results = {}

    for check_name, check_func in checks:
        print(f"\n📋 执行检查: {check_name}")
        print("-" * 40)

        try:
            result = check_func()
            results[check_name] = result

            if result:
                print(f"✅ {check_name}: 通过")
            else:
                print(f"❌ {check_name}: 失败")

        except Exception as e:
            print(f"💥 {check_name}: 检查过程出错 - {e}")
            results[check_name] = False

    # 总结
    print("\n" + "=" * 60)
    print("📊 验证结果总结")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for check_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {check_name}: {status}")

    print(f"\n🎯 总体结果: {passed}/{total} 项检查通过")

    if passed == total:
        print("🎉 所有验证检查均通过！Warmstart配置传递正常。")
        return 0
    else:
        print("⚠️  存在配置问题，需要进一步调试。")
        return 1


if __name__ == "__main__":
    exit(main())