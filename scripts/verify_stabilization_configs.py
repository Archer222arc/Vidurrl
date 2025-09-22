#!/usr/bin/env python3
"""
验证统计量稳定化配置完整性脚本

检查所有相关配置文件和脚本是否正确包含统计量稳定化参数
"""

import json
import re
from pathlib import Path

def check_json_config(config_path):
    """检查JSON配置文件"""
    print(f"\n🔍 检查配置文件: {config_path}")

    if not config_path.exists():
        print(f"❌ 文件不存在: {config_path}")
        return False

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        if 'statistics_stabilization' not in config:
            print("❌ 缺少 statistics_stabilization 配置")
            return False

        stab_config = config['statistics_stabilization']
        required_keys = [
            'enable_statistics_stabilization',
            'stabilization_steps',
            'stabilization_policy',
            'collect_baseline_stats',
            'enable_stabilization_logging'
        ]

        missing_keys = [key for key in required_keys if key not in stab_config]
        if missing_keys:
            print(f"❌ 缺少必要配置: {missing_keys}")
            return False

        print("✅ 配置完整")
        for key, value in stab_config.items():
            if key != 'comment':
                print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ 解析配置失败: {e}")
        return False

def check_script_for_stabilization(script_path, expected_patterns):
    """检查脚本是否包含统计量稳定化相关配置"""
    print(f"\n🔍 检查脚本: {script_path}")

    if not script_path.exists():
        print(f"❌ 文件不存在: {script_path}")
        return False

    try:
        with open(script_path, 'r') as f:
            content = f.read()

        found_patterns = []
        missing_patterns = []

        for pattern_name, pattern in expected_patterns.items():
            if re.search(pattern, content):
                found_patterns.append(pattern_name)
            else:
                missing_patterns.append(pattern_name)

        if missing_patterns:
            print(f"❌ 缺少预期的配置模式: {missing_patterns}")
            print(f"✅ 找到的配置模式: {found_patterns}")
            return False
        else:
            print("✅ 所有预期的配置模式都存在")
            for pattern_name in found_patterns:
                print(f"   ✓ {pattern_name}")
            return True

    except Exception as e:
        print(f"❌ 读取脚本失败: {e}")
        return False

def main():
    print("🧪 统计量稳定化配置完整性验证")
    print("=" * 60)

    # 检查JSON配置文件
    config_files = [
        Path("configs/ppo_warmstart.json"),
        Path("configs/standalone_pretrain.json")
    ]

    config_results = []
    for config_file in config_files:
        result = check_json_config(config_file)
        config_results.append((str(config_file), result))

    # 检查脚本文件
    script_checks = [
        {
            "path": Path("scripts/scheduler_comparison.sh"),
            "patterns": {
                "enable_statistics_stabilization": r"--p_p_o_global_scheduler_modular_config_enable_statistics_stabilization",
                "stabilization_steps": r"--p_p_o_global_scheduler_modular_config_statistics_stabilization_steps",
                "stabilization_logging": r"--p_p_o_global_scheduler_modular_config_enable_stabilization_logging"
            }
        },
        {
            "path": Path("scripts/train_ppo_warmstart.sh"),
            "patterns": {
                "disable_option": r"--disable-stats-stabilization",
                "enable_statistics_stabilization": r"--p_p_o_global_scheduler_modular_config_enable_statistics_stabilization",
                "parameter_handling": r"DISABLE_STATS_STABILIZATION"
            }
        },
        {
            "path": Path("scripts/test_statistics_stabilization.sh"),
            "patterns": {
                "enable_statistics_stabilization": r"--p_p_o_global_scheduler_modular_config_enable_statistics_stabilization",
                "disable_test": r"--no-p_p_o_global_scheduler_modular_config_enable_statistics_stabilization"
            }
        }
    ]

    script_results = []
    for script_check in script_checks:
        result = check_script_for_stabilization(script_check["path"], script_check["patterns"])
        script_results.append((str(script_check["path"]), result))

    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 验证结果汇总")
    print("=" * 60)

    all_passed = True

    print("\n📄 配置文件:")
    for file_path, result in config_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {file_path}: {status}")
        if not result:
            all_passed = False

    print("\n📋 脚本文件:")
    for file_path, result in script_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {file_path}: {status}")
        if not result:
            all_passed = False

    print(f"\n{'🎉 所有检查通过!' if all_passed else '⚠️ 存在问题，请检查上述失败项'}")

    if all_passed:
        print("\n💡 推荐测试:")
        print("   1. python scripts/quick_stabilization_test.py")
        print("   2. bash scripts/test_statistics_stabilization.sh")
        print("   3. bash scripts/scheduler_comparison.sh")

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)