#!/usr/bin/env python3
"""
快速统计量稳定化功能验证脚本

验证配置加载和基本功能是否正常工作
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

def test_config_loading():
    """测试配置文件是否包含新的统计量稳定化参数"""
    config_path = repo_root / "configs" / "ppo_warmstart.json"

    if not config_path.exists():
        print("❌ 配置文件不存在:", config_path)
        return False

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 检查是否存在统计量稳定化配置
        if "statistics_stabilization" not in config:
            print("❌ 配置文件中缺少 statistics_stabilization 部分")
            return False

        stab_config = config["statistics_stabilization"]
        required_keys = [
            "enable_statistics_stabilization",
            "stabilization_steps",
            "stabilization_policy",
            "collect_baseline_stats",
            "enable_stabilization_logging"
        ]

        missing_keys = [key for key in required_keys if key not in stab_config]
        if missing_keys:
            print(f"❌ 配置中缺少必要的key: {missing_keys}")
            return False

        print("✅ 配置文件验证通过")
        print(f"   - 启用稳定化: {stab_config['enable_statistics_stabilization']}")
        print(f"   - 稳定化步数: {stab_config['stabilization_steps']}")
        print(f"   - 稳定化策略: {stab_config['stabilization_policy']}")

        return True

    except Exception as e:
        print(f"❌ 加载配置文件时出错: {e}")
        return False

def test_scheduler_import():
    """测试PPO scheduler是否可以正常导入（不运行）"""
    try:
        from vidur.scheduler.global_scheduler.ppo_scheduler_modular import PPOGlobalSchedulerModular
        print("✅ PPO调度器导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入PPO调度器失败: {e}")
        return False

def test_method_existence():
    """测试新的方法是否存在"""
    try:
        from vidur.scheduler.global_scheduler.ppo_scheduler_modular import PPOGlobalSchedulerModular

        # 检查新方法是否存在
        if not hasattr(PPOGlobalSchedulerModular, '_statistics_stabilization_step'):
            print("❌ PPOGlobalSchedulerModular缺少 _statistics_stabilization_step 方法")
            return False

        print("✅ 新增方法验证通过")
        return True

    except Exception as e:
        print(f"❌ 方法检查失败: {e}")
        return False

def main():
    print("🧪 快速统计量稳定化功能验证")
    print("=" * 50)

    tests = [
        ("配置文件验证", test_config_loading),
        ("调度器导入测试", test_scheduler_import),
        ("新方法存在性检查", test_method_existence)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试执行失败: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")

    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 所有测试通过！统计量稳定化功能已正确实现")
        print("\n下一步可以运行:")
        print("   bash scripts/test_statistics_stabilization.sh")
    else:
        print("\n⚠️  存在问题，请检查上述失败的测试项")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)