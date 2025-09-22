#!/usr/bin/env python3
"""
预训练模型验证模块

提供模型兼容性检查、格式验证等功能。
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class PretrainedModelValidator:
    """预训练模型验证器"""

    @staticmethod
    def validate_model(model_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        验证预训练模型

        Returns:
            (is_valid, validation_info)
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return False, {"error": "模型文件不存在"}

            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')

            # 检查必要的键
            required_keys = ['model_state_dict', 'model_config']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                return False, {"error": f"缺少必要的键: {missing_keys}"}

            # 提取配置信息
            config = checkpoint['model_config']
            metadata = checkpoint.get('training_metadata', {})

            validation_info = {
                "valid": True,
                "model_config": config,
                "training_metadata": metadata,
                "file_size": model_path.stat().st_size,
                "file_path": str(model_path),
            }

            return True, validation_info

        except Exception as e:
            return False, {"error": f"模型验证失败: {str(e)}"}

    @staticmethod
    def check_compatibility(model_config: Dict[str, Any], target_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        检查模型配置兼容性

        Args:
            model_config: 预训练模型的配置
            target_config: 目标训练的配置

        Returns:
            (is_compatible, message)
        """
        critical_keys = ['state_dim', 'action_dim']

        for key in critical_keys:
            if model_config.get(key) != target_config.get(key):
                return False, f"关键配置不匹配: {key} (模型: {model_config.get(key)}, 目标: {target_config.get(key)})"

        # 检查其他配置项（警告但不阻止）
        warning_keys = ['hidden_size', 'layer_N', 'gru_layers']
        warnings = []

        for key in warning_keys:
            if model_config.get(key) != target_config.get(key):
                warnings.append(f"{key}: 模型={model_config.get(key)}, 目标={target_config.get(key)}")

        message = "兼容" if not warnings else f"兼容但有差异: {'; '.join(warnings)}"
        return True, message

    @staticmethod
    def print_model_info(validation_info: Dict[str, Any]):
        """打印模型信息"""
        if not validation_info.get("valid", False):
            print(f"❌ 模型验证失败: {validation_info.get('error', 'unknown')}")
            return

        config = validation_info["model_config"]
        metadata = validation_info.get("training_metadata", {})

        print("✅ 预训练模型验证通过")
        print(f"📂 文件路径: {validation_info['file_path']}")
        print(f"📊 文件大小: {validation_info['file_size'] / 1024 / 1024:.2f} MB")
        print(f"🔧 模型配置:")
        print(f"   - 状态维度: {config.get('state_dim', 'unknown')}")
        print(f"   - 动作维度: {config.get('action_dim', 'unknown')}")
        print(f"   - 隐藏层大小: {config.get('hidden_size', 'unknown')}")
        print(f"   - MLP层数: {config.get('layer_N', 'unknown')}")
        print(f"   - GRU层数: {config.get('gru_layers', 'unknown')}")

        if metadata:
            print(f"📈 训练信息:")
            print(f"   - 训练epoch: {metadata.get('epoch', 'unknown')}")
            print(f"   - 最佳损失: {metadata.get('best_loss', 'unknown')}")
            print(f"   - 模型类型: {metadata.get('model_type', 'unknown')}")
            print(f"   - 保存时间: {metadata.get('save_time', 'unknown')}")


def validate_pretrained_model(model_path: str, target_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    验证预训练模型的便捷函数

    Args:
        model_path: 模型文件路径
        target_config: 目标配置（可选）

    Returns:
        是否验证通过
    """
    validator = PretrainedModelValidator()

    # 基本验证
    is_valid, validation_info = validator.validate_model(model_path)

    if not is_valid:
        validator.print_model_info(validation_info)
        return False

    # 打印模型信息
    validator.print_model_info(validation_info)

    # 兼容性检查
    if target_config:
        model_config = validation_info["model_config"]
        is_compatible, message = validator.check_compatibility(model_config, target_config)

        if is_compatible:
            print(f"✅ 配置兼容性: {message}")
        else:
            print(f"❌ 配置兼容性: {message}")
            return False

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python -m src.pretraining.model_validator <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    success = validate_pretrained_model(model_path)
    sys.exit(0 if success else 1)