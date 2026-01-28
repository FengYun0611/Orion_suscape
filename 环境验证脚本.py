#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境验证脚本 - Environment Verification Script

运行此脚本以验证训练环境是否正确配置。
Run this script to verify that your training environment is correctly configured.
"""

import sys
import os
from pathlib import Path

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python():
    """检查Python版本"""
    print_section("1. Python环境检查")
    print(f"Python版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print("✓ Python版本符合要求 (>= 3.8)")
        return True
    else:
        print("✗ Python版本不符合要求，需要 >= 3.8")
        return False

def check_packages():
    """检查必需的包"""
    print_section("2. 必需包检查")
    
    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mmcv': 'MMCV',
        'numpy': 'NumPy',
        'transformers': 'Transformers',
    }
    
    all_ok = True
    for module_name, display_name in packages.items():
        try:
            if module_name == 'cv2':
                import cv2
                print(f"✓ {display_name}: {cv2.__version__}")
            elif module_name == 'torch':
                import torch
                print(f"✓ {display_name}: {torch.__version__}")
                # 检查CUDA
                if torch.cuda.is_available():
                    print(f"  ✓ CUDA可用: {torch.cuda.get_device_name(0)}")
                    print(f"  ✓ CUDA版本: {torch.version.cuda}")
                else:
                    print(f"  ⚠ CUDA不可用（将使用CPU训练）")
            elif module_name == 'transformers':
                import transformers
                print(f"✓ {display_name}: {transformers.__version__}")
            elif module_name == 'numpy':
                import numpy
                print(f"✓ {display_name}: {numpy.__version__}")
            elif module_name == 'mmcv':
                import mmcv
                print(f"✓ {display_name}: 已安装")
        except ImportError as e:
            print(f"✗ {display_name}: 未安装 ({e})")
            all_ok = False
    
    return all_ok

def check_files():
    """检查关键文件"""
    print_section("3. 关键文件检查")
    
    files_to_check = [
        ('adzoo/orion/train.py', '训练脚本'),
        ('adzoo/orion/configs/suscape_finetune.py', '微调配置文件'),
        ('mmcv/datasets/suscape_orion_dataset.py', 'SUScape数据集'),
        ('mmcv/models/detectors/orion.py', 'Orion模型'),
    ]
    
    all_ok = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {description}: {file_path} ({size} bytes)")
        else:
            print(f"✗ {description}: {file_path} (不存在)")
            all_ok = False
    
    return all_ok

def check_symlinks():
    """检查符号链接"""
    print_section("4. 符号链接检查")
    
    symlinks_to_check = [
        ('./ckpts', 'Checkpoints目录'),
        ('./ckpts/pretrain_qformer', 'Tokenizer目录'),
    ]
    
    any_exists = False
    for link_path, description in symlinks_to_check:
        if os.path.exists(link_path):
            if os.path.islink(link_path):
                target = os.readlink(link_path)
                print(f"✓ {description}: {link_path} -> {target}")
                any_exists = True
            elif os.path.isdir(link_path):
                print(f"✓ {description}: {link_path} (目录存在)")
                any_exists = True
        else:
            print(f"⚠ {description}: {link_path} (不存在)")
    
    if not any_exists:
        print("\n提示: 如果您的tokenizer在其他位置，请创建符号链接：")
        print("  ln -s /path/to/real/ckpts ./ckpts")
    
    return True  # 这不是致命错误

def check_data():
    """检查数据文件"""
    print_section("5. 数据文件检查")
    
    data_files = [
        'data/suscape_infos/suscape_infos_test.pkl',
        'data/suscape_infos/suscape_infos_train.pkl',
        'data/suscape_infos/suscape_infos_val.pkl',
    ]
    
    found_any = False
    for data_file in data_files:
        if os.path.exists(data_file):
            size = os.path.getsize(data_file)
            print(f"✓ {data_file} ({size / 1024 / 1024:.2f} MB)")
            found_any = True
        else:
            print(f"⚠ {data_file} (不存在)")
    
    if not found_any:
        print("\n提示: 请确保数据文件已准备好")
    
    return True  # 这不是致命错误

def check_recent_fixes():
    """检查最近的修复是否已应用"""
    print_section("6. 代码修复检查")
    
    # 检查关键修复
    fixes = []
    
    # Fix 1: --load-from parameter
    with open('adzoo/orion/train.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if '--load-from' in content:
            fixes.append(('--load-from参数', True))
        else:
            fixes.append(('--load-from参数', False))
    
    # Fix 2: Tokenizer local_files_only
    with open('mmcv/models/detectors/orion.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'local_files_only=True' in content:
            fixes.append(('Tokenizer local_files_only', True))
        else:
            fixes.append(('Tokenizer local_files_only', False))
    
    # Fix 3: PKL format handling
    with open('mmcv/datasets/suscape_orion_dataset.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if "isinstance(data_infos, dict) and 'infos' in data_infos" in content:
            fixes.append(('PKL格式处理', True))
        else:
            fixes.append(('PKL格式处理', False))
    
    # Fix 4: shuffler_sampler
    with open('adzoo/orion/configs/suscape_finetune.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'shuffler_sampler' in content:
            fixes.append(('DataLoader配置', True))
        else:
            fixes.append(('DataLoader配置', False))
    
    # Fix 5: VQA data format
    with open('mmcv/datasets/pipelines/formating.py', 'r', encoding='utf-8') as f:
        content = f.read()
        # 检查是否移除了DC包装（注释或pass）
        has_fix = 'input_ids' in content and ('pass' in content or '# Keep as-is' in content or 'Skip wrapping' in content)
        fixes.append(('VQA数据格式', has_fix))
    
    all_ok = True
    for fix_name, status in fixes:
        if status:
            print(f"✓ {fix_name}: 已应用")
        else:
            print(f"✗ {fix_name}: 未应用")
            all_ok = False
    
    return all_ok

def main():
    """主函数"""
    print("="*60)
    print("  Orion SUScape训练环境验证")
    print("  Environment Verification for Orion SUScape Training")
    print("="*60)
    
    results = []
    results.append(("Python环境", check_python()))
    results.append(("必需包", check_packages()))
    results.append(("关键文件", check_files()))
    results.append(("符号链接", check_symlinks()))
    results.append(("数据文件", check_data()))
    results.append(("代码修复", check_recent_fixes()))
    
    # 总结
    print_section("验证总结")
    
    all_ok = True
    for name, status in results:
        status_str = "✓ 通过" if status else "✗ 失败"
        print(f"{name}: {status_str}")
        if not status:
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ 所有检查通过！环境配置正确。")
        print("您可以开始训练了。")
    else:
        print("✗ 部分检查未通过。")
        print("请根据上述提示修复问题后再开始训练。")
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
