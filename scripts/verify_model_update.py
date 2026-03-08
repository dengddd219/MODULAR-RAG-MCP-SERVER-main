"""验证模型更新是否完成。

此脚本检查所有旧模型引用是否已被替换为新模型。
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_old_models():
    """检查是否还有旧模型的引用。"""
    print("=" * 60)
    print("验证模型更新")
    print("=" * 60)
    
    old_models = [
        "qwen2.5:7b",
        "llama3.1:8b",
        "ollama-qwen2.5:7b",
        "ollama-llama3.1:8b",
    ]
    
    new_models = [
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
        "ollama-qwen2.5:0.5b",
        "ollama-qwen2.5:1.5b",
    ]
    
    # 搜索所有 Python 文件
    python_files = list(project_root.rglob("*.py"))
    markdown_files = list(project_root.rglob("*.md"))
    all_files = python_files + markdown_files
    
    # 排除一些目录和文件
    excluded_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
    current_script = Path(__file__)
    files_to_check = [
        f for f in all_files
        if not any(excluded in f.parts for excluded in excluded_dirs)
        and f != current_script  # 排除当前脚本自身
    ]
    
    print(f"\n检查 {len(files_to_check)} 个文件...")
    
    found_old = []
    found_new = []
    
    for file_path in files_to_check:
        try:
            content = file_path.read_text(encoding="utf-8")
            
            # 检查旧模型
            for old_model in old_models:
                if old_model in content:
                    found_old.append((str(file_path.relative_to(project_root)), old_model))
            
            # 检查新模型
            for new_model in new_models:
                if new_model in content:
                    found_new.append((str(file_path.relative_to(project_root)), new_model))
        except Exception as e:
            # 跳过无法读取的文件
            continue
    
    # 报告结果
    print("\n" + "=" * 60)
    print("检查结果")
    print("=" * 60)
    
    if found_old:
        print("\n[X] 发现旧模型引用（需要替换）:")
        for file_path, model in found_old:
            print(f"   - {file_path}: {model}")
        return False
    else:
        print("\n[OK] 未发现旧模型引用")
    
    if found_new:
        print("\n[OK] 发现新模型引用:")
        new_model_files = {}
        for file_path, model in found_new:
            if model not in new_model_files:
                new_model_files[model] = []
            new_model_files[model].append(file_path)
        
        for model, files in new_model_files.items():
            print(f"\n   {model}:")
            for file_path in sorted(set(files))[:5]:  # 只显示前5个
                print(f"      - {file_path}")
            if len(set(files)) > 5:
                print(f"      ... 还有 {len(set(files)) - 5} 个文件")
    
    return True


def main():
    """主函数。"""
    success = check_old_models()
    
    print("\n" + "=" * 60)
    print("下一步操作")
    print("=" * 60)
    print("\n1. 下载新模型:")
    print("   ollama pull qwen2.5:0.5b")
    print("   ollama pull qwen2.5:1.5b")
    print("\n2. 验证模型已安装:")
    print("   ollama list")
    print("\n3. 测试模型性能:")
    print("   ollama run qwen2.5:0.5b '你好'")
    print("   ollama run qwen2.5:1.5b '你好'")
    print("\n4. 在 LLM Arena 中测试模型速度")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

