"""检查 LLM Arena 中配置的模型是否可用。

此脚本会：
1. 检查本地 Ollama 模型是否存在，如果不存在提供下载命令
2. 检查 API 模型的配置是否正确
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings


def check_ollama_models():
    """检查 Ollama 本地模型是否存在。"""
    print("\n" + "="*60)
    print("[检查] 本地 Ollama 模型")
    print("="*60)
    
    required_models = [
        "qwen2.5:7b",
        "llama3.1:8b",
        "glm4:9b",
    ]
    
    try:
        import httpx
        
        base_url = "http://localhost:11434"
        
        # 检查 Ollama 服务是否运行
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code != 200:
                print("[X] Ollama 服务未运行或无法访问")
                print("   请先启动 Ollama: ollama serve")
                return False
        except Exception as e:
            print(f"[X] 无法连接到 Ollama 服务: {e}")
            print("   请确保 Ollama 已安装并运行: ollama serve")
            return False
        
        # 获取已安装的模型列表
        response = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        installed_models = response.json().get("models", [])
        installed_model_names = {model.get("name", "") for model in installed_models}
        
        print(f"\n[OK] Ollama 服务运行正常")
        print(f"   已安装的模型: {len(installed_models)} 个")
        
        missing_models = []
        for model_name in required_models:
            # 检查完整名称或带版本号的名称
            found = False
            for installed in installed_model_names:
                if model_name in installed or installed.startswith(model_name.split(":")[0]):
                    found = True
                    print(f"   [OK] {model_name} - 已安装 (找到: {installed})")
                    break
            
            if not found:
                missing_models.append(model_name)
                print(f"   [X] {model_name} - 未安装")
        
        if missing_models:
            print("\n[下载] 需要下载以下模型:")
            for model in missing_models:
                print(f"   ollama pull {model}")
            print("\n[提示] 运行上述命令下载模型（可能需要一些时间）")
            return False
        else:
            print("\n[OK] 所有本地模型都已安装")
            return True
            
    except ImportError:
        print("[X] 缺少依赖: httpx")
        print("   请安装: pip install httpx")
        return False
    except Exception as e:
        print(f"[X] 检查失败: {e}")
        return False


def check_api_models():
    """检查 API 模型配置。"""
    print("\n" + "="*60)
    print("[检查] API 模型配置")
    print("="*60)
    
    try:
        settings = load_settings()
    except Exception as e:
        print(f"[X] 无法加载配置: {e}")
        return False
    
    api_models = {
        "api-deepseek-chat": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "required_config": ["api_key"],
        },
        "api-qwen-max": {
            "provider": "qwen",
            "model": "qwen-max",
            "required_config": ["api_key"],
        },
        "api-glm-4-plus": {
            "provider": "glm",
            "model": "glm-4-plus",
            "required_config": ["api_key"],
        },
        "api-gpt-4o-mini": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "required_config": ["api_key"],
        },
    }
    
    all_ok = True
    
    for model_id, config in api_models.items():
        provider = config["provider"]
        model_name = config["model"]
        required = config["required_config"]
        
        print(f"\n[模型] {model_id}:")
        print(f"   Provider: {provider}")
        print(f"   Model: {model_name}")
        
        # 检查 provider 配置
        if provider == "openai":
            llm_config = settings.llm
            if hasattr(llm_config, "api_key") and llm_config.api_key:
                print(f"   [OK] API Key: 已配置")
            else:
                print(f"   [X] API Key: 未配置")
                all_ok = False
        elif provider == "deepseek":
            # DeepSeek 使用独立的配置
            if hasattr(settings, "deepseek") and hasattr(settings.deepseek, "api_key"):
                if settings.deepseek.api_key:
                    print(f"   [OK] API Key: 已配置")
                else:
                    print(f"   [X] API Key: 未配置")
                    all_ok = False
            else:
                print(f"   [!] DeepSeek 配置: 未找到配置项")
                print(f"      需要在 settings.yaml 中添加 deepseek 配置")
                all_ok = False
        elif provider == "qwen":
            # Qwen 使用独立的配置
            if hasattr(settings, "qwen") and hasattr(settings.qwen, "api_key"):
                if settings.qwen.api_key:
                    print(f"   [OK] API Key: 已配置")
                else:
                    print(f"   [X] API Key: 未配置")
                    all_ok = False
            else:
                print(f"   [!] Qwen 配置: 未找到配置项")
                print(f"      需要在 settings.yaml 中添加 qwen 配置")
                all_ok = False
        elif provider == "glm":
            # GLM 使用独立的配置
            if hasattr(settings, "glm") and hasattr(settings.glm, "api_key"):
                if settings.glm.api_key:
                    print(f"   [OK] API Key: 已配置")
                else:
                    print(f"   [X] API Key: 未配置")
                    all_ok = False
            else:
                print(f"   [!] GLM 配置: 未找到配置项")
                print(f"      需要在 settings.yaml 中添加 glm 配置")
                all_ok = False
    
    if all_ok:
        print("\n[OK] 所有 API 模型配置检查通过")
        print("   [!] 注意: 配置存在不代表 API 可用，请在实际使用时测试")
    else:
        print("\n[X] 部分 API 模型配置缺失")
        print("   请在 config/settings.yaml 中添加相应的配置")
    
    return all_ok


def main():
    """主函数。"""
    print("="*60)
    print("LLM Arena 模型检查工具")
    print("="*60)
    
    # 检查本地模型
    ollama_ok = check_ollama_models()
    
    # 检查 API 模型
    api_ok = check_api_models()
    
    # 总结
    print("\n" + "="*60)
    print("[总结] 检查结果")
    print("="*60)
    
    if ollama_ok and api_ok:
        print("[OK] 所有模型检查通过！")
        return 0
    else:
        print("[!] 部分模型需要配置或下载")
        if not ollama_ok:
            print("   - 本地 Ollama 模型需要下载")
        if not api_ok:
            print("   - API 模型配置需要完善")
        return 1


if __name__ == "__main__":
    sys.exit(main())

