"""检查 DeepSeek API 连接状态的脚本。

此脚本会测试 DeepSeek API 的连接状态，包括：
1. 检查配置是否正确
2. 测试 API 连接
3. 测试简单的 API 调用
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import Message


def check_deepseek_connection() -> None:
    """检查 DeepSeek API 连接状态。"""
    print("=" * 60)
    print("DeepSeek API 连接状态检查")
    print("=" * 60)
    print()
    
    try:
        # 加载配置
        print("📋 步骤 1: 加载配置...")
        settings = load_settings()
        print(f"   ✓ 配置加载成功")
        print(f"   - Provider: {settings.llm.provider}")
        print(f"   - Model: {settings.llm.model}")
        print(f"   - Base URL: {getattr(settings.llm, 'base_url', '未设置')}")
        print(f"   - API Key: {'已设置' if getattr(settings.llm, 'api_key', None) else '未设置'}")
        print()
        
        # 检查是否使用智增增代理（LLM Arena中的配置方式）
        base_url = getattr(settings.llm, 'base_url', None)
        api_key = getattr(settings.llm, 'api_key', None)
        
        if not api_key:
            print("❌ 错误: 未找到 API Key")
            print("   请在 config/settings.yaml 中设置 llm.api_key")
            return False
        
        # 检查是否通过智增增代理访问 DeepSeek
        is_zhizengzeng = base_url and "zhizengzeng.com" in base_url
        is_deepseek_direct = base_url and "deepseek.com" in base_url
        
        if is_zhizengzeng:
            print("📋 步骤 2: 检测到智增增代理配置")
            print(f"   - 代理地址: {base_url}")
            print(f"   - 将通过代理访问 DeepSeek API")
            print()
            
            # 使用 OpenAI provider 通过代理访问 DeepSeek
            print("📋 步骤 3: 创建 LLM 客户端（通过智增增代理）...")
            try:
                from src.libs.llm.openai_llm import OpenAILLM
                
                # 创建 OpenAI LLM 实例（通过代理访问 DeepSeek）
                llm = OpenAILLM(
                    settings=settings,
                    model="deepseek-chat",  # DeepSeek 模型名称
                    base_url=base_url,
                    api_key=api_key,
                )
                print("   ✓ LLM 客户端创建成功")
                print()
            except Exception as e:
                print(f"   ❌ LLM 客户端创建失败: {e}")
                return False
        elif is_deepseek_direct:
            print("📋 步骤 2: 检测到直接 DeepSeek API 配置")
            print(f"   - API 地址: {base_url}")
            print()
            
            # 使用 DeepSeek provider
            print("📋 步骤 3: 创建 DeepSeek LLM 客户端...")
            try:
                from src.libs.llm.deepseek_llm import DeepSeekLLM
                
                llm = DeepSeekLLM(
                    settings=settings,
                    api_key=api_key,
                    base_url=base_url,
                )
                print("   ✓ DeepSeek LLM 客户端创建成功")
                print()
            except Exception as e:
                print(f"   ❌ DeepSeek LLM 客户端创建失败: {e}")
                return False
        else:
            # 使用工厂模式创建
            print("📋 步骤 2: 使用 LLM Factory 创建客户端...")
            try:
                llm = LLMFactory.create(settings)
                print("   ✓ LLM 客户端创建成功")
                print()
            except Exception as e:
                print(f"   ❌ LLM 客户端创建失败: {e}")
                return False
        
        # 测试 API 调用
        print("📋 步骤 4: 测试 API 连接...")
        try:
            test_message = Message(role="user", content="你好，请回复'连接成功'")
            print(f"   发送测试消息: {test_message.content}")
            
            response = llm.chat([test_message])
            
            print("   ✓ API 调用成功！")
            print(f"   - 响应内容: {response.content[:100]}...")
            print(f"   - 模型: {response.model}")
            if hasattr(response, 'usage') and response.usage:
                print(f"   - Token 使用: {response.usage}")
            print()
            
            print("=" * 60)
            print("✅ DeepSeek API 连接状态: 正常")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"   ❌ API 调用失败: {e}")
            print()
            print("=" * 60)
            print("❌ DeepSeek API 连接状态: 失败")
            print("=" * 60)
            print()
            print("可能的原因:")
            print("1. API Key 无效或已过期")
            print("2. 网络连接问题")
            print("3. 代理服务器问题（如果使用智增增代理）")
            print("4. DeepSeek API 服务暂时不可用")
            return False
            
    except Exception as e:
        print(f"❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = check_deepseek_connection()
    sys.exit(0 if success else 1)

