"""检查 LLM Arena 中 DeepSeek API 连接状态的脚本。

此脚本专门测试 LLM Arena 中配置的 api-deepseek-chat 模型连接状态。
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
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.base_llm import Message


def check_deepseek_arena_connection() -> None:
    """检查 LLM Arena 中 DeepSeek API 连接状态。"""
    print("=" * 60)
    print("LLM Arena - DeepSeek API 连接状态检查")
    print("=" * 60)
    print()
    
    try:
        # 加载配置
        print("步骤 1: 加载配置...")
        settings = load_settings()
        print(f"   ✓ 配置加载成功")
        print()
        
        # 获取 API Key 和 Base URL（从智增增代理）
        api_key = getattr(settings.llm, "api_key", None)
        base_url = "https://api.zhizengzeng.com/v1"  # LLM Arena 中使用的代理地址
        model_name = "deepseek-chat"  # LLM Arena 中使用的模型名称
        
        if not api_key:
            print("❌ 错误: 未找到 API Key")
            print("   请在 config/settings.yaml 中设置 llm.api_key")
            return False
        
        print("步骤 2: 创建 DeepSeek LLM 客户端（LLM Arena 配置）...")
        print(f"   - 模型名称: {model_name}")
        print(f"   - 代理地址: {base_url}")
        print(f"   - API Key: {'已设置' if api_key else '未设置'}")
        print()
        
        try:
            # 创建 OpenAI LLM 实例（通过智增增代理访问 DeepSeek）
            llm = OpenAILLM(
                settings=settings,
                model=model_name,
                base_url=base_url,
                api_key=api_key,
            )
            print("   ✓ LLM 客户端创建成功")
            print()
        except Exception as e:
            print(f"   ❌ LLM 客户端创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试 API 调用
        print("步骤 3: 测试 API 连接...")
        try:
            test_message = Message(role="user", content="你好，请回复'DeepSeek连接成功'")
            print(f"   发送测试消息: {test_message.content}")
            
            response = llm.chat([test_message])
            
            print("   ✓ API 调用成功！")
            print(f"   - 响应内容: {response.content}")
            print(f"   - 模型: {response.model}")
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"   - Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"   - Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"   - Total Tokens: {usage.get('total_tokens', 'N/A')}")
            print()
            
            print("=" * 60)
            print("✅ DeepSeek API (LLM Arena) 连接状态: 正常")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"   ❌ API 调用失败: {e}")
            print()
            import traceback
            traceback.print_exc()
            print()
            print("=" * 60)
            print("❌ DeepSeek API (LLM Arena) 连接状态: 失败")
            print("=" * 60)
            print()
            print("可能的原因:")
            print("1. API Key 无效或已过期")
            print("2. 网络连接问题")
            print("3. 智增增代理服务器问题")
            print("4. DeepSeek API 服务暂时不可用")
            print("5. 模型名称 'deepseek-chat' 在代理中不可用")
            return False
            
    except Exception as e:
        print(f"❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = check_deepseek_arena_connection()
    sys.exit(0 if success else 1)

