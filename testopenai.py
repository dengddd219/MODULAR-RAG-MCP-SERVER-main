from openai import OpenAI
import os

# 使用代理地址（和 settings.yaml 里保持一致）
base_url = "https://api.zhizengzeng.com/v1"
api_key = os.environ.get("OPENAI_API_KEY")

print(f"Using API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'None'}")
print(f"Using Base URL: {base_url}")

client = OpenAI(api_key=api_key, base_url=base_url)

emb = client.embeddings.create(
    model="text-embedding-3-small",
    input="hello world"
)
print(f"✅ Success! Embedding dimension: {len(emb.data[0].embedding)}")