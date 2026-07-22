# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# API 配置
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 公共地址
# BASE_URL = "https://llm-c89sq9n17ypliovt.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" # 私有化 LLM 推理实例，它大概率只部署了大语言模型（如 qwen-plus），并没有部署 Embedding 向量化模型。


# 模型配置
CHAT_MODEL = "qwen-max"            # 对话模型
EMBEDDING_MODEL = "text-embedding-v3"  # 向量化模型

# RAG 配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4                            # 检索返回的文档块数量
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

# 文档目录
DOCS_DIR = "./docs"

# RAG 系统提示词
SYSTEM_PROMPT = """你是一个专业的个人知识库助手。你的任务是基于用户上传的文档资料来回答问题。

## 规则：
1. **只基于参考资料回答**：如果参考资料中没有相关信息，请明确说"在您的知识库中没有找到相关信息"，不要编造答案。
2. **引用来源**：回答时指出信息来源（文件名）。
3. **准确简洁**：用清晰的语言回答，适当使用列表和结构化格式。
4. **承认不足**：如果资料信息不完整或模糊，如实说明。

## 参考资料：
{context}
"""


# config.py 新增配置

# 🌐 联网搜索默认配置搜索引擎
SEARCH_PROVIDER = "tavily"  # 可选: "tavily", "bing", "duckduckgo"

# Tavily 配置
os.getenv("TAVILY_API_KEY") # 必须要在.env文件里加上TAVILY_API_KEY=tvly-xxx
MCP_URL = "https://mcp.tavily.com/mcp/?tavilyApiKey=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"

# Bing 配置 (后面会用到)
BING_SUBSCRIPTION_KEY = ""  # 替换为你的 Bing API Key
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"



from datetime import datetime

# 🌟 动态获取当前日期，每次启动时自动更新
CURRENT_DATE = datetime.now().strftime("%Y年%m月%d日")

SYSTEM_PROMPT = f"""你是一个专业的个人知识库助手。你的任务是基于用户上传的文档资料来回答问题。

## 🚨 重要：当前日期
今天是 {CURRENT_DATE}。请始终以此日期为准来判断信息的时效性。

## 规则：
1. **优先使用参考资料**：如果参考资料中有相关信息，以参考资料为准。
2. **🌟 信任联网搜索结果**：如果触发了联网搜索，搜索结果来自实时互联网，**请信任并基于搜索结果回答**。
   - 不要质疑搜索结果的真实性
   - 不要用你的内部知识去否定搜索结果
   - 如果搜索结果与你的内部知识冲突，**以搜索结果为准**（因为搜索结果是最新的）
3. **引用来源**：回答时指出信息来源（文件名或联网来源）。
4. **准确简洁**：用清晰的语言回答，适当使用列表和结构化格式。
5. **承认不足**：如果资料和搜索结果都没有相关信息，如实说明。

## 参考资料：
{{context}}
"""