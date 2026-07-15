# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# API 配置
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 公共地址
# BASE_URL = "https://llm-c89sq9n17ypliovt.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" # 私有化 LLM 推理实例，它大概率只部署了大语言模型（如 qwen-plus），并没有部署 Embedding 向量化模型。


# 模型配置
CHAT_MODEL = "qwen-plus"            # 对话模型
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