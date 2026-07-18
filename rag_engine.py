"""
RAG 引擎：负责文档加载、切分、索引、检索和问答
Advanced RAG 引擎：负责文档加载、切分、混合检索、重排序和问答
"""
import os
import logging
from pathlib import Path
import hashlib
import pandas as pd

from typing import List
import uuid



from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma 
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,  # 🌟 新增：Word 文档加载器
    DirectoryLoader,
)
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 🌟 新增：Advanced RAG 核心组件

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# 1. 多查询检索器
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
# 2. 混合检索器 & 上下文压缩检索器
from langchain_classic.retrievers import (
    EnsembleRetriever, 
    ContextualCompressionRetriever
)
# 3. 交叉编码器重排序器
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# 🌟 纯 LCEL 核心组件 (完全基于 langchain-core)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 记忆与历史
from langchain_community.chat_message_histories import ChatMessageHistory
# Agent 相关 (LangGraph)

from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent # 🌟 1. 新的导入路径
from langchain.agents.middleware import ModelCallLimitMiddleware # 🌟 2. 引入生产级中间件
from langgraph.checkpoint.memory import MemorySaver # 或 InMemorySaver



# 🌟 联网搜索相关
# ✅ 换成这个（它返回结构化的 list[dict]）
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document

import config
logger = logging.getLogger(__name__)

# 🌐 联网搜索 - 多引擎支持
# Tavily
try:
    # from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


# DuckDuckGo (保留作为备选)
try:
    # from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False


# 🌟 核心：自定义一个带“清洗功能”的安全 Python 工具
# ==========================================
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
python_repl = PythonREPL()

@tool
def safe_python_repl(query: str) -> str:
    """
    用于执行 Python 代码来分析 CSV/Excel 数据。
    输入必须是合法的 Python 代码。支持多行代码。
    可用变量: df (当前加载的数据框)
    """
    # 🌟 核心修复：清洗大模型生成的错误转义符！
    # 把字面量 \\n 替换为真正的换行符 \n
    clean_code = query.replace("\\n", "\n")
    clean_code = clean_code.replace("\\\\", "\\")
    
    try:
        # 将 df 注入到 REPL 的全局变量中
        result = python_repl.run(clean_code, globals={"df": current_df, "pd": pd})
        # 截断过长的输出，防止 Token 爆炸
        return result[:3000] if len(result) > 3000 else result
    except Exception as e:
        return f"执行出错: {str(e)}"

# 用于在 Tool 中引用当前的 DataFrame
current_df = None


class RAGEngine:
    """个人知识库 RAG 引擎"""
    
    def __init__(self):
        # 1.初始化 LLM
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.1,  # 🔥 Advanced RAG 建议降低温度，让模型更忠实于检索内容.在 Advanced RAG 中，我们已经通过 Reranker 保证了喂给大模型的上下文是极度精准的
        )
        
        # 2.初始化 Embedding 模型
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            check_embedding_ctx_length=False,  # 🔥 关键！禁止 LangChain 预分词，直接发送原始字符串
            chunk_size=10,                     # 🔥 控制每次发送的文本块数量，防止单次请求超限
        )
        
        # 3.初始化向量数据库与检索器占位
        self.vectorstore = None
        self.all_chunks = []       # 🔥 新增：保存所有文档块，用于 BM25
        # （多查询）检索器
        # self.retriever = None
        self.advanced_retriever = None  # 🔥 改名：现在是高级检索器
        self.rag_chain = None

        # 🌟 新增：CSV 数据分析模块
        self.dataframes = {}  # 存储上传的 CSV: {"filename.csv": pd.DataFrame}
        self.data_agent = None # 数据分析 Agent
        
        # 4. 文档切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""],
        )
        
        # 5. 构建 RAG Chain 的提示词
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT),
            ("user", "{question}"),
        ])
        
        # 6. 🌟 新增：预加载 Reranker 模型 (Cross-Encoder)
        # 首次运行会自动下载模型(约1.1G)，后续会走本地缓存:C:\Users\yiquan\.cache\huggingface\hub\models--BAAI--bge-reranker-base
        logger.info("⏳ 正在加载 Reranker 重排序模型 (BAAI/bge-reranker-base)...")
        try:
            # 这里使用 base 版本，体积更小，速度更快。如果需要极致精度可换 bge-reranker-v2-m3
            self.cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            logger.info("✅ Reranker 模型加载完成")
        except Exception as e:
            logger.error(f"❌ Reranker 加载失败: {e}。将降级为基础 RAG。")
            self.cross_encoder = None
            
        # 尝试加载已有索引
        self._try_load_existing_index()

        # 🌟 新版 RAG 记忆：使用字典存储不同 session 的历史
        self.qa_store = {}
        
        # 🌟 新版 Agent 记忆：使用 LangGraph 的 MemorySaver
        self.agent_checkpointer = MemorySaver()
        self.data_agent = None
        
        # 用于区分不同会话的 ID
        self.qa_session_id = str(uuid.uuid4())
        self.data_session_id = str(uuid.uuid4())

        # 🌟 联网搜索配置
        self.web_search_enabled = True  # 全局开关
        self.relevance_threshold = 0.3  # 🌟 Reranker 相关性阈值 (低于此值触发联网)

        # 🌐 联网搜索初始化 (多引擎支持)
        # ==========================================
        self.web_search_enabled = True
        self.relevance_threshold = 0.3  # bge-reranker-base 建议阈值
        self.web_search_tool = None
        self.search_provider = config.SEARCH_PROVIDER  # 从配置读取
        
        self._init_web_search_tool()



    def _init_web_search_tool(self):
        """
        🌐 根据配置初始化对应的搜索引擎
        
        支持的搜索引擎:
        - tavily: 专为 AI 设计，返回清洗后的内容 (推荐)
        - bing: 微软必应搜索，中文质量最高
        - duckduckgo: 免费备选
        """
        provider = self.search_provider.lower()
        
        if provider == "tavily":
            if not TAVILY_AVAILABLE:
                logger.error("❌ Tavily 未安装！请执行: pip install tavily-python langchain-community")
                self.web_search_enabled = False
                return
            
            # 改成了.env 文件写 TAVILY_API_KEY=tvly-xxx , config.py 用 os.getenv("TAVILY_API_KEY")
            # api_key = getattr(config, 'TAVILY_API_KEY', '')
            # logger.info(f"秘钥: {api_key}")
            try:
                self.web_search_tool = TavilySearch(
                    max_results=10,           # 🌟 从 5 改成 10，召回更多结果
                    search_depth="advanced",  # 🌟 深度搜索，质量更高 (消耗更多 API 额度)
                    include_answer=True,       # 🌟 让 Tavily 直接生成一个 AI 摘要答案
                    include_raw_content=False, # 不返回原始网页内容 (节省 Token)
                    include_images=False,      # 不返回图片
                    # tavily_api_key=api_key,  # 🌟 直接告诉它！不需要绕道环境变量
                )
                logger.info("✅ Tavily 联网搜索引擎初始化成功 🚀")
            except Exception as e:
                logger.error(f"❌ Tavily 初始化失败: {e}")
                self.web_search_enabled = False
        
        elif provider == "bing":
            # Bing 搜索使用自定义实现 (见下方 _bing_search 方法)
            subscription_key = getattr(config, 'BING_SUBSCRIPTION_KEY', '')
            if not subscription_key:
                logger.error("❌ 请在 config.py 中设置 BING_SUBSCRIPTION_KEY")
                self.web_search_enabled = False
                return
            
            self.web_search_tool = "bing"  # 标记使用 Bing
            logger.info("✅ Bing 联网搜索引擎初始化成功 🚀")
        
        elif provider == "duckduckgo":
            if not DDG_AVAILABLE:
                logger.error("❌ DuckDuckGo 未安装！请执行: pip install duckduckgo-search")
                self.web_search_enabled = False
                return
            
            try:
                self.web_search_tool = DuckDuckGoSearchAPIWrapper(
                    max_results=10,   # 🌟 从 5 改成 10，召回更多结果
                    region="cn-zh",
                    backend="text",
                )
                logger.info("✅ DuckDuckGo 联网搜索引擎初始化成功 (免费备选)")
            except Exception as e:
                logger.error(f"❌ DuckDuckGo 初始化失败: {e}")
                self.web_search_enabled = False
        
        else:
            logger.warning(f"⚠️ 未知的搜索引擎: {provider}，联网搜索已禁用")
            self.web_search_enabled = False
    
    def switch_search_provider(self, provider: str):
        """
        🔄 运行时切换搜索引擎
        
        Args:
            provider: "tavily" | "bing" | "duckduckgo"
        """
        self.search_provider = provider
        self._init_web_search_tool()
        return f"✅ 搜索引擎已切换为: {provider.upper()}"
       

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """将检索到的文档列表格式化为纯文本字符串"""
        return "\n\n".join(doc.page_content for doc in docs)

    def chat_with_memory(self, question: str) -> dict:
        
        """非流式带记忆问答 (直接复用 self.rag_chain)"""
        if not self.rag_chain:
            return {"answer": "⚠️ 知识库未初始化！", "sources": []}
        
        try:
            # 1. 定义 System Prompt
            system_prompt = (
                "你是一个专业的知识库问答助手。请使用以下检索到的上下文来回答用户的问题。\n"
                "如果你不知道答案，就说你不知道，不要编造。\n"
                "保持回答简洁专业。\n\n"
                "上下文:\n{context}"
            )
            # 1. 获取历史
            history = self._get_qa_session_history(self.qa_session_id)
            
            # 2. 调用纯净的 Chain (手动传入 history)
            answer_text = self.rag_chain.invoke({
                "question": question,
                "chat_history": history.messages
            })
            # 3. 🌟 手动保存记忆 (因为 Chain 是纯净的，所以这里存一次刚刚好)
            history.add_user_message(question)
            history.add_ai_message(answer_text)
            
            # 4. 滑动窗口裁剪 (保留最近 25 轮)
            if len(history.messages) > 50:
                history.messages = history.messages[-50:]

            # 5. 获取来源
            source_docs = self.advanced_retriever.invoke(question)
            sources = [{"content": d.page_content[:100], "source": d.metadata.get("source", "未知")} for d in source_docs]
            
            return {"answer": answer_text, "sources": sources}
        except Exception as e:
            logger.error(f"❌ 问答失败: {e}")
            return {"answer": f"回答失败: {str(e)}", "sources": []}
        

    def clear_qa_memory(self):
        """清空知识问答记忆 (重置 session)"""
        self.qa_session_id = str(uuid.uuid4()) # 生成新 ID 即可变相清空
        return "✅ 对话记忆已清空！"
    
    def add_to_qa_memory(self, user_message: str, ai_message: str):
        """
        🌟 手动将一轮对话添加到 RAG 记忆中 ((专供 query_stream 流式输出后调用))
        """
        history = self._get_qa_session_history(self.qa_session_id)
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)
        
        # 🌟 滑动窗口控制：只保留最近 25 轮 (50 条消息)，防止 Token 爆炸
        # 因为 ChatMessageHistory 没有自带 k=25 功能，我们手动裁剪
        if len(history.messages) > 50:
            history.messages = history.messages[-50:]

    
    def _build_advanced_retriever(self):
        """
        🌟 核心改造：组装 Advanced 检索器
        流程：向量检索 + BM25 → 混合检索(Ensemble) → 多查询改写(MultiQuery) → 重排序(Reranker)
        如何验证 Advanced RAG 真的生效了:
        在 Gradio 界面提问一个包含生僻专有名词的问题（比如你们公司内部的某个项目代号），观察 get_sources 返回的参考文档。
        如果是基础 RAG，它大概率会搜偏；换成 Advanced RAG 后，你会发现 BM25 会死死咬住那个专有名词，Reranker 会把最准的那条排在第一位
        """
        if not self.vectorstore or not self.all_chunks:
            logger.warning("⚠️ 向量库或文档块为空，无法构建高级检索器")
            return

        # Step A: 基础向量检索器 (召回 Top 15 供后续精排),底层检索器各召回 15 条
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        
        # Step B: BM25 关键词检索器 (召回 Top 15)
        logger.info("🔧 正在构建 BM25 索引...")
        bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
        bm25_retriever.k = 15
        
        # Step C: 🌟 混合检索 (Ensemble) - 权重可调
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]  # 语义占 60%，关键词占 40%
        )
        logger.info("✅ 混合检索器 (BM25 + Vector) 构建完成")

        # Step D: 🌟 查询改写 (Multi-Query)
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever, 
            llm=self.llm
        )
        logger.info("✅ 多查询改写器构建完成")

        # Step E: 🌟 重排序 (Reranker)
        if self.cross_encoder:
            compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=config.TOP_K)
            self.advanced_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=multi_retriever
            )
            logger.info(f"✅ 重排序器构建完成 (最终保留 Top {config.TOP_K})")
        else:
            # 降级处理
            self.advanced_retriever = multi_retriever
            logger.warning("⚠️ Reranker 不可用，已降级为 Multi-Query 检索")    
    
    def _try_load_existing_index(self):
        """尝试加载已有的向量数据库，并重建 BM25 所需的 chunks"""
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            try:
                self.vectorstore = Chroma(
                    persist_directory=config.CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings,
                    collection_name=config.COLLECTION_NAME,
                )
                # 检查是否有数据
                count = self.vectorstore._collection.count()
                if count > 0:
                    logger.info(f"✅ 加载已有向量库，包含 {count} 个文档块")
                    # 🔥 关键：从 Chroma 反向提取所有文档，重建 all_chunks 供 BM25 使用
                    raw_data = self.vectorstore._collection.get(include=["documents", "metadatas"])
                    self.all_chunks = [
                        Document(page_content=doc, metadata=meta)
                        for doc, meta in zip(raw_data["documents"], raw_data["metadatas"])
                    ]
                    logger.info(f"✅ 已从向量库反向恢复 {len(self.all_chunks)} 个文档块用于 BM25")

                    # 构建高级检索器和 Chain
                    self._build_advanced_retriever()
                    self._build_chain()

                else:
                    self.vectorstore = None
            except Exception as e:
                logger.warning(f"⚠️ 加载已有索引失败: {e}")
                self.vectorstore = None
    
    def _build_chain(self):
        """构建 带记忆的 RAG LCEL Chain (不带自动记忆包装) """
        if not self.advanced_retriever:
            return
        
        # 🌟 1. 重构 Prompt：注入 MessagesPlaceholder
        # 注意：这里 config.SYSTEM_PROMPT  是系统提示词字符串
        self.rag_prompt_with_history = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT + "\n\n请参考以下上下文信息:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"), # 🌟 注入历史对话
            ("human", "{question}"),
        ])

        # 🌟 2. 构建纯洁的基础 RAG Chain (纯 LCEL) (不包裹 RunnableWithMessageHistory)
        # 这样 Chain 就只负责“根据输入生成输出”，不负责“偷偷存记忆”
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.advanced_retriever.invoke(x["question"]))
            )

            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("✅ 纯净版 RAG Chain 构建完成")

    
    def _get_qa_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取 RAG 问答的 session 历史"""
        if session_id not in self.qa_store:
            self.qa_store[session_id] = ChatMessageHistory()
        return self.qa_store[session_id] 
    

    def _should_use_web_search(self, question: str) -> tuple[bool, list[Document]]:
        """
        🌟 路由决策：判断是否需要联网搜索
        
        原理：
        1. 先执行 Advanced RAG 检索（包含 Reranker）
        2. 检查 Reranker 返回的最高分文档的相关性得分
        3. 如果最高分 < threshold，说明知识库中没有相关内容，触发联网
        
        返回: (是否需要联网, 本地检索到的文档)
        """
        # if not self.web_search_enabled or not self.advanced_retriever:
        #     return False, [] # <--- 💥 致命陷阱在这里,结果：use_web = False，且 local_docs = []（空列表）
        
        # 🚨 检查本地检索器是否存在（如果没有本地库，那只能看联网开关了）
        if not self.advanced_retriever:
            logger.warning("⚠️ 无本地知识库检索器")
            if self.web_search_enabled:
                return True, []  # 没本地库，但开了联网，去联网
            else:
                return False, [] # 没本地库，也没联网，彻底没招了
            
        # 🌟 核心修复：无论是否开启联网，先强制执行一次本地检索！
        try:
            local_docs = self.advanced_retriever.invoke(question)
            logger.info(f"🔍 本地检索返回文档数: {len(local_docs)}") # 看看是 0 还是大于 0
        except Exception as e:
            logger.error(f"❌ 本地检索异常: {e}")
            local_docs = []


        # 决策分支 1：用户在前端【关闭】了联网搜索
        if not self.web_search_enabled:
            logger.info("🔒 联网搜索已禁用，强制仅使用本地知识库")
            # 直接返回 False (不联网) 和 本地检索到的文档 (可能为空，但尽力了)
            return False, local_docs
        
        # 决策分支 2：用户【开启】了联网搜索，根据 Reranker 分数决定
        if not local_docs:
            logger.info("🌐 知识库无相关文档，触发联网搜索")
            return True, []
        
        # 获取 Reranker 最高分
        # 🌟 关键：检查 Reranker 的相关性得分
        # Cross-Encoder (bge-reranker) 的输出分数通常在 -10 ~ 10 之间
        # 经过实际测试，阈值设为 0.0 ~ 1.0 之间比较合理
        # 如果你的 Reranker 是 bge-reranker-base，建议阈值 0.0
        top_score = self._get_reranker_score(question, local_docs[0])
        # 🌟 新增 如果打分失败（返回负数），直接触发联网
        if top_score < 0:
            logger.info("🌐 Reranker 打分异常，安全起见触发联网搜索")
            return True, local_docs
        logger.info(f"📊 Reranker 最高分: {top_score:.4f} (阈值: {self.relevance_threshold})")

        if top_score < self.relevance_threshold:
            logger.info(f"🌐 知识库相关性不足 ({top_score:.4f} < {self.relevance_threshold})，触发联网搜索")
            return True, local_docs  # 即使触发联网，也返回本地文档作为补充
        logger.info(f"📚 知识库命中 ({top_score:.4f} >= {self.relevance_threshold})，使用本地知识")
        return False, local_docs


    def _get_reranker_score(self, question: str, doc: Document) -> float:
        """
        获取 Reranker 对单个文档的打分
        使用 CrossEncoder 直接计算 query-document 相关性
        """
        if not self.cross_encoder:
            # 如果没有 Reranker，返回负数触发联网搜索,或者给一个默认中等分数（不触发联网）
            # 没有 Reranker 时，应该返回一个会触发联网的分数
            return -1.0 # 而不是 0.5
        
        try:
            # 🌟 关键修复：langchain 的 HuggingFaceCrossEncoder 使用 score 方法  不是predict方法
            text_pairs = [(question, doc.page_content)]
            scores = self.cross_encoder.score(text_pairs)
            # scores 是一个列表，取第一个元素
            return float(scores[0])
        except Exception as e:
            logger.warning(f"⚠️ Reranker 打分失败: {e}")
            # 打分失败时返回负数，触发联网搜索
            # # 打分失败时，应该触发联网搜索，而不是假装命中
            return -1.0   # 而不是 0.5  



    ##=================  联网搜索引擎  ========================
    def web_search(self, query: str) -> list[dict]:
        """
        🌐 统一联网搜索入口 (自动路由到对应搜索引擎)
        
        返回统一格式: [{"title": "xxx", "snippet": "xxx", "url": "xxx"}, ...]
        """
        if not self.web_search_enabled or not self.web_search_tool:
            return []
        
        provider = self.search_provider.lower()
        
        try:
            if provider == "tavily":
                return self._tavily_search(query)
            elif provider == "bing":
                return self._bing_search(query)
            elif provider == "duckduckgo":
                return self._duckduckgo_search(query)
            else:
                logger.error(f"❌ 未知搜索引擎: {provider}")
                return []
        except Exception as e:
            logger.error(f"❌ 联网搜索失败 ({provider}): {e}")
            return []
    
    def _tavily_search(self, query: str) -> list[dict]:
        """
        🌟 Tavily 搜索实现
        
        Tavily 的核心优势：
        1. include_answer=True 时，会返回一个 AI 生成的摘要答案
        2. 返回的 content 是清洗过的核心文本，可直接喂给 LLM
        3. search_depth="advanced" 会进行更深度的内容提取

        4. 增加 Tavily 搜索实现 (增强版：支持 AI 摘要答案)
        """
        logger.info(f"🔍 [Tavily] 正在搜索: {query}")
        
        from tavily import TavilyClient

        # 🌟 使用 TavilyClient 直接调用，可以获取 answer
        # client = TavilyClient(api_key=config.TAVILY_API_KEY) # 使用.env文件传入，这里不需要再次传入
        client = TavilyClient()
        response = client.search(
            query=query,
            max_results=10,             # 🌟 从 5 改成 10，召回更多结果
            search_depth="advanced",
            include_answer=True,        # 🌟 获取 AI 摘要答案
            include_raw_content=False,
            include_images=False,
            # 🌟 加上这个参数，让 Tavily 搜索更多相关主题
            topic="general",
        )
        raw_results = response.get("results", [])
        logger.info(f"🔍 Tavily 原始返回数量: {len(raw_results)}") # 🌟 打印原始数量
        
        # 🌟 保存 AI 摘要答案 (供前端展示)
        self._tavily_answer = response.get("answer", "")
        
        formatted = []
        for r in response.get("results", []):
            formatted.append({
                "title": r.get("title", "无标题"),
                "snippet": r.get("content", "无摘要"),  # 🌟 Tavily 的 content 是清洗过的核心内容
                "url": r.get("url", ""),
                "source": "🌐 Tavily 搜索",
                "score": r.get("score", 0),  # 🌟 Tavily 提供相关性分数
            })
        
        logger.info(f"✅ [Tavily] 搜索完成，获取 {len(formatted)} 条结果")
        if self._tavily_answer:
            logger.info(f"💡 [Tavily] AI 摘要: {self._tavily_answer[:100]}...")

        return formatted
    

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        """清理 HTML 标签"""
        import re
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
    
    def _bing_search(self, query: str) -> list[dict]:
        """
        🌟 Bing Search API 搜索实现
        
        Bing 的核心优势：
        1. 微软官方搜索索引，质量最高
        2. 中文搜索效果远优于 DuckDuckGo
        3. 返回结构化数据 (网页摘要、日期、语言等)
        
        API 文档: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/
        """
        import requests
        
        logger.info(f"🔍 [Bing] 正在搜索: {query}")
        
        subscription_key = config.BING_SUBSCRIPTION_KEY
        search_url = config.BING_SEARCH_URL
        
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        params = {
            "q": query,
            "count": 5,               # 返回结果数量
            "textDecorations": True,   # 启用文本装饰 (高亮匹配词)
            "textFormat": "HTML",      # 返回 HTML 格式
            "mkt": "zh-CN",           # 🌟 中国中文市场 (返回中文结果)
            "setLang": "zh-Hans",     # 简体中文
        }
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            formatted = []
            
            # 🌟 Bing 的搜索结果在 webPages.value 中
            web_pages = data.get("webPages", {}).get("value", [])
            
            for page in web_pages:
                # 清理 HTML 标签
                snippet = page.get("snippet", "")
                snippet = self._strip_html_tags(snippet)
                
                formatted.append({
                    "title": self._strip_html_tags(page.get("name", "无标题")),
                    "snippet": snippet,
                    "url": page.get("url", ""),
                    "source": "🌐 Bing 搜索",
                    "date": page.get("dateLastCrawled", ""),  # 🌟 Bing 提供网页最后更新时间
                    "language": page.get("language", ""),
                })
            
            logger.info(f"✅ [Bing] 搜索完成，获取 {len(formatted)} 条结果")
            return formatted
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("❌ [Bing] API Key 无效或已过期！请检查 config.py")
            elif e.response.status_code == 403:
                logger.error("❌ [Bing] 免费额度已用完 (1000次/月)")
            elif e.response.status_code == 429:
                logger.error("❌ [Bing] 请求过于频繁 (限制 3次/秒)")
            else:
                logger.error(f"❌ [Bing] HTTP 错误: {e}")
            return []
        except requests.exceptions.Timeout:
            logger.error("❌ [Bing] 请求超时 (10秒)")
            return []
        except Exception as e:
            logger.error(f"❌ [Bing] 搜索异常: {e}")
            return []
    

    def _duckduckgo_search(self, query: str) -> list[dict]:
        """DuckDuckGo 搜索实现 (免费备选)"""
        logger.info(f"🔍 [DuckDuckGo] 正在搜索: {query}")
        
        # 🌟 关键修复：DuckDuckGoSearchAPIWrapper.results() 直接返回 list[dict]
        # 每个 dict 包含: title, link, snippet
        raw_results = self.web_search_tool.results(query, max_results=10)
        
        
        # 只需要检查是否为空列表
        if not raw_results:
            logger.warning("⚠️ 联网搜索返回空结果")
            return []
        
        logger.info(f"🔍 原始搜索结果类型: {type(raw_results)}, 数量: {len(raw_results)}")

        formatted = []
        for r in raw_results:
            formatted.append({
                "title": r.get("title", "无标题"),
                "snippet": r.get("snippet", r.get("body", "无摘要")),
                "url": r.get("link", r.get("url", "")),
                "source": "🌐 DuckDuckGo 搜索",
            })
        
        logger.info(f"✅ [DuckDuckGo] 搜索完成，获取 {len(formatted)} 条结果")
        return formatted

    def _web_results_to_context(self, results: list[dict]) -> str:
        """将联网搜索结果格式化为 LLM 可读的上下文"""
        if not results:
            return ""
        
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"### [联网来源 {i}] 🌐 {r['title']}\n"
                f"链接: {r['url']}\n"
                f"内容: {r['snippet']}"
            )
        return "\n\n---\n\n".join(formatted)    
        


    def _format_docs(self, docs: list[Document]) -> str:
        """格式化检索到的文档"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            filename = Path(source).name if source else "未知"
            formatted.append(
                f"### [参考资料 {i}] 📄 {filename}\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)
    
    
    def load_documents(self, docs_dir: str = None) -> dict:
        """
        加载并索引文档
        返回：加载统计信息
        """
        docs_dir = docs_dir or config.DOCS_DIR
        
        if not os.path.exists(docs_dir):
            return {"error": f"文档目录不存在: {docs_dir}"}
        
        all_docs = []
        
        # 加载 TXT 文件
        txt_loader = DirectoryLoader(
            docs_dir, glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        
        # 加载 PDF 文件
        pdf_loader = DirectoryLoader(
            docs_dir, glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        
        # 加载 Markdown 文件
        md_loader = DirectoryLoader(
            docs_dir, glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        # 🌟 新增：加载 Word (.docx) 文件
        docx_loader = DirectoryLoader(
            docs_dir, 
            glob="**/*.docx", 
            loader_cls=Docx2txtLoader, 
            show_progress=True,
            # 如果遇到损坏的 docx 文件，跳过而不是让整个程序崩溃
            use_multithreading=True, 
        )
        
        txt_loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)
        pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        md_loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)
        # docx_loader = DirectoryLoader(docs_dir, glob="**/*.docx", loader_cls=Docx2txtLoader,loader_kwargs={"encoding": "utf-8"},show_progress=True)

        for loader in [txt_loader, pdf_loader, md_loader,docx_loader]:
            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"⚠️ 加载失败: {e}")
        
        if not all_docs:
            return {"error": "未找到任何文档（支持 .txt, .pdf, .md,.docx）"}
        
        # 切分文档
        chunks = self.splitter.split_documents(all_docs) #  🔥 存入实例变量
        
        # 存入向量数据库
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        
        # 🌟 构建高级检索器和 Chain
        self._build_advanced_retriever()
        self._build_chain()
        
        stats = {
            "文件数": len(all_docs),
            "文本块数": len(chunks),
            "状态": "✅ Advanced RAG 索引构建成功",
        }
        logger.info(f"✅ 索引完成: {stats}")
        return stats
    





    def load_dataframe(self, file_path: str) -> dict:
        """
        加载 CSV 文件到内存，并初始化数据分析 Agent
        🌟 通用表格数据加载器：支持 CSV (.csv) 和 Excel (.xlsx, .xls)
        将数据加载到内存，并初始化 Pandas Data Agent
        """
        global current_df # 引用全局变量
        path = Path(file_path)
        
        # 🌟 移除或放宽后缀检查，因为 Gradio 临时文件可能没有 .csv 后缀
        if not path.exists():
            return {"error": f"文件不存在: 文件路径为:{file_path}"}
        
        suffix = path.suffix.lower()
        if suffix not in [".csv", ".xlsx", ".xls"]:
            return {"error": "不支持的文件格式，请上传 CSV 或 Excel 文件"}
        try:
            # 1. 读取 CSV
            df = None
            # 常见的 CSV 编码顺序：UTF-8 (最通用), GBK (中文 Windows Excel 默认), latin1 (兜底，绝对不会报错)
            encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1'] 
            
            for enc in encodings_to_try:
                try:
                    if suffix == ".csv":
                    # 尝试使用当前编码读取
                        df = pd.read_csv(file_path, encoding=enc)
                        logger.info(f"✅ 成功使用 '{enc}' 编码读取 CSV: {path.name}")
                        file_type_desc = f"CSV (编码: {enc})"  # ✅ 加上这一行
                        break # 读取成功，跳出循环
                    else:
                        # 🌟 读取 Excel：默认读取第一个 Sheet (sheet_name=0)
                        # 如果需要读取特定 Sheet，后续可以扩展参数
                        df = pd.read_excel(file_path, sheet_name=0)
                        file_type_desc = f"Excel (Sheet: {df.name if hasattr(df, 'name') else 'Sheet1'})"
                except UnicodeDecodeError:
                    # 如果解码失败，继续尝试下一个编码
                    continue
                except Exception as e:
                    # 如果是其他错误（比如根本不是 CSV 格式），直接抛出
                    logger.error(f"❌ 读取 CSV 时发生非编码错误: {e}")
                    return {"error": f"文件内容解析失败 (可能不是有效的 CSV 格式): {str(e)}"}
            

            # 如果所有编码都失败了（理论上 latin1 不会失败，但以防万一）
            if df is None:
                return {"error": "无法解析文件编码(确保是：UTF-8 或 GBK,gb2312, latin1)。"}

            # 2. 基础数据清洗 (可选：去除全空的行列)
            df.dropna(how='all', inplace=True) 
            df.dropna(axis=1, how='all', inplace=True)

            # 3. 存入内存字典
            self.dataframes[path.name] = df
            current_df = df  # 🌟 让 Tool 能够访问最新的 df

            # 🌟 4. 核心改造：使用 LangGraph 构建现代 Agent
            # 构建系统提示词 (包含数据概览，让 LLM 知道 df 长什么样)
            rows, cols = df.shape
            columns_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
            df_head = df.head(3).to_string()


            # 🌟 5. system_prompt 直接接受纯字符串 (不再需要 SystemMessage 包装)
            system_prompt_text = (
                "你是一个专业的数据分析师。你可以使用 Pandas 对提供的 DataFrame (`df`) 进行数据分析。\n"
                "请使用中文回答。在编写 Python 代码时，请直接使用变量 `df`，无需重新读取文件。\n"
                "【重要代码格式警告】：当你调用工具执行代码时，必须确保代码中的换行符是真正的换行符，"
                "绝对不要输出字面量 '\\n' (反斜杠+n)。\n\n"
                f"当前数据概览:\n"
                f"- 行数: {rows}, 列数: {cols}\n"
                f"- 列名及类型: {columns_info}\n"
                f"- 前3行预览:\n{df_head}"
            )

            # 初始化 LangGraph Checkpointer (现在它真的生效了！)
            if not hasattr(self, 'agent_checkpointer') or self.agent_checkpointer is None:
                self.agent_checkpointer = MemorySaver()

            # 🌟 6. 构建生产级 Agent (引入 Middleware 中间件)
            self.data_agent = create_agent(
                model=self.llm,
                tools=[safe_python_repl],
                system_prompt=system_prompt_text, # 🌟 新版参数：直接传字符串
                checkpointer=self.agent_checkpointer,
                middleware=[
                    # 🌟 生产级防护 1：限制最大模型调用次数，彻底杜绝死循环和 Token 爆炸！
                    # 如果 Agent 陷入“思考->报错->再思考”的死循环，达到 15 次后会强制中断并返回已有结果。
                    ModelCallLimitMiddleware(run_limit=15),
                    
                    # 🌟 生产级防护 2 (可选)：如果你希望长对话自动压缩，可以取消下面这行的注释
                    # SummarizationMiddleware(max_tokens=4000), 
                ]
            )

            logger.info("✅ 数据分析 Agent (create_agent + Middleware) 构建完成")

            return {
                "status": "success",
                "filename": path.name,
                "file_type": file_type_desc,
                "rows": rows,
                "columns": cols,
                "columns_info": columns_info,
                "preview": df.head(3).to_markdown(index=False)
            }
        except Exception as e:
            logger.error(f"❌ 表格数据加载失败: {e}")
            return {"error": f"解析失败: {str(e)}"}
        

    def query_data(self, question: str) -> str:
        """
        针对 CSV 数据进行提问分析
        """
        if not self.data_agent:
            return "⚠️ 尚未加载数据！请先上传 CSV/Excel 文件。"
        
        try:
            # 调用 Agent 进行分析
            # 🌟 传入 thread_id (等同于 session_id)
            config = {"configurable": {"thread_id": self.data_session_id}}
            response = self.data_agent.invoke(
                {"input": question}, 
                config=config
            )
            return response["output"]
        except Exception as e:
            logger.error(f"❌ 数据分析失败: {e}")
            return f"分析失败: {str(e)}"
        

    def clear_data_memory(self):
        """清空数据分析记忆 (重置 thread_id)"""
        self.data_session_id = str(uuid.uuid4())
        return "✅ 分析记忆已清空！"
    


    
    # -------- ---------
    def query(self, question: str) -> str:
        """提问（非流式）"""
        if not self.rag_chain:
            return "⚠️ 知识库尚未构建索引！请先上传文档。"
        return self.rag_chain.invoke(question)
    
    def query_stream(self, question: str):
        """
        🌟 带记忆 + 联网增强的流式问答
        
        决策流程：
        1. 先判断知识库是否有相关内容
        2. 如果有 → 使用本地知识回答
        3. 如果没有 → 联网搜索，用搜索结果回答
        4. 混合模式 → 本地知识 + 联网结果一起喂给 LLM
        """
        if not self.advanced_retriever:
            yield "⚠️ 知识库尚未构建索引！请先上传文档。"
            return
        
        # 存储本次问答的元信息（供前端展示）
        self._last_query_meta = {
            "used_web_search": False,
            "web_results": [],
            "local_sources": [],
            "route_decision": ""
        }
        
        try:
            # 第二步：进入 query_stream
            # 🌟 Step 1: 路由决策
            use_web, local_docs = self._should_use_web_search(question)
            
            # 记录本地来源
            self._last_query_meta["local_sources"] = [
                {
                    "source": doc.metadata.get("source", "未知"),
                    "filename": Path(doc.metadata.get("source", "")).name,
                    "content_preview": doc.page_content[:200],
                }
                for doc in local_docs
            ]
            
            # 🌟 Step 2: 构建上下文
            context_parts = []
            # 组装上下文
            # local_docs 是空列表 []，在 Python 中 bool([]) 是 False！
            if local_docs and not use_web:
                # 情况 A: 纯本地知识
                context_parts.append("【知识库内容】\n" + self._format_docs(local_docs))
                self._last_query_meta["route_decision"] = "📚 使用知识库回答"
                
            elif use_web:
                # 情况 B/C: 需要联网搜索
                yield "🔍 *知识库未找到相关内容，正在联网搜索...*\n\n"
                
                web_results = self.web_search(question)
                # 🌟 如果使用 Tavily，显示 AI 快速摘要
                tavily_answer = getattr(self, '_tavily_answer', '')
                if tavily_answer and self.search_provider == "tavily":
                    yield f"💡 **Tavily AI 快速摘要**: {tavily_answer}\n\n"

                self._last_query_meta["used_web_search"] = True
                self._last_query_meta["web_results"] = web_results
                
                if web_results:
                    web_context = self._web_results_to_context(web_results)
                    context_parts.append("【联网搜索结果】\n" + web_context)
                    
                    # 如果本地也有一些相关文档，作为补充
                    if local_docs:
                        context_parts.append("【知识库补充内容】\n" + self._format_docs(local_docs))
                        self._last_query_meta["route_decision"] = "🌐 联网搜索 + 知识库补充"
                    else:
                        self._last_query_meta["route_decision"] = "🌐 纯联网搜索回答"
                else:
                    # 联网也失败了，降级为本地
                    if local_docs:
                        context_parts.append("【知识库内容（联网搜索失败，降级使用）】\n" + self._format_docs(local_docs))
                    else:
                        yield "😔 抱歉，知识库和联网搜索均未找到相关内容。请尝试换一种问法。"
                        return
            
            # 🌟 Step 3: 构建带联网上下文的 Prompt
            full_context = "\n\n".join(context_parts)
            
            # 使用专门的联网+知识库 Prompt
            web_aware_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 config.SYSTEM_PROMPT + 
                 "\n\n请参考以下上下文信息来回答用户的问题。\n"
                 "如果上下文中包含【联网搜索结果】，请在回答中注明信息来源链接。\n"
                 "如果知识库和联网搜索都没有相关信息，请诚实说明你不知道。\n\n"
                 "上下文:\n{context}"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])
            
            # 🌟 Step 4: 构建并执行 Chain
            temp_chain = (
                web_aware_prompt

                | self.llm
                | StrOutputParser()
            )
            
            # 获取历史
            history = self._get_qa_session_history(self.qa_session_id).messages
            
            for chunk in temp_chain.stream({
                "question": question,
                "context": full_context,
                "chat_history": history,
            }):
                yield chunk
                
        except Exception as e:
            logger.error(f"❌ 流式问答失败: {e}")
            yield f"\n\n❌ 回答出错: {str(e)}"

    # 供前端获取元信息
    def get_last_query_meta(self) -> dict:
        """获取上一次问答的路由决策和来源信息"""
        return getattr(self, '_last_query_meta', {
            "used_web_search": False,
            "web_results": [],
            "local_sources": [],
            "route_decision": "未知"
        })
    
    def get_sources(self, question: str) -> list[dict]:
        """获取问题相关的文档来源"""
        if not self.advanced_retriever:
            return []
        
        docs = self.advanced_retriever.invoke(question)
        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "未知"),
                "filename": Path(doc.metadata.get("source", "")).name,
                "content_preview": doc.page_content[:200],
            })
        return sources
    
    def get_index_info(self) -> dict:
        """获取当前索引信息"""
        if not self.vectorstore:
            return {"状态": "❌ 未构建索引"}
        
        try:
            count = self.vectorstore._collection.count()
            return {
                "状态": "✅ 已构建",
                "文档块数": count,
                "模型": config.CHAT_MODEL,
                "Embedding": config.EMBEDDING_MODEL,
            }
        except:
            return {"状态": "⚠️ 索引状态异常"}
        

    def _generate_chunk_id(self, source: str, index: int) -> str:
        """
        为每个文档块生成唯一 ID
        格式：source_hash + chunk_index
        例如：a1b2c3_0, a1b2c3_1, a1b2c3_2
        """
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{source_hash}_{index}"
    

    def list_documents(self) -> list[dict]:
        """
        列出知识库中所有已索引的文档（按文件名聚合）
        返回：[{"filename": "xxx.pdf", "chunks": 5, "source": "/path/xxx.pdf"}, ...]
        """
        if not self.vectorstore:
            return []
        
        try:
            # 从 Chroma 获取所有 metadata
            raw_data = self.vectorstore._collection.get(include=["metadatas"])
            metadatas = raw_data["metadatas"]
            
            # 按 source 聚合统计
            doc_stats = {}
            for meta in metadatas:
                source = meta.get("source", "未知")
                filename = Path(source).name if source else "未知"
                
                if source not in doc_stats:
                    doc_stats[source] = {
                        "filename": filename,
                        "source": source,
                        "chunks": 0,
                    }
                doc_stats[source]["chunks"] += 1
            
            return list(doc_stats.values())
        except Exception as e:
            logger.error(f"❌ 列出文档失败: {e}")
            return []
        

    def delete_document(self, source: str) -> dict:
        """
        从知识库中删除指定文档的所有文档块
        source: 文档的完整路径或文件名
        """
        if not self.vectorstore:
            return {"error": "向量库未初始化"}
        
        try:
            filename = Path(source).name
            
            # Step 1: 从 Chroma 中删除（按 metadata 中的 source 匹配）
            # 先查出所有匹配的 ID
            results = self.vectorstore._collection.get(
                where={"source": source},
                include=[]
            )
            ids_to_delete = results["ids"]
            
            if not ids_to_delete:
                # 尝试用文件名匹配
                all_data = self.vectorstore._collection.get(include=["metadatas"])
                ids_to_delete = [
                    id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
                    if Path(meta.get("source", "")).name == filename
                ]
            
            if not ids_to_delete:
                return {"error": f"未找到文档: {filename}"}
            
            # 执行删除
            self.vectorstore._collection.delete(ids=ids_to_delete)
            
            # Step 2: 从内存中的 all_chunks 也删除
            self.all_chunks = [
                doc for doc in self.all_chunks 
                if doc.metadata.get("source", "") != source 
                and Path(doc.metadata.get("source", "")).name != filename
            ]
            
            # Step 3: 重建检索器（因为 BM25 需要更新）
            remaining_count = self.vectorstore._collection.count()
            if remaining_count > 0:
                self._build_advanced_retriever()
                self._build_chain()
            else:
                self.advanced_retriever = None
                self.rag_chain = None
            
            logger.info(f"🗑️ 已删除 {filename} 的 {len(ids_to_delete)} 个文档块")
            return {
                "status": "success",
                "filename": filename,
                "deleted_chunks": len(ids_to_delete),
                "remaining_chunks": remaining_count,
            }
            
        except Exception as e:
            logger.error(f"❌ 删除文档失败: {e}")
            return {"error": str(e)}
        
    def add_documents(self, file_paths: list[str]) -> dict:
        """
        向已有知识库中追加新文档
        file_paths: 文件路径列表
        """
        if not self.embeddings:
            return {"error": "Embedding 模型未初始化"}
        
        all_new_docs = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                continue
            
            # 根据文件类型选择 Loader
            try:
                if path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(path))
                elif path.suffix.lower() in (".md", ".txt"):
                    loader = TextLoader(str(path), encoding="utf-8")
                # 🌟 新增：处理 Word 文档
                elif path.suffix.lower() == ".docx":
                    loader = Docx2txtLoader(str(path))
                else:
                    logger.warning(f"⚠️ 不支持的文件类型: {path.suffix}")
                    continue
                
                docs = loader.load()
                all_new_docs.extend(docs)
            except Exception as e:
                logger.warning(f"⚠️ 加载 {path.name} 失败: {e}")
        
        if not all_new_docs:
            return {"error": "没有成功加载任何文档"}
        
        # 切分文档
        new_chunks = self.splitter.split_documents(all_new_docs)
        
        # 为每个 chunk 生成唯一 ID
        chunk_ids = []
        for i, chunk in enumerate(new_chunks):
            source = chunk.metadata.get("source", "unknown")
            chunk_ids.append(self._generate_chunk_id(source, i))
        
        # 追加到向量库
        if self.vectorstore is None:
            # 首次建库
            self.vectorstore = Chroma.from_documents(
                documents=new_chunks,
                embedding=self.embeddings,
                ids=chunk_ids,
                collection_name=config.COLLECTION_NAME,
                persist_directory=config.CHROMA_PERSIST_DIR,
            )
        else:
            # 追加到已有库
            # ⚠️ 先删除同名文件（避免重复）
            for file_path in file_paths:
                source = str(Path(file_path).resolve())
                filename = Path(file_path).name
                try:
                    existing = self.vectorstore._collection.get(
                        where={"source": {"$contains": filename}},
                        include=[]
                    )
                    if existing["ids"]:
                        self.vectorstore._collection.delete(ids=existing["ids"])
                        logger.info(f"🔄 覆盖已有文件: {filename}")
                except:
                    pass
            
            # 添加新文档
            self.vectorstore.add_documents(documents=new_chunks, ids=chunk_ids)
        
        # 更新内存中的 chunks
        self.all_chunks.extend(new_chunks)
        
        # 重建检索器
        self._build_advanced_retriever()
        self._build_chain()
        
        stats = {
            "status": "success",
            "新增文件数": len(all_new_docs),
            "新增文本块数": len(new_chunks),
            "知识库总块数": self.vectorstore._collection.count(),
        }
        logger.info(f"✅ 追加文档完成: {stats}")
        return stats
    
    def clear_all(self) -> dict:
        """清空整个知识库"""
        try:
            import shutil
            if os.path.exists(config.CHROMA_PERSIST_DIR):
                shutil.rmtree(config.CHROMA_PERSIST_DIR)
            
            self.vectorstore = None
            self.all_chunks = []
            self.advanced_retriever = None
            self.rag_chain = None
            
            logger.info("🗑️ 知识库已清空")
            return {"status": "success", "message": "知识库已完全清空"}
        except Exception as e:
            return {"error": str(e)}

    def upload_files_to_temp(self, files) -> list[str]:
        """
        将 Gradio 上传的文件保存到临时目录
        files: Gradio File 组件返回的文件对象列表
        返回：保存后的文件路径列表
        """
        upload_dir = Path("./uploaded_docs")
        upload_dir.mkdir(exist_ok=True)
        
        saved_paths = []
        for file in files:
            # Gradio 上传的文件是一个临时路径
            src_path = Path(file.name if hasattr(file, 'name') else file)
            dest_path = upload_dir / src_path.name
            # 复制文件
            import shutil
            shutil.copy2(str(src_path), str(dest_path))
            saved_paths.append(str(dest_path))
        
        return saved_paths
    

    def debug_retrieval(self, question: str) -> dict:
        """
        为了让"检索透视"生效，先给 RAGEngine 加一个调试方法
        🔍 检索过程透视：返回检索链路每一步的详细结果
        用于在 Gradio 中可视化展示
        
        """
        if not self.vectorstore:
            return {"error": "知识库未构建"}
        
        result = {
            "original_query": question,
            "rewritten_queries": [],
            "vector_results": [],
            "bm25_results": [],
            "merged_results": [],
            "reranked_results": [],
        }
        
        try:
            # Step 1: 查询改写
            # from langchain.prompts import ChatPromptTemplate as CPT
            from langchain_core.prompts import ChatPromptTemplate as CPT
            rewrite_prompt = CPT.from_template(
                "你是一个查询改写专家。请将用户的问题改写为 3 个不同角度的搜索查询，每行一个，不要输出其他内容。\n\n用户问题: {question}"
            )
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            rewritten = rewrite_chain.invoke({"question": question})
            result["rewritten_queries"] = [q.strip() for q in rewritten.strip().split("\n") if q.strip()]
            
            # Step 2: 向量检索 (Top 10)
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            all_queries = [question] + result["rewritten_queries"]
            
            vector_docs = {}
            for q in all_queries:
                for doc in vector_retriever.invoke(q):
                    doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
                    if doc_id not in vector_docs:
                        vector_docs[doc_id] = doc
            result["vector_results"] = [
                {"content": d.page_content[:100], "source": Path(d.metadata.get("source", "")).name}
                for d in list(vector_docs.values())[:10]
            ]
            
            # Step 3: BM25 检索 (Top 10)
            if self.all_chunks:
                bm25 = BM25Retriever.from_documents(self.all_chunks)
                bm25.k = 10
                bm25_docs = bm25.invoke(question)
                result["bm25_results"] = [
                    {"content": d.page_content[:100], "source": Path(d.metadata.get("source", "")).name}
                    for d in bm25_docs
                ]
            
            # Step 4: 最终精排结果
            if self.advanced_retriever:
                final_docs = self.advanced_retriever.invoke(question)
                result["reranked_results"] = [
                    {
                        "rank": i + 1,
                        "content": d.page_content[:150],
                        "source": Path(d.metadata.get("source", "")).name,
                    }
                    for i, d in enumerate(final_docs)
                ]
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"检索调试失败: {e}")
    
        return result