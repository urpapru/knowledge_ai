"""
RAG 引擎：负责文档加载、切分、索引、检索和问答
"""
import os
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma 
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


class RAGEngine:
    """个人知识库 RAG 引擎"""
    
    def __init__(self):
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.3,  # RAG 场景用低温度，确保准确性
        )
        
        # 初始化 Embedding 模型
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            check_embedding_ctx_length=False,  # 🔥 关键！禁止 LangChain 预分词，直接发送原始字符串
            chunk_size=10,                     # 🔥 控制每次发送的文本块数量，防止单次请求超限
        )
        
        # 初始化向量数据库
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # 文档切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""],
        )
        
        # 构建 RAG Chain 的提示词
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT),
            ("user", "{question}"),
        ])
        
        # 尝试加载已有索引
        self._try_load_existing_index()
    
    def _try_load_existing_index(self):
        """尝试加载已有的向量数据库"""
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
                    self.retriever = self.vectorstore.as_retriever(
                        search_kwargs={"k": config.TOP_K}
                    )
                    self._build_chain()
                    logger.info(f"✅ 加载已有索引，包含 {count} 个文档块")
                else:
                    self.vectorstore = None
            except Exception as e:
                logger.warning(f"⚠️ 加载已有索引失败: {e}")
                self.vectorstore = None
    
    def _build_chain(self):
        """构建 RAG Chain"""
        if not self.retriever:
            return
        
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }

            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
    
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
        
        for loader in [txt_loader, pdf_loader, md_loader]:
            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"⚠️ 加载失败: {e}")
        
        if not all_docs:
            return {"error": "未找到任何文档（支持 .txt, .pdf, .md）"}
        
        # 切分文档
        chunks = self.splitter.split_documents(all_docs)
        
        # 存入向量数据库
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        
        # 创建检索器
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )
        
        # 构建 Chain
        self._build_chain()
        
        stats = {
            "文件数": len(all_docs),
            "文本块数": len(chunks),
            "状态": "✅ 索引构建成功",
        }
        logger.info(f"✅ 索引完成: {stats}")
        return stats
    
    def query(self, question: str) -> str:
        """提问（非流式）"""
        if not self.rag_chain:
            return "⚠️ 知识库尚未构建索引！请先上传文档。"
        return self.rag_chain.invoke(question)
    
    def query_stream(self, question: str):
        """提问（流式）"""
        if not self.rag_chain:
            yield "⚠️ 知识库尚未构建索引！请先上传文档。"
            return
        
        for chunk in self.rag_chain.stream(question):
            yield chunk
    
    def get_sources(self, question: str) -> list[dict]:
        """获取问题相关的文档来源"""
        if not self.retriever:
            return []
        
        docs = self.retriever.invoke(question)
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