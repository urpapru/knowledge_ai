"""
RAG 引擎：负责文档加载、切分、索引、检索和问答
Advanced RAG 引擎：负责文档加载、切分、混合检索、重排序和问答
"""
import os
import logging
from pathlib import Path
import hashlib

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma  # 旧的导入
from langchain_chroma import Chroma 
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
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
# from langchain.retrievers.multi_query import MultiQueryRetriever  #  官方已经把旧代码移到了 langchain-classic 包
# from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever # 官方已经把旧代码移到了 langchain-classic 包
# from langchain.retrievers.document_compressors import CrossEncoderReranker # 移动到了langchain-classic包里
from langchain_community.retrievers import BM25Retriever
# from langchain_huggingface import HuggingFaceCrossEncoder #  实际上被归类在社区包（Community） 的交叉编码器模块下
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

import config

logger = logging.getLogger(__name__)


class RAGEngine:
    """个人知识库 RAG 引擎"""
    
    def __init__(self):
        # 1.初始化 LLM
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            # temperature=0.3,  # RAG 场景用低温度，确保准确性
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


    def _build_advanced_retriever(self):
        """
        🌟 核心改造：组装 Advanced 检索器
        流程：向量检索 + BM25 -> 混合检索 -> 多查询改写 -> 重排序
        如何验证 Advanced RAG 真的生效了:
        在 Gradio 界面提问一个包含生僻专有名词的问题（比如你们公司内部的某个项目代号），观察 get_sources 返回的参考文档。
        如果是基础 RAG，它大概率会搜偏；换成 Advanced RAG 后，你会发现 BM25 会死死咬住那个专有名词，Reranker 会把最准的那条排在第一位
        """
        if not self.vectorstore or not self.all_chunks:
            logger.warning("⚠️ 向量库或文档块为空，无法构建高级检索器")
            return

        # Step A: 基础向量检索器 (召回 Top 15 供后续精排)
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

                    # self.retriever = self.vectorstore.as_retriever(
                    #     search_kwargs={"k": config.TOP_K}
                    # )
                    # self._build_chain()
                    # logger.info(f"✅ 加载已有索引，包含 {count} 个文档块")

                else:
                    self.vectorstore = None
            except Exception as e:
                logger.warning(f"⚠️ 加载已有索引失败: {e}")
                self.vectorstore = None
    
    def _build_chain(self):
        """构建 RAG LCEL Chain"""
        if not self.advanced_retriever:
            return
        
        self.rag_chain = (
            {
                "context": self.advanced_retriever | self._format_docs,
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
            return {"error": "未找到任何文档（支持 .txt, .pdf, .md）"}
        
        # 切分文档
        chunks = self.splitter.split_documents(all_docs) #  🔥 存入实例变量
        
        # 存入向量数据库
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        
        # 创建检索器
        # self.retriever = self.vectorstore.as_retriever(
        #     search_kwargs={"k": config.TOP_K}
        # )
        
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
    # -------- ---------
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