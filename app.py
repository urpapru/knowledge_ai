"""
Advanced RAG 个人知识库 AI 助手 - 专业级 Web 界面
"""
import gradio as gr
import logging
import os
import json
from pathlib import Path

from rag_engine import RAGEngine
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化 RAG 引擎
engine = RAGEngine()


# ===================================================================
#                         功能函数
# ===================================================================

def chat_fn(message: str, history: list) -> str:
    """聊天处理（流式）"""
    if not message.strip():
        return ""
    
    response = ""
    for chunk in engine.query_stream(message):
        response += chunk
    
    # 附加来源信息
    sources = engine.get_sources(message)
    if sources:
        response += "\n\n---\n📚 **参考来源:**\n"
        seen = set()
        for i, src in enumerate(sources, 1):
            if src["filename"] not in seen:
                response += f"  {i}. 📄 `{src['filename']}`\n"
                seen.add(src["filename"])
    
    return response


def debug_retrieval_fn(question: str) -> tuple[str, str, str, str]:
    """
    检索透视：展示检索链路每一步的详细结果
    返回 4 个字符串，分别对应 4 个展示区域
    """
    if not question.strip():
        return "请输入问题", "", "", ""
    
    result = engine.debug_retrieval(question)
    
    if "error" in result:
        return f"❌ 错误: {result['error']}", "", "", ""
    
    # 1. 查询改写结果
    rewrite_md = f"### 🔄 原始问题\n> {result['original_query']}\n\n"
    rewrite_md += "### ✍️ AI 改写后的查询\n"
    for i, q in enumerate(result.get("rewritten_queries", []), 1):
        rewrite_md += f"{i}. `{q}`\n"
    
    # 2. 向量检索结果
    vector_md = f"### 🧠 语义检索结果 (Top {len(result.get('vector_results', []))})\n"
    vector_md += "| # | 内容预览 | 来源 |\n|---|---------|------|\n"
    for i, r in enumerate(result.get("vector_results", []), 1):
        vector_md += f"| {i} | {r['content'][:60]}... | `{r['source']}` |\n"
    
    # 3. BM25 检索结果
    bm25_md = f"### 🔤 关键词检索结果 (Top {len(result.get('bm25_results', []))})\n"
    bm25_md += "| # | 内容预览 | 来源 |\n|---|---------|------|\n"
    for i, r in enumerate(result.get("bm25_results", []), 1):
        bm25_md += f"| {i} | {r['content'][:60]}... | `{r['source']}` |\n"
    
    # 4. Reranker 精排结果
    rerank_md = f"### ⚖️ Reranker 精排结果 (最终 Top {len(result.get('reranked_results', []))})\n"
    rerank_md += "| 排名 | 内容预览 | 来源 |\n|------|---------|------|\n"
    for r in result.get("reranked_results", []):
        rerank_md += f"| 🏆 #{r['rank']} | {r['content'][:60]}... | `{r['source']}` |\n"
    
    return rewrite_md, vector_md + "\n" + bm25_md, rerank_md, json.dumps(result, ensure_ascii=False, indent=2)


def list_documents_fn() -> str:
    """列出知识库中的所有文档"""
    docs = engine.list_documents()
    
    if not docs:
        return "📭 知识库为空，请先上传文档。"
    
    md = "### 📚 知识库文档列表\n\n"
    md += "| 文件名 | 文档块数 | 操作 |\n"
    md += "|--------|---------|------|\n"
    for doc in docs:
        md += f"| 📄 `{doc['filename']}` | {doc['chunks']} 块 | - |\n"
    
    md += f"\n**总计**: {len(docs)} 个文件"
    return md


def upload_and_add_fn(files):
    """上传并追加文档到知识库"""
    if not files:
        return "⚠️ 请选择要上传的文件", list_documents_fn()
    
    # 保存上传的文件
    saved_paths = engine.upload_files_to_temp(files)
    
    # 追加到知识库
    result = engine.add_documents(saved_paths)
    
    # 构建状态信息
    if "error" in result:
        status = f"❌ 上传失败: {result['error']}"
    else:
        status = f"✅ 上传成功！\n"
        status += f"- 新增文件: {result['新增文件数']} 个\n"
        status += f"- 新增文本块: {result['新增文本块数']} 个\n"
        status += f"- 知识库总块数: {result['知识库总块数']} 个\n"
        
        # 列出上传的文件名
        status += "\n📁 已上传文件:\n"
        for p in saved_paths:
            status += f"  - `{Path(p).name}`\n"
    
    return status, list_documents_fn()


def delete_document_fn(filename: str) -> tuple[str, str]:
    """删除指定文档"""
    if not filename.strip():
        return "⚠️ 请输入要删除的文件名", list_documents_fn()
    
    result = engine.delete_document(filename.strip())
    
    if "error" in result:
        status = f"❌ 删除失败: {result['error']}"
    else:
        status = f"✅ 删除成功！\n"
        status += f"- 文件: `{result['filename']}`\n"
        status += f"- 删除文本块: {result['deleted_chunks']} 个\n"
        status += f"- 剩余文本块: {result['remaining_chunks']} 个\n"
    
    return status, list_documents_fn()


def clear_all_fn() -> tuple[str, str]:
    """清空知识库"""
    result = engine.clear_all()
    if "error" in result:
        return f"❌ 清空失败: {result['error']}", list_documents_fn()
    return "✅ 知识库已完全清空！", list_documents_fn()


def build_index_fn(docs_dir: str) -> tuple[str, str]:
    """从目录构建索引"""
    result = engine.load_documents(docs_dir)
    info = engine.get_index_info()
    
    status = "📊 **索引信息:**\n"
    for k, v in info.items():
        status += f"- {k}: {v}\n"
    for k, v in result.items():
        status += f"- {k}: {v}\n"
    
    return status, list_documents_fn()


# ===================================================================
#                         构建 Gradio 界面
# ===================================================================

# 自定义 CSS（让界面更专业）
CUSTOM_CSS = """
.main-header { text-align: center; margin-bottom: 5px; }
.sub-header { text-align: center; color: #666; margin-bottom: 20px; }
.stat-box { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; padding: 15px; border-radius: 10px; 
    text-align: center;
}
"""

with gr.Blocks(
    title="📚 Advanced RAG 知识库 AI 助手",
    css=CUSTOM_CSS,
) as demo:
    
    # ===== 顶部标题 =====
    gr.HTML("""
    <div class="main-header">
        <h1>📚 Advanced RAG 知识库 AI 助手</h1>
    </div>
    <div class="sub-header">
        <p>混合检索 + 查询改写 + 重排序 | Powered by Qwen + LangChain + ChromaDB</p>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        
        # =====================================================
        # Tab 1: 💬 知识库问答
        # =====================================================
        with gr.Tab("💬 知识库问答", id="chat"):
            with gr.Row():
                # 左侧：聊天主区域
                with gr.Column(scale=3):
                    chatbot = gr.ChatInterface(
                        fn=chat_fn,
                        # type="messages",
                        examples=[
                            "总结一下知识库中关于 Python 的内容",
                            "有哪些重要的技术文档？",
                            "请引用来源回答我的问题",
                        ],
                    )
                
                # 右侧：知识库状态面板
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 知识库状态")
                    index_info_btn = gr.Button("🔄 刷新状态", size="sm")
                    index_info_display = gr.Markdown("点击刷新查看状态")
                    
                    def refresh_info():
                        info = engine.get_index_info()
                        md = ""
                        for k, v in info.items():
                            md += f"- **{k}**: {v}\n"
                        docs = engine.list_documents()
                        if docs:
                            md += f"\n- **文件数**: {len(docs)} 个\n"
                        return md
                    
                    index_info_btn.click(refresh_info, outputs=index_info_display)
                    demo.load(refresh_info, outputs=index_info_display)  # 页面加载时自动刷新
        
        # =====================================================
        # Tab 2: 🔍 检索透视
        # =====================================================
        with gr.Tab("🔍 检索透视", id="debug"):
            gr.Markdown("""
            ### 🔬 检索过程透视
            > 输入一个问题，观察 Advanced RAG 检索链路的每一步。
            > 理解 AI 是如何改写查询、混合检索、以及重排序的。
            """)
            
            debug_input = gr.Textbox(
                label="输入问题",
                placeholder="例如：Python 的装饰器怎么用？",
                lines=2,
            )
            debug_btn = gr.Button("🔍 开始检索分析", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    rewrite_output = gr.Markdown(label="查询改写")
                with gr.Column():
                    rerank_output = gr.Markdown(label="精排结果")
            
            with gr.Accordion("📋 详细检索数据（点击展开）", open=False):
                raw_output = gr.Code(label="原始 JSON 数据", language="json")
            
            retrieval_detail = gr.Markdown(label="检索详情")
            
            debug_btn.click(
                fn=debug_retrieval_fn,
                inputs=[debug_input],
                outputs=[rewrite_output, retrieval_detail, rerank_output, raw_output],
            )
        
        # =====================================================
        # Tab 3: 📁 文档管理
        # =====================================================
        with gr.Tab("📁 文档管理", id="docs"):
            
            # --- 3.1 文档列表 ---
            gr.Markdown("### 📋 当前知识库文档")
            
            with gr.Row():
                doc_list_display = gr.Markdown(
                    value="点击刷新查看文档列表",
                    scale=3,
                )
                with gr.Column(scale=1):
                    refresh_doc_btn = gr.Button("🔄 刷新列表", size="sm")
                    refresh_doc_btn.click(list_documents_fn, outputs=doc_list_display)
                    demo.load(list_documents_fn, outputs=doc_list_display)
            
            gr.Markdown("---")
            
            # --- 3.2 上传文档 ---
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📤 上传新文档")
                    file_upload = gr.File(
                        label="选择文件（支持 .txt, .pdf, .md）",
                        file_count="multiple",
                        file_types=[".txt", ".pdf", ".md"],
                    )
                    upload_btn = gr.Button("📤 上传并添加到知识库", variant="primary")
                    upload_status = gr.Markdown()
                
                with gr.Column():
                    gr.Markdown("### 🗑️ 删除文档")
                    delete_input = gr.Textbox(
                        label="输入要删除的文件名",
                        placeholder="例如：report.pdf",
                    )
                    delete_btn = gr.Button("🗑️ 删除文档", variant="stop")
                    delete_status = gr.Markdown()
                    
                    gr.Markdown("---")
                    gr.Markdown("### ⚠️ 危险操作")
                    clear_btn = gr.Button("💣 清空整个知识库", variant="stop")
                    clear_status = gr.Markdown()
            
            # 绑定事件
            upload_btn.click(
                fn=upload_and_add_fn,
                inputs=[file_upload],
                outputs=[upload_status, doc_list_display],
            )
            delete_btn.click(
                fn=delete_document_fn,
                inputs=[delete_input],
                outputs=[delete_status, doc_list_display],
            )
            clear_btn.click(
                fn=clear_all_fn,
                outputs=[clear_status, doc_list_display],
            )
            
            gr.Markdown("---")
            
            # --- 3.3 从目录构建 ---
            with gr.Accordion("📂 从本地目录批量构建（高级）", open=False):
                docs_input = gr.Textbox(
                    label="文档目录路径",
                    value=config.DOCS_DIR,
                    placeholder="./docs",
                )
                build_btn = gr.Button("🔨 从目录构建索引", variant="secondary")
                build_status = gr.Markdown()
                
                build_btn.click(
                    fn=build_index_fn,
                    inputs=[docs_input],
                    outputs=[build_status, doc_list_display],
                )
        
        # =====================================================
        # Tab 4: ℹ️ 关于
        # =====================================================
        with gr.Tab("ℹ️ 关于", id="about"):
            gr.Markdown(f"""
            ### 🛠️ 技术架构
            
            ```
            用户提问
               ↓
            ┌─────────────────────────┐
            │ ① 查询改写 (Multi-Query)│  LLM 生成 3 个不同角度的查询
            └────────────┬────────────┘
                         ↓
            ┌─────────────────────────┐
            │ ② 混合检索 (Ensemble)   │
            │   ├─ BM25 (关键词匹配)  │  精确匹配专有名词
            │   └─ Vector (语义匹配)  │  理解语义相似性
            │   └─ RRF 融合去重       │
            └────────────┬────────────┘
                         ↓
            ┌─────────────────────────┐
            │ ③ 重排序 (Reranker)     │  Cross-Encoder 精准打分
            │   粗排 Top15 → 精排 Top4 │  只保留最相关的文档
            └────────────┬────────────┘
                         ↓
            ┌─────────────────────────┐
            │ ④ LLM 生成回答          │  基于精准上下文生成答案
            └─────────────────────────┘

            ```
            
            ### 📊 当前配置

            | 配置项 | 值 |
            |--------|-----|
            | 对话模型 | `{config.CHAT_MODEL}` |
            | Embedding | `{config.EMBEDDING_MODEL}` |
            | Chunk 大小 | {config.CHUNK_SIZE} 字符 |
            | Chunk 重叠 | {config.CHUNK_OVERLAP} 字符 |
            | 精排保留 | Top {config.TOP_K} |
            | 向量数据库 | ChromaDB |
            
            ### 🎯 Advanced RAG vs Naive RAG
            
            | 特性 | Naive RAG | Advanced RAG |
            |------|-----------|-------------|
            | 口语化提问 | ❌ 容易搜不到 | ✅ 查询改写解决 |
            | 专有名词 | ❌ 语义偏移 | ✅ BM25 精确匹配 |
            | 无关噪音 | ⚠️ 大量擦边内容 | ✅ Reranker 过滤 |
            | Token 消耗 | 高 (~3000) | 低 (~1200) |
            | 幻觉风险 | 较高 | 大幅降低 |
            
            ### 💡 使用技巧
            1. **文档越结构化越好**：有标题、分段落、有列表的文档检索效果最佳
            2. **提问要具体**：越具体的问题，检索越精准
            3. **善用检索透视**：在"检索透视" Tab 观察 AI 是如何搜索的
            4. **定期管理文档**：及时删除过时文档，保持知识库质量
            """)


# ===================================================================
#                         启动
# ===================================================================

if __name__ == "__main__":
    demo.launch(
        # server_name="0.0.0.0",
        server_name="127.0.0.1",  # 改成本地回环地址，Windows 浏览器完美识别
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
    )