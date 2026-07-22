"""
Advanced RAG 个人知识库 AI 助手 - 专业级 Web 界面

app.py
├── 0. 导入与全局配置
├── 1. Tab 1 功能函数: 知识库问答 (6 个函数)混合检索 + 查询改写 + 重排序 + 联网查询 + 内存记忆
├── 2. Tab 2 功能函数: 检索透视 (1 个函数)
├── 3. Tab 3 功能函数: 文档管理 (5 个函数)
├── 4. Tab 4 功能函数: 关于 (纯静态，0 个函数)
├── 5. Tab 5 功能函数: 数据分析师 (3 个函数)
|___6. Tab 6 功能函数：机器学习实验室()
├── 7. 构建 Gradio 界面 (纯 UI 组装)
│   ├── Tab 1: 💬 知识库问答
│   ├── Tab 2: 🔍 检索透视
│   ├── Tab 3: 📁 文档管理
│   ├── Tab 4: ℹ️ 关于
│   └── Tab 5: 📈 数据分析师
|   |__ Tab 6: 机器学习实验室
└── 8. 启动应用
"""


# ============================================================
# 0. 导入与全局配置 (Imports & Globals)
# ============================================================

import gradio as gr
import logging
import json
from pathlib import Path
import uuid
import re
import config

from langchain_core.messages import AIMessage


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)



from rag_engine import RAGEngine
# 初始化 RAG 引擎
engine = RAGEngine()  # 全局单例

from data_engine import DataEngine
# 初始化数据分析引擎
data_engine = DataEngine()

from ml_engine import MLEngine
# 初始化 ML 引擎（和 RAGEngine 并列）
ml_engine = MLEngine()



# ===================================================================================
# 1. Tab 1 功能函数: 💬 知识库问答
#    包含：流式问答、路由决策展示、联网搜索控制、记忆管理
# ==================================================================================

def chat_fn(message: str, history: list) -> str:
    """聊天处理（带记忆 + 联网搜索的流式）"""
    if not message.strip():
        return ""
    
    response = ""
    # 1. 流式接收回答
    for chunk in engine.query_stream(message):
        response += chunk
    
    # 2. 获取路由决策元信息
    meta = engine.get_last_query_meta()
    
    # 3. 在回答开头显示路由决策标识
    route_badge = f"\n\n> {meta.get('route_decision', '📚 知识库回答')}\n"
    
    # 附加本地来源信息
    local_sources = meta.get("local_sources", [])
    if local_sources:
        route_badge += "\n📚 **知识库来源:**\n"
        seen = set()
        for i, src in enumerate(local_sources, 1):
            if src["filename"] not in seen:
                route_badge += f"  {i}. 📄 `{src['filename']}`\n"
                seen.add(src["filename"])
    
    #  附加联网搜索来源
    web_results = meta.get("web_results", [])
    if web_results:
        route_badge += "\n🌐 **联网搜索来源:**\n"
        for i, r in enumerate(web_results[:10], 1):
            title = r.get("title", "无标题")[:40]
            url = r.get("url", "")
            if url:
                route_badge += f"  {i}. [{title}]({url})\n"
            else:
                route_badge += f"  {i}. {title}\n"
    
    # 4. 手动写入记忆 (流式输出的标准做法) 流式结束后，手动将本轮对话写入后端 Memory！
    pure_ai_response = response.split("\n\n---\n📚 **参考来源:**")[0]
    engine.add_to_qa_memory(message, pure_ai_response)
    
    return response + route_badge



# 清空Rag_memory记忆函数
def clear_rag_memory_fn():
    """清空 RAG 对话记忆"""
    msg = engine.clear_qa_memory()
    return msg

def refresh_info():
    """刷新知识库状态信息"""
    info = engine.get_index_info()
    md = ""
    for k, v in info.items():
        md += f"- **{k}**: {v}\n"
    docs = engine.list_documents()
    if docs:
        md += f"\n- **文件数**: {len(docs)} 个\n"
    return md



## 搜索引擎内的小函数 3个
def toggle_web_search(enabled):
    """切换联网搜索开关"""
    engine.web_search_enabled = enabled
    status = f"{'✅ 联网搜索已启用' if enabled else '❌ 联网搜索已禁用'}"
    if enabled:
        status += f"\n当前引擎: **{engine.search_provider.upper()}**"
    return status


def update_threshold(value):
    """更新相关性阈值"""
    engine.relevance_threshold = value
    return f"✅ 相关性阈值: {value}"


def switch_provider(provider):
    """切换搜索引擎"""
    return engine.switch_search_provider(provider)



# =================================================================================
# 2. Tab 2 功能函数: 🔍 检索透视
#    包含：检索链路调试、查询改写展示、重排序结果展示
# =============================================================================
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
    
    # 5. 原始 JSON 数据
    raw_json = json.dumps(result, ensure_ascii=False, indent=2)

    return rewrite_md, vector_md + "\n" + bm25_md, rerank_md, raw_json







# ============================================================================================
# 3. Tab 3 功能函数: 📁 文档管理
#    包含：文档列表、上传、删除、清空、从目录构建索引
# =============================================================================================

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








# ===============================================================================
# 4. Tab 4 功能函数: ℹ️ 关于
#    (纯静态展示，无需额外函数，内容直接写在 UI 中)
# ====================================================================================





# ============================================================
# 5. Tab 5 功能函数: 📈 数据分析师 (CSV/Excel Agent)
#    包含：数据加载、流式问答、记忆管理
# ============================================================

def upload_data_fn(files):
    """处理 CSV/Excel 上传并加载到 DataEngine"""
    if not files:
        return "⚠️ 请选择 CSV/xls/xlsx 文件", ""

    file_obj = files
    # 正确获取路径
    if hasattr(file_obj, "name"):
        # 如果是文件对象，取 .name 属性 (临时文件绝对路径)
        file_path = file_obj.name
        # 如果直接是字符串路径，直接使用
    else:
        file_path = str(file_obj)

    print(f"🔍 DEBUG: 解析到的文件路径 -> {file_path}")

    if not file_path or not Path(file_path).exists():
        return f"❌ 文件路径无效或不存在: {file_path}", ""

    result = data_engine.load_dataframe(file_path)

    if "error" in result:
        return f"❌ 加载失败: {result['error']}", ""

    status_md = f"""
    ### ✅ 数据加载成功！
    - **文件名**: `{result['filename']}`
    - **数据规模**: {result['rows']} 行 × {result['columns']} 列
    - **字段信息**: {result['columns_info']}
    """
    preview_md = f"### 📊 数据预览 (前 3 行)\n{result['preview']}"
    return status_md, preview_md


#==========  核心：适配 LangGraph 的流式处理函数 ==============

def data_chat_fn(
    message: str, history: list, user_already_added: bool = False, thinking_added: bool = False
):
    """
    处理数据分析 Agent 的流式问答 + 图表展示
    适配 LangGraph create_agent + Gradio 6.0 messages 格式
    """
    # 第一步：立即 yield 清空输入框
    if not message.strip():
        yield history, None
        return

    if not hasattr(data_engine, 'agent') or data_engine.agent is None:
        # 如果用户消息还没添加（兼容旧逻辑），则添加
        if not user_already_added:
            history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": "⚠️ 请先在上方上传 CSV 或 Excel 文件加载数据！"}
        )
        yield history, None
        return

    try:
        # 1. 确保 session_id 存在
        if not hasattr(data_engine, 'session_id') or not data_engine.session_id:
            data_engine.session_id = str(uuid.uuid4())
        # 2. 清空上一次的图表
        data_engine.clear_last_chart()

        full_response = ""
        first_chunk_received = False  #  标记是否收到第一个 AI 回复

        for event in data_engine.stream_raw(message):
            messages = event.get("messages", [])
            if not messages:
                continue

            last_msg = messages[-1]

            if (
                isinstance(last_msg, AIMessage)
                and last_msg.content
                and not last_msg.tool_calls
            ):
                if last_msg.content != full_response:
                    full_response = last_msg.content
                    # 关键改造：第一次收到 AI 回复时，替换掉"正在思考"提示
                    if thinking_added and not first_chunk_received:
                        if history and history[-1].get("role") == "assistant":
                            history[-1] = {
                                "role": "assistant",
                                "content": full_response,
                            }
                        else:
                           
                            history.append(
                                {"role": "assistant", "content": full_response}
                            )
                        first_chunk_received = True
                    else:
                        if history and history[-1].get("role") == "assistant":
                            history[-1] = {
                                "role": "assistant",
                                "content": full_response,
                            }
                        else:
                            # 添加 AI 的回复
                            history.append(
                                {"role": "assistant", "content": full_response}
                            )

                    chart_path = data_engine.get_last_chart()
                    yield history, chart_path

                    # 为下一次更新做准备：弹出临时的 AI 消息
                    if history and history[-1].get("role") == "assistant":
                        history.pop()

        # 流结束后，检查是否有图表生成
        chart_path = data_engine.get_last_chart()

        # 最终确认
        if not full_response:
            full_response = "✅ 分析完成，但未生成文本回复（可能仅执行了代码或生成了图表）。"

        # 清理掉 [CHART_SAVED: ...] 内部标记
        full_response = re.sub(r'\[CHART_SAVED:.*?\]', '', full_response).strip()

        # 极端情况：AI 从未回复过，弹掉"正在思考"提示，
        if thinking_added and not first_chunk_received:
            if (
                history
                and history[-1].get("role") == "assistant"
                and "正在执行" in history[-1].get("content", "")
            ):
                history.pop() # 弹掉"正在思考"提示

        # 正式添加 AI 回复到 history
        history.append({"role": "assistant", "content": full_response})
        yield history, chart_path

    except Exception as e:
        logger.error(f"❌ 数据分析 Agent 执行失败: {e}")
        # 错误时，如果"正在思考"还在，也要弹掉它
        if thinking_added and not first_chunk_received:
            if (
                history
                and history[-1].get("role") == "assistant"
                and "正在执行" in history[-1].get("content", "")
            ):
                history.pop()
        history.append({"role": "assistant", "content": f"\n\n❌ 分析出错: {str(e)}"})
        yield history, None

        
def clear_data_memory_fn():
    """清空数据分析 Agent 记忆"""
    return data_engine.clear_memory()








# ============================================================================================
# 6. Tab 6 功能函数: 🤖 机器学习实验室
# ============================================================================================

def upload_ml_data_fn(files):
    """处理 CSV/Excel 上传并加载到 ML 引擎"""
    if not files:
        return "⚠️ 请选择 CSV/xls/xlsx 文件", ""
    
    file_obj = files
    if hasattr(file_obj, "name"):
        file_path = file_obj.name
    else:
        file_path = str(file_obj)
    
    print(f"🔍 DEBUG ML: 解析到的文件路径 -> {file_path}")
    
    if not file_path or not Path(file_path).exists():
        return f"❌ 文件路径无效或不存在: {file_path}", ""
    
    result = ml_engine.load_dataframe(file_path)
    
    if "error" in result:
        return f"❌ 加载失败: {result['error']}", ""
    
    status_md = f"""
    ### ✅ 数据加载成功！
    - **文件名**: `{result['filename']}`
    - **数据规模**: {result['rows']} 行 × {result['columns']} 列
    - **数值列**: {', '.join(result.get('numeric_cols', [])) or '无'}
    - **分类列**: {', '.join(result.get('categorical_cols', [])) or '无'}
    - **时间列**: {', '.join(result.get('datetime_cols', [])) or '无'}
    - **文本列**: {', '.join(result.get('text_cols', [])) or '无'}
    """
    preview_md = f"### 📊 数据预览 (前 3 行)\n{result['preview']}"
    return status_md, preview_md


def clear_ml_memory_fn():
    """清空机器学习记忆"""
    return ml_engine.clear_memory()


# ========== ML 流式聊天核心 ==========
def ml_chat_fn(message: str, history: list, user_already_added: bool = False, thinking_added: bool = False):
    """
    ML 实验室流式问答核心（适配 LangGraph create_agent）
    """
    if not message.strip():
        yield history, None, None
        return
    
    if not hasattr(ml_engine, 'agent') or ml_engine.agent is None:
        if not user_already_added:
            history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ 请先在上方上传 CSV 或 Excel 文件加载数据！"})
        yield history, None, None
        return
    
    try:
        if not hasattr(ml_engine, 'session_id') or not ml_engine.session_id:
            ml_engine.session_id = str(uuid.uuid4())
        
        ml_engine.clear_last_chart()
        
        full_response = ""
        first_chunk_received = False
        
        for chunk in ml_engine.query_stream(message):
            full_response = chunk
            
            # 清理内部标记
            clean_resp = re.sub(r'\[CHART_SAVED:.*?\]', '', full_response).strip()
            
            if thinking_added and not first_chunk_received:
                # 替换"正在思考"提示
                if history and history[-1].get("role") == "assistant":
                    history[-1] = {"role": "assistant", "content": clean_resp}
                else:
                    history.append({"role": "assistant", "content": clean_resp})
                first_chunk_received = True
            else:
                if history and history[-1].get("role") == "assistant":
                    history[-1] = {"role": "assistant", "content": clean_resp}
                else:
                    history.append({"role": "assistant", "content": clean_resp})
            
            # 检查图表
            chart_path = ml_engine.get_last_chart()
            yield history, chart_path, None
        
        # 最终检查模型文件
        model_path = ml_engine.get_latest_model()
        pipe_path = ml_engine.get_latest_pipeline()
        
        # 清理最终响应中的标记
        if history and history[-1].get("role") == "assistant":
            final_clean = re.sub(r'\[CHART_SAVED:.*?\]', '', history[-1].get("content", "")).strip()
            history[-1] = {"role": "assistant", "content": final_clean}
        
        # 如果有管道，优先返回管道；否则返回模型
        download_path = pipe_path or model_path
        yield history, ml_engine.get_last_chart(), download_path
        
    except Exception as e:
        logger.error(f"❌ ML Agent 执行失败: {e}")
        if thinking_added and not first_chunk_received:
            if history and history[-1].get("role") == "assistant" and "正在执行" in history[-1].get("content", ""):
                history.pop()
        history.append({"role": "assistant", "content": f"❌ 分析出错: {str(e)}"})
        yield history, None, None





# ============================================================
# 6. 构建 Gradio 界面 (UI Assembly)
# ============================================================

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
    # css=CUSTOM_CSS, 放在了 launch() 
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
                            "总结一下知识库中关于的内容",
                            "有哪些重要的技术文档？",
                            "请引用来源回答我的问题",
                        ],
                    )
                
                # 右侧：知识库状态面板
                with gr.Column(scale=1):

                    # --- 知识库状态 ---
                    gr.Markdown("### 📊 知识库状态")
                    index_info_btn = gr.Button("🔄 刷新状态", size="sm")
                    index_info_display = gr.Markdown("点击刷新查看状态")
                                     
                    index_info_btn.click(refresh_info, outputs=index_info_display)
                    demo.load(refresh_info, outputs=index_info_display)  # 页面加载时自动刷新


                    # --- 联网搜索控制 ---
                    # 🌐 联网搜索控制面板
                    gr.Markdown("---")
                    gr.Markdown("### 🌐 联网搜索设置")
                    
                    web_search_toggle = gr.Checkbox(
                        label="启用联网搜索",
                        value=True,
                        interactive=True,
                    )
                    
                    # 🌟搜索引擎下拉选择
                    search_provider_dropdown = gr.Dropdown(
                        label="搜索引擎",
                        choices=["tavily","duckduckgo","bing"],
                        value=config.SEARCH_PROVIDER,
                        interactive=True,
                    )
                    
                    relevance_slider = gr.Slider(
                        label="知识库与问题相关性阈值 (默认阈值是0.3,小于阈值联网)",
                        minimum=-1.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        interactive=True,
                    )
                    
                    web_search_status = gr.Markdown(
                        f"✅ 联网搜索已启用\n当前引擎: **{config.SEARCH_PROVIDER.upper()}**"
                    )
                    # 
                    # 第一步：前端触发:前端取消勾选 web_search_toggle，触发 toggle_web_search(False)，成功将 engine.web_search_enabled 设为 False
                    # 第二步：进入 query_stream: use_web, local_docs = self._should_use_web_search(question)
                    # 第三步：进入 _should_use_web_search
                    # 第四步：回到 query_stream 组装上下文
                    # 第五步：大模型拿到上下文
                    web_search_toggle.change(
                        fn=toggle_web_search, 
                        inputs=[web_search_toggle], 
                        outputs=[web_search_status]
                    )
                    relevance_slider.change(
                        fn=update_threshold, 
                        inputs=[relevance_slider], 
                        outputs=[web_search_status]
                    )
                    #  绑定切换事件
                    search_provider_dropdown.change(
                        fn=switch_provider,
                        inputs=[search_provider_dropdown],
                        outputs=[web_search_status]
                    )



                    # --- 对话记忆管理 ---
                    # 清空 RAG 记忆按钮
                    gr.Markdown("---")
                    gr.Markdown("### 🧠 对话记忆管理")
                    clear_rag_memory_btn = gr.Button(
                        "🗑️ 清空当前对话记忆", variant="stop", size="sm"
                    )
                    rag_memory_status = gr.Markdown("")
                    
                    clear_rag_memory_btn.click(
                        fn=clear_rag_memory_fn, 
                        outputs=rag_memory_status
                    )


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
                placeholder="例如：总结知识库？",
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
        
        # ================================================================================================
        # Tab 3: 📁 文档管理
        # ==============================================================================================
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
                        label="选择文件（支持 .txt, .pdf, .md,.docx）",
                        file_count="multiple",
                        file_types=[".txt", ".pdf", ".md",".docx"],
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




        # =====================================================
        # Tab 5: 📈 数据分析师 (CSV/Excel + 图表可视化)
        # =====================================================
        with gr.Tab("📈 数据分析师", id="data"):
            gr.Markdown("""
            ### 📊 AI 数据分析师 & 可视化专家
            > 上传你的 `.csv` 或 `.xlsx/.xls` 数据文件，AI 将自动读取表结构，并允许你用自然语言进行复杂的数据分析和**图表生成**。
            > 
            > 🎨 **支持图表库**: Matplotlib (基础图表) | Seaborn (统计美化图表)
            """)
            
            # --- 数据上传区域 ---
            with gr.Row():
                with gr.Column(scale=1):
                    data_upload = gr.File(
                        label="上传数据文件 (支持 .csv, .xlsx, .xls)",
                        file_count="single",
                        file_types=[".csv", ".xlsx", ".xls"],
                        type="filepath"
                    )
                    upload_data_btn = gr.Button("📥 加载数据", variant="primary")
                
                with gr.Column(scale=2):
                    data_status = gr.Markdown("等待上传数据...")
                    data_preview = gr.Markdown("")
            
            gr.Markdown("---")
            
            # --- 💬 数据分析 + 图表展示区域 ---
            gr.Markdown("### 💬 向数据提问 (支持图表生成)")
            gr.Markdown("""
            **提问示例：**
            - 📊 "画出各产品类别的销售额柱状图"
            - 📈 "用折线图展示每月的销售趋势"  
            - 🔥 "生成一个数值列之间的相关性热力图"
            - 🥧 "画一个饼图，展示各分类的占比"
            - 📦 "用箱线图展示数据的分布情况"
            """)
            
            with gr.Row():
                # 🌟 左侧：聊天区域
                with gr.Column(scale=3):
                    # 手动构建 Chatbot (替代 ChatInterface，以支持多输出)
                    data_chatbot = gr.Chatbot(
                        label="对话记录",
                        height=500,
                    )
                    
                    with gr.Row():
                        data_question = gr.Textbox(
                            label="输入你的问题",
                            placeholder="例如：帮我画一个柱状图，展示各部门的人数",
                            lines=2,
                            scale=4,
                        )
                        data_ask_btn = gr.Button(
                            "🚀 发送", 
                            variant="primary", 
                            scale=1
                        )
                    
                    # 快捷示例按钮
                    with gr.Row():
                        gr.Markdown("**💡 快捷示例：**")
                    with gr.Row():
                        ex_btn_1 = gr.Button("📊 柱状图", size="sm")
                        ex_btn_2 = gr.Button("📈 折线图", size="sm")
                        ex_btn_3 = gr.Button("🥧 饼图", size="sm")
                        ex_btn_4 = gr.Button("🔥 热力图", size="sm")
                        ex_btn_5 = gr.Button("📦 箱线图", size="sm")
                
                # 🌟 右侧：图表展示区域
                with gr.Column(scale=2):
                    gr.Markdown("#### 🖼️ 生成的图表")
                    
                    chart_image = gr.Image(
                        label="图表预览",
                        type="filepath",
                        height=400,
                        interactive=False,
                    )
                    
                    chart_download = gr.File(
                        label="📥 下载图表",
                        visible=True,
                    )
                    
                    # 清空图表按钮
                    clear_chart_btn = gr.Button("🗑️ 清除当前图表", size="sm")
                    
                    def clear_chart_display():
                        return None, None
                    
                    clear_chart_btn.click(
                        fn=clear_chart_display,
                        outputs=[chart_image, chart_download],
                    )
            
            # --- 记忆管理 ---
            gr.Markdown("---")
            with gr.Row():
                clear_data_memory_btn = gr.Button(
                    "🗑️ 清空分析记忆 (重置上下文)", 
                    variant="stop", 
                    size="sm"
                )
                data_memory_status = gr.Markdown("")
            
            # ========== 事件绑定 ==========
            
            # 1. 加载数据
            upload_data_btn.click(
                fn=upload_data_fn,
                inputs=[data_upload],
                outputs=[data_status, data_preview]
            )

             # 🌟 用一个隐藏的状态组件暂存用户消息
            pending_message = gr.State(None)
            
            # 步骤 1：保存消息 + 清空输入框（立即执行，无需等待）
            def save_and_clear(message):
                if not message.strip():
                    return gr.update(), None  # 空消息不清空，也不保存
                return "", message  # 清空输入框，同时保存消息到 State
            
            # 步骤 2：执行 AI 分析
            def process_and_respond(saved_message, history):
                """从 State 中取出消息，执行 AI 分析"""
                if not saved_message:
                    yield history, None
                    return
                
                # 先把用户消息加入 history，并立刻 yield 一次！
                # 这样用户的消息会瞬间显示在聊天框中，不用等 AI 思考
                history = history or []
                history.append({"role": "user", "content": saved_message})
                yield history, None  # 立刻更新 UI，显示用户消息

                # ② 🌟 添加"正在思考"提示
                history.append({"role": "assistant", "content": "AI 正在分析，请稍候..."})
                yield history, None  # 立刻显示思考提示

                # ③ 调用 data_chat_fn 获取 AI 回复
                # 传入 user_already_added=True 和 thinking_added=True
                # 让 data_chat_fn 知道用户消息和思考提示都已经添加过了
                yield from data_chat_fn(saved_message, history, user_already_added=True, thinking_added=True)

            # 2. 发送问题   点击发送按钮的链式调用
            data_ask_btn.click(
                fn=save_and_clear,
                inputs=[data_question],
                outputs=[data_question, pending_message],
            ).then(
                fn=process_and_respond,
                inputs=[pending_message, data_chatbot],
                outputs=[data_chatbot, chart_image],
            )
            
            # 3.回车发送也用同样的链式逻辑
            data_question.submit(
                fn=save_and_clear,
                inputs=[data_question],
                outputs=[data_question, pending_message],
            ).then(
                fn=process_and_respond,
                inputs=[pending_message, data_chatbot],
                outputs=[data_chatbot, chart_image],
            )

            # 4. 清空记忆
            clear_data_memory_btn.click(
                fn=clear_data_memory_fn,
                outputs=data_memory_status
            )
            
            # 5. 快捷示例按钮绑定
            def make_example_fn(text):
                def fn():
                    return text
                return fn
            
            ex_btn_1.click(fn=make_example_fn("请用 Seaborn 画一个柱状图，展示数据中主要分类和数值的对比"), outputs=[data_question])
            ex_btn_2.click(fn=make_example_fn("请用 Matplotlib 画一个折线图，展示数据的趋势变化"), outputs=[data_question])
            ex_btn_3.click(fn=make_example_fn("请画一个饼图，展示数据中各类别的占比分布（取前10个最多的类别）"), outputs=[data_question])
            ex_btn_4.click(fn=make_example_fn("请用 Seaborn 生成一个所有数值列之间的相关性热力图"), outputs=[data_question])
            ex_btn_5.click(fn=make_example_fn("请用 Seaborn 画一个箱线图，展示所有数值列的数据分布情况"), outputs=[data_question])
            
            # 6. 图表下载联动 (当 chart_image 更新时，同步更新下载文件)
            def sync_download(chart_path):
                if chart_path and Path(chart_path).exists():
                    return gr.update(value=chart_path, visible=True)
                return gr.update(value=None, visible=False)
            
            chart_image.change(
                fn=sync_download,
                inputs=[chart_image],
                outputs=[chart_download],
            )

            # 发送后清空输入框
            def clear_input():
                return ""
            
            data_ask_btn.click(fn=clear_input, outputs=[data_question], queue=False)
            data_question.submit(fn=clear_input, outputs=[data_question], queue=False)








        # =====================================================
        # Tab 6: 🤖 机器学习实验室
        # =====================================================
        with gr.Tab("🤖 机器学习实验室", id="ml"):
            gr.Markdown("""
            ### 🤖 AI 机器学习实验室
            > 上传你的 `.csv` 或 `.xlsx/.xls` 数据文件，AI 将自动完成从数据预处理到模型部署的全流程。
            > 
            > **支持任务**: 分类 | 回归 | 聚类 | 降维  
            > **支持算法**: Scikit-learn, XGBoost, LightGBM, CatBoost, Optuna(调参), SHAP(解释)
            """)
            
            # --- 数据上传区域 ---
            with gr.Row():
                with gr.Column(scale=1):
                    ml_upload = gr.File(
                        label="上传数据文件 (支持 .csv, .xlsx, .xls)",
                        file_count="single",
                        file_types=[".csv", ".xlsx", ".xls"],
                        type="filepath"
                    )
                    ml_upload_btn = gr.Button("📥 加载数据", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🎯 任务类型")
                    ml_task_type = gr.Radio(
                        choices=["自动检测", "分类", "回归", "聚类", "降维"],
                        value="自动检测",
                        label="选择任务类型（可选，AI 也可自动推断）",
                        interactive=True,
                    )
                
                with gr.Column(scale=2):
                    ml_status = gr.Markdown("等待上传数据...")
                    ml_preview = gr.Markdown("")
            
            gr.Markdown("---")
            
            # --- 聊天与可视化区域 ---
            gr.Markdown("### 💬 向机器学习助手提问")
            gr.Markdown("""
            **提问示例：**
            - 🎯 "走完整的分类流程，目标列是 'Survived'"
            - 📈 "用 XGBoost 做回归预测，并用 Optuna 调参"
            - 🔍 "对数据进行聚类分析，比较 KMeans 和 DBSCAN"
            - ⚙️ "做特征工程，训练随机森林，并解释特征重要性"
            - 🔮 "使用 PCA 降维到 2 维并可视化"
            """)
            
            with gr.Row():
                # 左侧：聊天
                with gr.Column(scale=3):
                    ml_chatbot = gr.Chatbot(
                        label="对话记录",
                        height=500,
                    )
                    
                    with gr.Row():
                        ml_question = gr.Textbox(
                            label="输入你的问题",
                            placeholder="例如：用随机森林分类器预测目标变量，走完整流程并调参",
                            lines=2,
                            scale=4,
                        )
                        ml_ask_btn = gr.Button("🚀 发送", variant="primary", scale=1)
                    
                    with gr.Row():
                        gr.Markdown("**💡 快捷示例：**")
                    with gr.Row():
                        ml_ex_1 = gr.Button("🎯 完整分类流程", size="sm")
                        ml_ex_2 = gr.Button("📈 完整回归流程", size="sm")
                        ml_ex_3 = gr.Button("🔍 聚类分析", size="sm")
                        ml_ex_4 = gr.Button("⚙️ 超参数调优", size="sm")
                        ml_ex_5 = gr.Button("🔮 SHAP解释", size="sm")
                
                # 右侧：图表与模型下载
                with gr.Column(scale=2):
                    gr.Markdown("#### 🖼️ 生成的图表")
                    ml_chart_image = gr.Image(
                        label="图表预览",
                        type="filepath",
                        height=400,
                        interactive=False,
                    )
                    ml_chart_download = gr.File(
                        label="📥 下载图表",
                        visible=True,
                    )
                    
                    gr.Markdown("#### 💾 模型与管道")
                    ml_model_download = gr.File(
                        label="📥 下载训练好的模型/管道",
                        visible=True,
                    )
                    
                    clear_ml_chart_btn = gr.Button("🗑️ 清除当前图表", size="sm")
                    
                    def clear_ml_display():
                        return None, None
                    
                    clear_ml_chart_btn.click(
                        fn=clear_ml_display,
                        outputs=[ml_chart_image, ml_chart_download],
                    )
            
            gr.Markdown("---")
            with gr.Row():
                clear_ml_memory_btn = gr.Button(
                    "🗑️ 清空机器学习记忆", variant="stop", size="sm"
                )
                ml_memory_status = gr.Markdown("")
            
            # ========== 事件绑定 ==========
            
            # 1. 加载数据
            ml_upload_btn.click(
                fn=upload_ml_data_fn,
                inputs=[ml_upload],
                outputs=[ml_status, ml_preview]
            )
            
            # 2. 聊天流式处理（参照 Tab5 的 State + .then 链式模式）
            ml_pending = gr.State(None)
            
            def ml_save_and_clear(msg):
                if not msg.strip():
                    return gr.update(), None
                return "", msg
            
            def ml_process_and_respond(saved_msg, history):
                if not saved_msg:
                    yield history, None, None
                    return
                
                history = history or []
                history.append({"role": "user", "content": saved_msg})
                yield history, None, None
                
                history.append({"role": "assistant", "content": "🤖 AI 正在执行机器学习分析，请稍候..."})
                yield history, None, None
                
                # 调用流式生成器
                for h, chart, model in ml_chat_fn(saved_msg, history, user_already_added=True, thinking_added=True):
                    yield h, chart, model
            
            ml_ask_btn.click(
                fn=ml_save_and_clear,
                inputs=[ml_question],
                outputs=[ml_question, ml_pending],
            ).then(
                fn=ml_process_and_respond,
                inputs=[ml_pending, ml_chatbot],
                outputs=[ml_chatbot, ml_chart_image, ml_model_download],
            )
            
            ml_question.submit(
                fn=ml_save_and_clear,
                inputs=[ml_question],
                outputs=[ml_question, ml_pending],
            ).then(
                fn=ml_process_and_respond,
                inputs=[ml_pending, ml_chatbot],
                outputs=[ml_chatbot, ml_chart_image, ml_model_download],
            )
            
            # 3. 清空记忆
            clear_ml_memory_btn.click(
                fn=clear_ml_memory_fn,
                outputs=ml_memory_status
            )
            
            # 4. 快捷示例按钮
            def make_ml_example(text):
                def fn():
                    return text
                return fn
            
            ml_ex_1.click(fn=make_ml_example("请走完整的分类流程，自动检测目标变量，进行数据预处理、特征工程、模型训练、调参和评估"), outputs=[ml_question])
            ml_ex_2.click(fn=make_ml_example("请走完整的回归流程，进行数据预处理、特征工程、训练 XGBoost 和 LightGBM 模型，用 Optuna 调参并比较性能"), outputs=[ml_question])
            ml_ex_3.click(fn=make_ml_example("对数据进行聚类分析，比较 KMeans、DBSCAN 和层次聚类的效果，并可视化"), outputs=[ml_question])
            ml_ex_4.click(fn=make_ml_example("使用 GridSearchCV 和 Optuna 对当前最佳模型进行超参数调优，并绘制调参结果"), outputs=[ml_question])
            ml_ex_5.click(fn=make_ml_example("训练一个树模型，使用 SHAP 解释特征重要性，并绘制 summary plot 和 dependence plot"), outputs=[ml_question])
            
            # 5. 图表下载联动
            def sync_ml_download(path):
                if path and Path(path).exists():
                    return gr.update(value=path, visible=True)
                return gr.update(value=None, visible=False)
            
            ml_chart_image.change(
                fn=sync_ml_download,
                inputs=[ml_chart_image],
                outputs=[ml_chart_download],
            )
            
            # 6. 清空输入框
            def clear_input():
                return ""
            
            ml_ask_btn.click(fn=clear_input, outputs=[ml_question], queue=False)
            ml_question.submit(fn=clear_input, outputs=[ml_question], queue=False)


# ===================================================================
#  7. 启动
# ===================================================================

if __name__ == "__main__":
    demo.launch(
        # server_name="0.0.0.0",
        server_name="127.0.0.1",  # 改成本地回环地址，Windows 浏览器完美识别
        share=True,                
        server_port=7860,
        # share=False,    # To create a public link, set `share=True` in `launch()`
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        css=CUSTOM_CSS,
    )