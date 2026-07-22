"""
data_engine.py
数据分析引擎 - 负责 EDA 数据探索、统计分析与可视化图表生成
"""
import io
import sys
import re
import logging
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import MemorySaver

import config

logger = logging.getLogger(__name__)

# ==================== 全局状态（供 Tool 访问）====================
_current_df: Optional[pd.DataFrame] = None
_last_chart_path: Optional[str] = None

# 输出目录
_CHART_OUT = Path("./chart_outputs")
_CHART_OUT.mkdir(exist_ok=True)


# ==================== 安全导入白名单 ====================
_ALLOWED_MODULES = {
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy',
    'json', 'math', 'random', 'statistics', 'itertools', 'collections',
    'datetime', 'typing', 're', 'string', 'warnings', 'copy', 'functools',
    'hashlib', 'base64', 'io', 'pathlib', 'inspect', 'textwrap', 'enum',
    'numbers', 'decimal', 'fractions', 'builtins',
}


def _safe_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base in _ALLOWED_MODULES:
        return __builtins__['__import__'](name, *args, **kwargs)
    raise ImportError(
        f"模块 '{name}' 不在白名单中。允许导入: pandas, numpy, matplotlib, seaborn 及 Python 标准库"
    )


# ==================== 图表自动保存 ====================
def _auto_save_figures() -> List[str]:
    """自动检测并保存所有打开的 matplotlib 图形到 chart_outputs/
     原理：
        - plt.get_fignums() 返回当前所有打开的 figure 编号
        - 如果代码中创建了 figure 但没有 savefig，我们在这里自动保存
        - 保存后关闭所有 figure 释放内存
     返回: 保存的图片路径列表
    
    """
    global _last_chart_path
    saved: List[str] = []
    fig_nums = plt.get_fignums()
    if not fig_nums:
        return saved

    ts = int(time.time() * 1000) # 毫秒级时间戳，避免文件名冲突
    for i, num in enumerate(fig_nums):

        # 检查 figure 是否有实际内容 (排除空 figure)
        fig = plt.figure(num)
        if not fig.axes:
            continue
        # 自动生成文件名
        filepath = _CHART_OUT / f"chart_{ts}_{i}.png"
        try:
            fig.savefig(
                str(filepath),
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
            )
            saved.append(str(filepath))
            logger.info(f"📊 图表保存: {filepath}")
        except Exception as e:
            logger.warning(f"⚠️ 保存图表失败: {e}")
    #  关键：关闭所有图形，释放内存
    plt.close('all')
    if saved:
        _last_chart_path = saved[0]
    return saved


# ==================== Tool 1: 通用代码执行 ====================
@tool
def safe_python_repl(query: str) -> str:
    """
    执行 Python 代码进行数据分析。
    可用变量: df (当前DataFrame), pd, np, plt, sns
    """
    global _current_df, _last_chart_path
    #  1. 清洗大模型生成的错误转义符
    clean_code = query.replace("\\n", "\n").replace("\\\\", "\\")
    #  2. 准备安全的执行环境
    safe_builtins = {
        "print": print, "len": len, "max": max, "min": min, "sum": sum,
        "abs": abs, "round": round, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple, "set": set, "range": range,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "next": next, "iter": iter,
        "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
        "setattr": setattr, "type": type, "bool": bool,
        "True": True, "False": False, "None": None,
        "__import__": _safe_import,
    }
    
    safe_globals = {
        "df": _current_df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "__builtins__": safe_builtins,
    }
    # 3 捕获 print 输出
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        plt.close('all') # 执行代码前，关闭所有残留的旧图形
        exec(clean_code, safe_globals) # 执行代码
        output = mystdout.getvalue() # 获取 print 输出

        # 如果没有 print 输出，尝试获取最后一个表达式的值
        if not output.strip():
            lines = clean_code.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                skip_prefixes = (
                    '#', 'import ', 'from ', 'def ', 'class ',
                    'if ', 'for ', 'while ', 'with ', 'try:', 'except', '@',
                )
                if (
                    last_line
                    and not last_line.startswith(skip_prefixes)
                    and '=' not in last_line.split('#')[0]
                ):
                    try:
                        result = eval(last_line, safe_globals)
                        if result is not None:
                            output = str(result)
                    except Exception:
                        pass

        # 自动保存图表
        figs = _auto_save_figures()
        # 核心 自动检测是否有未保存的 matplotlib 图形
        chart_info = ""
        if figs:
            chart_info = f"\n\n[CHART_SAVED: {figs[0]}]" # 取第一张图

        result_text = (
            output[:4000]
            if len(output) > 4000
            else output if output else "代码执行成功，无输出。"
        )
        return result_text + chart_info

    except Exception as e:
        plt.close('all') # # 出错也必须关闭图形，防止内存泄漏
        return f"执行出错: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout


# ==================== Tool 2: 专用绘图工具 ====================
@tool
def create_chart(code: str) -> str:
    """
    🎨 专用绘图工具：执行 Matplotlib/Seaborn 代码并保存图表。
    可用变量: df, pd, np, plt, sns
    不需要调用 plt.savefig() 或 plt.show()，系统会自动保存。
    """
    global _current_df, _last_chart_path
    # # 清洗代码
    clean_code = code.replace("\\n", "\n").replace("\\\\", "\\")
    # # 准备执行环境
    safe_builtins = {
        "print": print, "len": len, "max": max, "min": min, "sum": sum,
        "abs": abs, "round": round, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple, "set": set, "range": range,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "next": next, "iter": iter,
        "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
        "setattr": setattr, "type": type,
        "True": True, "False": False, "None": None,
        "__import__": _safe_import,
    }

    safe_globals = {
        "df": _current_df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "__builtins__": safe_builtins,
    }

    try:
        # 执行代码前，关闭所有残留的旧图形
        plt.close('all')
        # 执行代码
        exec(clean_code, safe_globals)
        # 自动保存
        saved = _auto_save_figures()
        if saved:
            _last_chart_path = saved[0]
            return f"✅ 图表已生成并保存！路径: {saved[0]}"
        return "⚠️ 代码执行成功，但未检测到图形。请确保代码中创建了 figure。"
    except Exception as e:
        plt.close('all')
        return f"❌ 绘图失败: {type(e).__name__}: {str(e)}"


# ==================== DataEngine 主类 ====================
class DataEngine:
    """数据分析引擎"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.1,
        )
        self.agent = None
        # # 用于区分不同会话的 ID
        self.session_id = str(uuid.uuid4())
        self.dataframes: Dict[str, pd.DataFrame] = {}  # CSV /execl数据分析模块占位
        self.current_df_name: Optional[str] = None
        # 内存记忆初始化 (短暂记忆),使用 LangGraph 的 MemorySaver
        self.checkpointer = MemorySaver()
        # 初始化 matplotlib 中文字体
        self._setup_matplotlib_chinese()
        logger.info("✅ DataEngine 初始化完成")

    def _setup_matplotlib_chinese(self):
        """配置 Matplotlib 中文字体  (跨平台兼容)
        
        """
        import platform
        system = platform.system()
        # 关键：必须在 sns.set_theme() 之后设置字体，否则会被 Seaborn 覆盖！
        # ① 先设置 Seaborn 主题（这会重置 matplotlib 的 rcParams）
        sns.set_theme(style="whitegrid", palette="husl")
        # ② 然后再设置中文字体（覆盖 Seaborn 的默认字体）
        if system == "Windows":
            #  # 微软雅黑（Win7+ 自带）,黑体（所有 Windows 自带），宋体（备选）
            plt.rcParams['font.sans-serif'] = [
                'Microsoft YaHei', 'SimHei', 'SimSun'
            ]
        elif system == "Darwin":
            # macOS，苹方（macOS 自带），黑体-SC，华文黑体
            plt.rcParams['font.sans-serif'] = [
                'PingFang SC', 'Heiti SC', 'STHeiti'
            ]
        else:
            plt.rcParams['font.sans-serif'] = [
                'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans'
            ]
        # ③ 解决负号 '-' 显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False

    def load_dataframe(self, file_path: str) -> dict:
        """加载 CSV/Excel 到内存，并构建 Data Agent
        通用表格数据加载器：支持 CSV (.csv) 和 Excel (.xlsx, .xls)
        将数据加载到内存，并初始化 Pandas Data Agent
        
        """
        # # 引用全局变量
        global _current_df
        path = Path(file_path)
        #  移除或放宽后缀检查，因为 Gradio 临时文件可能没有 .csv 后缀
        if not path.exists():
            return {"error": f"文件不存在: {file_path}"}

        suffix = path.suffix.lower()
        if suffix not in [".csv", ".xlsx", ".xls"]:
            return {"error": "不支持的文件格式，请上传 CSV 或 Excel 文件"}

        try:
            # 1. 读取 CSV
            df = None
            # 常见的 CSV 编码顺序：UTF-8 (最通用), GBK (中文 Windows Excel 默认), latin1 (兜底)
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            file_type_desc = ""

            for enc in encodings:
                try:
                    if suffix == ".csv":
                        # 尝试使用当前编码读取
                        df = pd.read_csv(file_path, encoding=enc)
                        logger.info(f"✅ 成功使用 '{enc}' 编码读取 CSV: {path.name}")
                        file_type_desc = f"CSV (编码: {enc})"
                        break # 读取成功，跳出循环
                    else:
                        # 读取 Excel：默认读取第一个 Sheet (sheet_name=0)，如果需要读取特定 Sheet，后续可以扩展参数
                        df = pd.read_excel(file_path, sheet_name=0)
                        file_type_desc = "Excel (Sheet1)"
                        break
                except UnicodeDecodeError:
                    # 如果解码失败，继续尝试下一个编码
                    continue
                except Exception as e:
                    # 如果是其他错误（比如根本不是 CSV 格式），直接抛出
                    return {"error": f"文件解析失败: {str(e)}"}
            # 如果所有编码都失败了（理论上 latin1 不会失败，但以防万一）
            if df is None:
                return {
                    "error": "无法解析文件编码，请确保文件为 UTF-8/GBK/latin1 编码"
                }

            # 2. 基础清洗（可选：去除全空的行列）
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            # 3. 存入内存字典
            self.dataframes[path.name] = df
            self.current_df_name = path.name
            _current_df = df  # 🌟 让 Tool 能够访问最新的 df

            rows, cols = df.shape
            # 构建系统提示词 (包含数据概览，让 LLM 知道 df 长什么样)
            columns_info = ", ".join(
                [f"{col} ({df[col].dtype})" for col in df.columns]
            )
            # 数据前三行
            df_head = df.head(3).to_string()
            # 调佣构建agent 函数
            self._build_agent(
                filename=path.name,
                rows=rows,
                cols=cols,
                columns_info=columns_info,
                df_head=df_head,
            )

            return {
                "status": "success",
                "filename": path.name,
                "file_type": file_type_desc,
                "rows": rows,
                "columns": cols,
                "columns_info": columns_info,
                "preview": df.head(3).to_markdown(index=False),
            }

        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return {"error": f"解析失败: {str(e)}"}

    def _build_agent(self, filename, rows, cols, columns_info, df_head):
        """构建数据分析 Agent"""
        # system_prompt 直接接受纯字符串 (不再需要 SystemMessage 包装)
        system_prompt = (
            "你是一个专业的数据分析师和数据可视化专家。\n"
            "你可以使用 Pandas 对提供的 DataFrame (`df`) 进行数据分析，"
            "并使用图表工具生成精美的可视化图表。\n"
            "请使用中文回答。编写 Python 代码时直接使用变量 `df`，无需重新读取文件。\n"
            "【重要】代码中的换行符必须是真正的换行符，"
            "绝对不要输出字面量 '\\\\n'。\n\n"
            "## 你的工具：\n"
            "1. **safe_python_repl**: 执行 Python 代码进行数据分析、统计计算、数据清洗等\n"
            "2. **create_chart**: 专门用于生成 Matplotlib/Seaborn 图表\n\n"
            "## 工作流程：\n"
            "1. 如果用户要求分析数据：使用 safe_python_repl 执行分析代码\n"
            "2. 如果用户要求画图：使用 create_chart 工具生成图表\n"
            "3. 如果需要先分析再画图：先用 safe_python_repl 准备数据，"
            "再用 create_chart 画图\n\n"
            "## 图表规范：\n"
            "- 始终设置 figsize 保证图表足够大 (建议 10x6 或 12x8)\n"
            "- 标题使用中文，fontsize=16, fontweight='bold'\n"
            "- 轴标签使用中文，fontsize=12\n"
            "- 如果 x 轴标签过长，使用 plt.xticks(rotation=45)\n"
            "- 始终调用 plt.tight_layout() 防止标签被裁剪\n"
            "- 使用 sns 的调色板让图表更美观\n\n"
            f"当前数据概览:\n"
            f"- 文件名: {filename}\n"
            f"- 行数: {rows}, 列数: {cols}\n"
            f"- 列名及类型: {columns_info}\n"
            f"- 前3行预览:\n{df_head}"
        )
        # 初始化 LangGraph Checkpointer
        if not hasattr(self, 'checkpointer') or self.checkpointer is None:
            self.checkpointer = MemorySaver()

        # 使用 LangGraph 构建现代 Agent， (引入 Middleware 中间件)
        self.agent = create_agent(
            model=self.llm,
            tools=[safe_python_repl, create_chart], # #  两个工具
            system_prompt=system_prompt, #  新版参数：直接传字符串
            checkpointer=self.checkpointer,
            middleware=[
                # 1：限制最大模型调用次数，彻底杜绝死循环和 Token 爆炸！如果 Agent 陷入“思考->报错->再思考”的死循环，达到 60 次后会强制中断并返回已有结果。
                ModelCallLimitMiddleware(run_limit=60)
                # 2 (可选)：如果你希望长对话自动压缩，可以取消这行的注释 # SummarizationMiddleware(max_tokens=4000), 
                ],
        )
        # # 用于区分不同会话的 ID
        self.session_id = str(uuid.uuid4())
        logger.info("✅ 数据分析 Agent 构建完成")

    def stream_raw(self, question: str):
        """
        返回原始 LangGraph event 流，供前端精细控制流式输出。
        """
        if not self.agent:
            return

        global _current_df
        if self.current_df_name and self.current_df_name in self.dataframes:
            _current_df = self.dataframes[self.current_df_name]

        cfg = {"configurable": {"thread_id": self.session_id}}
        yield from self.agent.stream(
            {"messages": [("user", question)]},
            config=cfg,
            stream_mode="values",
        )

    def query_stream(self, question: str):
        """流式执行数据分析（简化版，直接 yield 字符串）"""
        if not self.agent:
            yield "⚠️ 数据分析 Agent 尚未初始化！请先上传数据文件。"
            return

        global _current_df
        if self.current_df_name and self.current_df_name in self.dataframes:
            _current_df = self.dataframes[self.current_df_name]

        try:
            # 配置 LangGraph
            cfg = {"configurable": {"thread_id": self.session_id}}
            full_response = ""
            # 流式调用 Agent
            for event in self.agent.stream(
                {"messages": [("user", question)]},
                config=cfg,
                stream_mode="values",
            ):
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
                        yield full_response

            if not full_response:
                yield "✅ 分析完成，但未生成文本回复（可能仅执行了代码或生成了图表）。"

        except Exception as e:
            logger.error(f"❌ 数据分析 Agent 执行失败: {e}")
            yield f"❌ 分析出错: {str(e)}"

    def query(self, question: str) -> str:
        """非流式查询（兼容旧接口）"""
        if not self.agent:
            return "⚠️ 数据分析 Agent 尚未初始化！"

        try:
            #  配置 LangGraph
            cfg = {"configurable": {"thread_id": self.session_id}}
            response = self.agent.invoke(
                {"messages": [("user", question)]},
                config=cfg,
            )
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"❌ 数据分析失败: {e}")
            return f"分析失败: {str(e)}"

    def clear_memory(self):
        """清空数据分析记忆 ,生成新 ID 即可变相清空"""
        self.session_id = str(uuid.uuid4()) 
        return "✅ 数据分析记忆已清空！"

    def get_last_chart(self) -> Optional[str]:
        global _last_chart_path
        if _last_chart_path and Path(_last_chart_path).exists():
            return _last_chart_path
        return None

    def clear_last_chart(self):
        global _last_chart_path
        _last_chart_path = None