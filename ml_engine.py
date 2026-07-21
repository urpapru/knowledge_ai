"""
ml_engine.py
机器学习实验室后端引擎
支持：数据预处理、EDA可视化、特征工程、分类、回归、聚类、降维、调参、模型解释、模型保存
"""
import os
import io
import sys
import re
import base64
import logging
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Matplotlib 非交互后端（必须最先设置）
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
_ml_current_df: Optional[pd.DataFrame] = None
_ml_last_chart_path: Optional[str] = None

# 输出目录
_ML_OUT = Path("./ml_outputs")
_ML_OUT.mkdir(exist_ok=True)
(_ML_OUT / "charts").mkdir(exist_ok=True)
(_ML_OUT / "models").mkdir(exist_ok=True)
(_ML_OUT / "pipelines").mkdir(exist_ok=True)


# ==================== 安全导入白名单 ====================
_ALLOWED_ML_MODULES = {
    'sklearn', 'xgboost', 'lightgbm', 'catboost', 'optuna', 'shap',
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'joblib',
    'imblearn', 'category_encoders', 'feature_engine',
    'json', 'math', 'random', 'statistics', 'itertools', 'collections',
    'datetime', 'typing', 're', 'string', 'warnings', 'copy', 'functools',
    'hashlib', 'base64', 'io', 'pathlib', 'inspect', 'textwrap', 'enum',
    'numbers', 'decimal', 'fractions', 'builtins',
}

def _safe_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base in _ALLOWED_ML_MODULES:
        return __builtins__['__import__'](name, *args, **kwargs)
    raise ImportError(
        f"模块 '{name}' 不在白名单中。允许导入: sklearn, xgboost, lightgbm, catboost, optuna, shap, pandas, numpy 及标准库"
    )


# ==================== 图表自动保存 ====================
def _auto_save_figures_ml() -> List[str]:
    """自动检测并保存所有 matplotlib 图形到 ml_outputs/charts/"""
    global _ml_last_chart_path
    saved: List[str] = []
    fig_nums = plt.get_fignums()
    if not fig_nums:
        return saved
    
    ts = int(time.time() * 1000)
    for i, num in enumerate(fig_nums):
        fig = plt.figure(num)
        if not fig.axes:
            continue
        filepath = _ML_OUT / "charts" / f"ml_chart_{ts}_{i}.png"
        try:
            fig.savefig(str(filepath), dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            saved.append(str(filepath))
            logger.info(f"📊 ML图表保存: {filepath}")
        except Exception as e:
            logger.warning(f"⚠️ 保存图表失败: {e}")
    
    plt.close('all')
    if saved:
        _ml_last_chart_path = saved[0]
    return saved


# ==================== Tool 1: 通用代码执行 ====================
@tool
def safe_python_repl_ml(query: str) -> str:
    """
    执行 Python 代码进行机器学习全流程分析。
    可用变量: df (当前DataFrame), pd, np, plt, sns, joblib
    可用导入: sklearn, xgboost, lightgbm, catboost, optuna, shap, imblearn, category_encoders, feature_engine 等
    """
    global _ml_current_df, _ml_last_chart_path
    clean_code = query.replace("\\n", "\n").replace("\\\\", "\\")
    
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
    
    import joblib as _joblib
    safe_globals = {
        "df": _ml_current_df,
        "pd": pd, "np": np, "plt": plt, "sns": sns,
        "joblib": _joblib,
        "__builtins__": safe_builtins,
    }
    
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    
    try:
        plt.close('all')
        exec(clean_code, safe_globals)
        output = mystdout.getvalue()
        
        # 尝试获取最后一个表达式的值
        if not output.strip():
            lines = clean_code.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                skip_prefixes = ('#', 'import ', 'from ', 'def ', 'class ',
                                'if ', 'for ', 'while ', 'with ', 'try:', 'except', '@')
                if (last_line and 
                    not last_line.startswith(skip_prefixes) and 
                    '=' not in last_line.split('#')[0]):
                    try:
                        result = eval(last_line, safe_globals)
                        if result is not None:
                            output = str(result)
                    except:
                        pass
        
        # 自动保存图表
        figs = _auto_save_figures_ml()
        chart_info = ""
        if figs:
            chart_info = f"\n\n[CHART_SAVED: {figs[0]}]"
        
        result_text = output[:4000] if len(output) > 4000 else output if output else "代码执行成功，无输出。"
        return result_text + chart_info
        
    except Exception as e:
        plt.close('all')
        return f"执行出错: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout


# ==================== Tool 2: 专用绘图工具 ====================
@tool
def create_chart_ml(code: str) -> str:
    """
    🎨 专用绘图工具：执行 Matplotlib/Seaborn 代码并保存图表。
    可用变量: df, pd, np, plt, sns
    不需要调用 plt.savefig() 或 plt.show()，系统会自动保存。
    """
    global _ml_current_df, _ml_last_chart_path
    clean_code = code.replace("\\n", "\n").replace("\\\\", "\\")
    
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
        "df": _ml_current_df, "pd": pd, "np": np, "plt": plt, "sns": sns,
        "__builtins__": safe_builtins,
    }
    
    try:
        plt.close('all')
        exec(clean_code, safe_globals)
        saved = _auto_save_figures_ml()
        if saved:
            _ml_last_chart_path = saved[0]
            return f"✅ 图表已生成并保存！路径: {saved[0]}"
        return "⚠️ 代码执行成功，但未检测到图形。请确保代码中创建了 figure。"
    except Exception as e:
        plt.close('all')
        return f"❌ 绘图失败: {type(e).__name__}: {str(e)}"


# ==================== MLEngine 主类 ====================
class MLEngine:
    """机器学习实验室引擎"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.1,
        )
        self.agent = None
        self.session_id = str(uuid.uuid4())
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.current_df_name: Optional[str] = None
        self.checkpointer = MemorySaver()
        self._setup_matplotlib_chinese()
        logger.info("✅ MLEngine 初始化完成")
    
    def _setup_matplotlib_chinese(self):
        """配置 Matplotlib 中文字体"""
        import platform
        system = platform.system()
        sns.set_theme(style="whitegrid", palette="husl")
        
        if system == "Windows":
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
        elif system == "Darwin":
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False
    
    # ==================== 数据加载 ====================
    def load_dataframe(self, file_path: str) -> dict:
        """加载 CSV/Excel 到内存，并构建 ML Agent"""
        global _ml_current_df
        path = Path(file_path)
        
        if not path.exists():
            return {"error": f"文件不存在: {file_path}"}
        
        suffix = path.suffix.lower()
        if suffix not in [".csv", ".xlsx", ".xls"]:
            return {"error": "不支持的文件格式，请上传 CSV 或 Excel 文件"}
        
        try:
            df = None
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            file_type_desc = ""
            
            for enc in encodings:
                try:
                    if suffix == ".csv":
                        df = pd.read_csv(file_path, encoding=enc)
                        file_type_desc = f"CSV (编码: {enc})"
                        break
                    else:
                        df = pd.read_excel(file_path, sheet_name=0)
                        file_type_desc = "Excel (Sheet1)"
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    return {"error": f"文件解析失败: {str(e)}"}
            
            if df is None:
                return {"error": "无法解析文件编码，请确保文件为 UTF-8/GBK/latin1 编码"}
            
            # 基础清洗
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            
            self.dataframes[path.name] = df
            self.current_df_name = path.name
            _ml_current_df = df
            
            # 列类型分析
            rows, cols = df.shape
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # 启发式检测文本列（长字符串）
            text_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > 50:
                        text_cols.append(col)
            
            columns_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
            df_head = df.head(3).to_string()
            
            # 构建 Agent
            self._build_agent(
                filename=path.name,
                rows=rows, cols=cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                datetime_cols=datetime_cols,
                text_cols=text_cols,
                columns_info=columns_info,
                df_head=df_head
            )
            
            return {
                "status": "success",
                "filename": path.name,
                "file_type": file_type_desc,
                "rows": rows,
                "columns": cols,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "datetime_cols": datetime_cols,
                "text_cols": text_cols,
                "columns_info": columns_info,
                "preview": df.head(3).to_markdown(index=False)
            }
            
        except Exception as e:
            logger.error(f"❌ ML数据加载失败: {e}")
            return {"error": f"解析失败: {str(e)}"}
    
    # ==================== 构建 ML Agent ====================
    def _build_agent(self, filename, rows, cols, numeric_cols, categorical_cols,
                     datetime_cols, text_cols, columns_info, df_head):
        """构建具备完整 ML 工作流能力的 Agent"""
        
        system_prompt = (
            "你是一个顶级的机器学习工程师和数据科学家。你可以使用 Pandas、NumPy、Matplotlib、Seaborn、"
            "Scikit-learn、XGBoost、LightGBM、CatBoost、Optuna、SHAP、imbalanced-learn、category_encoders、"
            "feature-engine 对数据进行完整的机器学习分析。\n"
            "请始终使用中文回答。编写 Python 代码时直接使用变量 `df`，无需重新读取文件。\n"
            "【重要】代码中的换行符必须是真正的换行符，绝对不要输出字面量 '\\\\n'。\n\n"
            
            f"当前数据概览:\n"
            f"- 文件名: {filename}\n"
            f"- 行数: {rows}, 列数: {cols}\n"
            f"- 数值列: {numeric_cols}\n"
            f"- 分类列: {categorical_cols}\n"
            f"- 时间列: {datetime_cols}\n"
            f"- 文本列: {text_cols}\n"
            f"- 列名及类型: {columns_info}\n"
            f"- 前3行预览:\n{df_head}\n\n"
            
            "## 标准机器学习工作流程（必须严格遵守）\n\n"
            
            "### 阶段 1: 数据理解与预处理\n"
            "1. df.info(), df.describe() 了解数据质量\n"
            "2. 缺失值检查: df.isnull().sum()，策略：数值用中位数/均值填充，类别用众数/'Unknown'填充，或删除缺失过多列\n"
            "3. 重复值检查: df.duplicated().sum()\n"
            "4. 异常值检测: 箱线图或 IQR 方法，对极端异常值进行截断或标记\n\n"
            
            "### 阶段 2: 探索性数据分析 (EDA) 与可视化\n"
            "1. 数值特征: distplot/kdeplot 分布图, boxplot 箱线图\n"
            "2. 类别特征: countplot 柱状图, pie 饼图\n"
            "3. 目标变量: 分布分析（分类看平衡性，回归看正态性）\n"
            "4. 特征关系: heatmap 相关性热力图, scatterplot, pairplot（数据量<1000时）\n"
            "5. 使用 create_chart_ml 工具生成图表，或直接在代码中绘图（会自动保存到 ./ml_outputs/charts/）\n\n"
            
            "### 阶段 3: 特征工程\n"
            "1. 特征构造: 从时间列提取年月日/季度/是否周末；从文本列提取长度/词数；数值列构造交叉特征\n"
            "2. 特征编码:\n"
            "   - 低基数类别(<=10): OneHotEncoder, pd.get_dummies\n"
            "   - 高基数类别(>10): TargetEncoder (category_encoders), CatBoost 原生支持\n"
            "   - 有序类别: LabelEncoder, OrdinalEncoder\n"
            "   - 文本特征: TfidfVectorizer, CountVectorizer\n"
            "3. 特征缩放:\n"
            "   - StandardScaler: 数据近似正态分布\n"
            "   - MinMaxScaler: 数据有明确边界\n"
            "   - RobustScaler: 数据含异常值\n"
            "4. 特征选择:\n"
            "   - SelectKBest (f_classif, f_regression, chi2)\n"
            "   - RFE (递归特征消除)\n"
            "   - 基于模型的特征重要性 (RandomForest, XGBoost)\n"
            "5. 降维（可选）:\n"
            "   - PCA: 纯数值特征\n"
            "   - MCA: 纯类别特征 (prince 库或 sklearn 组合)\n"
            "   - FAMD: 混合类型特征\n"
            "   - KernelPCA: 非线性降维\n"
            "   - UMAP/t-SNE: 可视化降维\n\n"
            
            "### 阶段 4: 数据划分\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y[分类时])\n"
            "数据量<1000时务必使用交叉验证: StratifiedKFold(分类) / KFold(回归)\n\n"
            
            "### 阶段 5: 模型训练\n"
            "**分类任务**:\n"
            "  - 基线: LogisticRegression(max_iter=1000), GaussianNB\n"
            "  - 树模型: DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier\n"
            "  - 梯度提升: GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier\n"
            "  - SVM: SVC(probability=True) [仅小数据]\n"
            "**回归任务**:\n"
            "  - 线性: LinearRegression, Ridge, Lasso, ElasticNet\n"
            "  - 树模型: DecisionTreeRegressor, RandomForestRegressor\n"
            "  - 梯度提升: GradientBoostingRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor\n"
            "  - 支持向量: SVR\n"
            "  - 计数回归: PoissonRegressor, GammaRegressor (非负整数目标)\n"
            "**聚类任务**:\n"
            "  - 划分: KMeans, KModes(纯分类), KPrototypes(混合)\n"
            "  - 密度: DBSCAN, HDBSCAN(如果已安装)\n"
            "  - 层次: AgglomerativeClustering\n"
            "  - 概率: GaussianMixture\n"
            "**降维任务**:\n"
            "  - 线性: PCA, TruncatedSVD, GaussianRandomProjection\n"
            "  - 非线性: KernelPCA\n"
            "  - 流形: TSNE, UMAP(如果已安装)\n\n"
            
            "### 阶段 6: 超参数调优（按场景选择）\n"
            "1. GridSearchCV: 参数空间小(<1000组合)，穷举搜索\n"
            "2. RandomizedSearchCV: 参数空间大，指定 n_iter=20/50\n"
            "3. HalvingGridSearchCV / HalvingRandomSearchCV: 连续减半，速度最快\n"
            "4. Optuna: 贝叶斯优化，最适合 XGBoost/LightGBM/CatBoost\n"
            "   示例框架:\n"
            "   import optuna\n"
            "   def objective(trial):\n"
            "       params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500),\n"
            "                 'max_depth': trial.suggest_int('max_depth', 3, 10)}\n"
            "       model = XGBClassifier(**params, random_state=42)\n"
            "       model.fit(X_train, y_train)\n"
            "       return accuracy_score(y_test, model.predict(X_test))\n"
            "   study = optuna.create_study(direction='maximize')\n"
            "   study.optimize(objective, n_trials=50)\n\n"
            
            "### 阶段 7: 模型评估\n"
            "**分类**: accuracy, precision, recall, f1, roc_auc, confusion_matrix, classification_report\n"
            "  - 必画: ROC曲线, PR曲线, 混淆矩阵热力图\n"
            "**回归**: MSE, RMSE, MAE, MAPE, R², 调整R²\n"
            "  - 必画: 预测vs真实值散点图, 残差图, 学习曲线\n"
            "**聚类**: silhouette_score, calinski_harabasz_score, davies_bouldin_score\n"
            "  - 必画: PCA降维后的聚类散点图, 轮廓系数图\n\n"
            
            "### 阶段 8: 模型解释\n"
            "1. 特征重要性: model.feature_importances_ (树模型)\n"
            "2. Permutation Importance: sklearn.inspection.permutation_importance\n"
            "3. SHAP:\n"
            "   import shap\n"
            "   explainer = shap.TreeExplainer(model)  # 树模型\n"
            "   # 或 explainer = shap.Explainer(model, X_train)  # 通用\n"
            "   shap_values = explainer(X_test)\n"
            "   shap.summary_plot(shap_values, X_test, show=False)  # 自动保存\n"
            "4. 模型代理: 用 DecisionTreeRegressor 拟合黑盒模型的预测，解释近似规则\n\n"
            
            "### 阶段 9: 产物保存\n"
            "使用 joblib 保存:\n"
            "  joblib.dump(best_model, './ml_outputs/models/best_model_{任务}_{算法}.pkl')\n"
            "  joblib.dump(pipeline, './ml_outputs/pipelines/feature_pipeline_{任务}.pkl')\n"
            "告知用户保存路径。\n\n"
            
            "## 图表规范\n"
            "- figsize=(10, 6) 或 (12, 8)\n"
            "- 标题中文, fontsize=16, fontweight='bold'\n"
            "- 轴标签中文, fontsize=12\n"
            "- x轴标签过长: plt.xticks(rotation=45)\n"
            "- 始终 plt.tight_layout()\n"
            "- 使用 Seaborn 调色板\n\n"
            
            "## 关键规则\n"
            "- 用户未指定目标变量时，主动询问或基于列名推断并说明\n"
            "- 分类任务前检查类别平衡，不平衡时提及并建议 class_weight='balanced' 或 SMOTE\n"
            "- 高基数类别(>10)优先用 TargetEncoder 或 CatBoost 原生处理，避免维度爆炸\n"
            "- GridSearchCV 参数网格不要过大（避免超时）\n"
            "- 始终 random_state=42\n"
            "- 内存不足时避免 pairplot，改用逐对 scatterplot\n"
            "- 保存模型前确保是训练好的最终模型（已 fit）"
        )
        
        self.agent = create_agent(
            model=self.llm,
            tools=[safe_python_repl_ml, create_chart_ml],
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
            middleware=[ModelCallLimitMiddleware(run_limit=60)],
        )
        self.session_id = str(uuid.uuid4())
        logger.info("✅ ML Agent 构建完成")
    
    # ==================== 流式查询 ====================
    def query_stream(self, question: str):
        """流式执行机器学习分析"""
        if not self.agent:
            yield "⚠️ 机器学习 Agent 尚未初始化！请先上传数据文件。"
            return
        
        global _ml_current_df
        if self.current_df_name and self.current_df_name in self.dataframes:
            _ml_current_df = self.dataframes[self.current_df_name]
        
        try:
            cfg = {"configurable": {"thread_id": self.session_id}}
            full_response = ""
            
            for event in self.agent.stream(
                {"messages": [("user", question)]},
                config=cfg,
                stream_mode="values"
            ):
                messages = event.get("messages", [])
                if not messages:
                    continue
                
                last_msg = messages[-1]
                if (isinstance(last_msg, AIMessage) and 
                    last_msg.content and 
                    not last_msg.tool_calls):
                    
                    if last_msg.content != full_response:
                        full_response = last_msg.content
                        yield full_response
            
            if not full_response:
                yield "✅ 分析完成，但未生成文本回复（可能仅执行了代码或保存了模型）。"
                
        except Exception as e:
            logger.error(f"❌ ML Agent 执行失败: {e}")
            yield f"❌ 分析出错: {str(e)}"
    
    # ==================== 记忆与图表管理 ====================
    def clear_memory(self):
        """清空机器学习记忆"""
        self.session_id = str(uuid.uuid4())
        return "✅ 机器学习记忆已清空！"
    
    def get_last_chart(self) -> Optional[str]:
        global _ml_last_chart_path
        if _ml_last_chart_path and Path(_ml_last_chart_path).exists():
            return _ml_last_chart_path
        return None
    
    def clear_last_chart(self):
        global _ml_last_chart_path
        _ml_last_chart_path = None
    
    def get_latest_model(self) -> Optional[str]:
        """获取最近保存的模型文件路径"""
        model_dir = _ML_OUT / "models"
        if not model_dir.exists():
            return None
        models = sorted(
            [p for p in model_dir.iterdir() if p.suffix in ('.pkl', '.joblib', '.pickle')],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return str(models[0]) if models else None
    
    def get_latest_pipeline(self) -> Optional[str]:
        """获取最近保存的管道文件路径"""
        pipe_dir = _ML_OUT / "pipelines"
        if not pipe_dir.exists():
            return None
        pipes = sorted(
            [p for p in pipe_dir.iterdir() if p.suffix in ('.pkl', '.joblib', '.pickle')],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return str(pipes[0]) if pipes else None