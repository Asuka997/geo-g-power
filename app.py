"""
GEO 大模型品牌表现分析平台 — Streamlit 版
数据格式：CSV 或 Excel，第1列=问题，第2列=回答（支持有/无表头）
"""

import streamlit as st
import pandas as pd
import requests
import json
import math
import time
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime

# ─── 配置 ────────────────────────────────────────────────────────────────────
import os
_API_KEY      = os.environ.get("OPENROUTER_API_KEY", "")
_ENDPOINT_ID  = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
_BASE_URL     = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
_TIMEOUT      = 300
_MAX_RETRIES  = 4
_CONCURRENCY  = 10

CF_THRESHOLD = 10
ALPHA_0 = 0.3
ALPHA_MAX = 0.8
N_THRESHOLD = 20
BASELINE_WEIGHTS = {"V": 0.20, "D": 0.20, "R": 0.20, "C": 0.20, "A": 0.20}

ENTITY_CONFIG = {
    "brand":   {"type": "品牌名称",   "examples": "小米、华为、苹果、OPPO、vivo"},
    "product": {"type": "具体产品型号", "examples": "小米13 Ultra、华为P60 Pro、iPhone 15 Pro Max"},
}

DEFAULT_CATEGORY = {
    "brand":   "手机品牌",
    "product": "手机产品型号",
}

# ─── Prompts ──────────────────────────────────────────────────────────────────
def _prompt_extract(entity_type, examples, text, extract_scope: str = ""):
    scope_line = f"\n提取范围限定：{extract_scope}（仅提取符合该描述的{entity_type}，不符合的一律忽略）" if extract_scope else ""
    return f"""请从以下文本中提取所有提及的{entity_type}。{scope_line}

要求：
1. 只提取{entity_type}，不要提取其他内容
2. 标准化名称（如"小米13Ultra"统一为"小米13 Ultra"）
3. 去重，每个实体只出现一次
4. 返回JSON格式：{{"entities": ["实体1", "实体2", ...]}}

示例{entity_type}：{examples}

文本内容：
{text}

请只返回JSON，不要任何解释。"""


def _prompt_audit(brand_list, category_def, answer_text):
    brands_str = "\n".join(f"- {b}" for b in brand_list)
    return f"""你是一位专业的品牌分析审计员。请严格分析以下回答文本，一次性完成全量品牌扫描与监测品牌详细分析。

**品类定义：** {category_def}（仅计入与目标品类构成直接购买替代关系的品牌；操作系统、芯片品牌、电商平台等上下游实体不计入）

**监测品牌列表：**
{brands_str}

**回答原文：**
{answer_text}

**输出格式（严格JSON，不输出任何其他内容）：**
{{
  "total_brands_mentioned": <整数，原文中同品类品牌总数，含非监测品牌>,
  "all_mentioned_brands": [
    {{
      "brand": "品牌名称",
      "physical_rank": <排名整数，按推荐顺序；无明确顺序则按首次出现先后>,
      "evidence_text": "该品牌在原文中所有相关语句的原文摘录，禁止改写"
    }}
  ],
  "brand_analysis": [
    {{
      "brand": "品牌名称（与监测列表保持一致）",
      "is_mentioned": true或false,
      "evidence_text": "所有相关原文语句（未提及输出空字符串）",
      "physical_rank": <在所有同品类品牌中的排名；未提及输出0>,
      "sentiment_score": <-1.0到+1.0的浮点数，+1.0=极力推荐，0=中立，-1.0=极度负面；未提及输出null>,
      "citation_authority_score": <0-100的浮点数，评估回答中提及该品牌时所依据来源的权威性：权威媒体/政府/学术机构→100，行业垂直媒体→70，一般网站/博客→40，无任何来源依据→0；未提及输出null>
    }}
  ]
}}

约束：
- all_mentioned_brands数组长度必须等于total_brands_mentioned
- 所有条目的physical_rank必须连续且唯一（1到total_brands_mentioned），不得重复或缺失
- brand_analysis中有提及品牌的physical_rank必须与all_mentioned_brands中对应条目一致
- 别名/简称/旗下产品均算提及母品牌（如iPhone→Apple，华子→华为）
- 以对比方式出现的品牌也算提及
- 禁止AI直接统计字数，evidence_text只摘录原文

请只返回JSON，不要任何解释。"""


# ─── API 调用 ─────────────────────────────────────────────────────────────────
def _call_api(prompt: str, max_tokens: int = 1000) -> str:
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _ENDPOINT_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(_BASE_URL, headers=headers, json=payload, timeout=_TIMEOUT)
            if resp.status_code == 429:
                delay = min(5 * (2 ** attempt), 20)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                time.sleep(2)
    raise RuntimeError(f"API调用失败: {last_err}")


# ─── 文件解析 ─────────────────────────────────────────────────────────────────
def parse_uploaded_file(uploaded_file) -> list[dict]:
    """返回 [{"question": ..., "answer": ...}, ...]"""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # 尝试检测编码
        raw = uploaded_file.read()
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        df = pd.read_csv(io.StringIO(text), header=None)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, header=None)
    else:
        raise ValueError("不支持的格式，请上传 CSV 或 Excel 文件")

    # 如果第一行看起来像表头则丢掉
    first_row = [str(df.iloc[0, c]).strip() for c in range(min(2, df.shape[1]))]
    if any(h in first_row for h in ["问题", "question", "Question", "回答", "answer", "Answer"]):
        df = df.iloc[1:].reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        if df.shape[1] < 2:
            continue
        q = str(row.iloc[0]).strip()
        a = str(row.iloc[1]).strip()
        if q and a and q != "nan" and a != "nan":
            rows.append({"question": q, "answer": a})
    return rows


# ─── 实体提取 ─────────────────────────────────────────────────────────────────
def extract_entities(answers: list[str], dimension: str,
                     extract_scope: str = "",
                     progress_bar=None, status_text=None) -> list[tuple[str, int]]:
    cfg = ENTITY_CONFIG[dimension]
    mention_counts: dict[str, int] = {}
    lock = Lock()
    total = len(answers)

    def process_one(answer: str):
        prompt = _prompt_extract(cfg["type"], cfg["examples"], answer, extract_scope)
        try:
            response = _call_api(prompt, max_tokens=500)
            m = re.search(r'\{[\s\S]*\}', response)
            if m:
                data = json.loads(m.group(0))
                # 对单条回答内去重，避免 LLM 返回重复实体导致计数翻倍
                seen = set()
                for name in data.get("entities", []):
                    n = name.strip()
                    if n and n not in seen:
                        seen.add(n)
                        with lock:
                            mention_counts[n] = mention_counts.get(n, 0) + 1
        except Exception:
            pass

    # UI 更新必须在主线程（as_completed 循环运行在主线程）
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
        futures = [ex.submit(process_one, a) for a in answers]
        for i, f in enumerate(as_completed(futures), 1):
            f.result()
            if progress_bar:
                progress_bar.progress(i / total)
            if status_text:
                status_text.text(f"提取中… {i} / {total} 条")

    return sorted(mention_counts.items(), key=lambda x: -x[1])


# ─── GEO 分析 ─────────────────────────────────────────────────────────────────
def _parse_audit_json(response: str) -> dict:
    m = re.search(r'\{[\s\S]*\}', response)
    if not m:
        raise ValueError("未找到有效 JSON")
    json_str = m.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 截断修复
        last = json_str.rfind('},')
        if last > 0:
            json_str = json_str[:last + 1] + ']}}'
        else:
            json_str = re.sub(r',?\s*\{[^}]*$', '', json_str) + ']}}'
        return json.loads(json_str)


def _accumulate(audit_result: dict, entities: list[str], acc: dict):
    all_brands = audit_result.get("all_mentioned_brands", [])
    total_brands = len(all_brands)
    x_max = max((len((b.get("evidence_text") or "").strip()) for b in all_brands), default=0)
    brand_analysis = audit_result.get("brand_analysis", [])

    for entity in entities:
        bd = next((b for b in brand_analysis if b["brand"] == entity), None)
        if not bd or bd.get("is_mentioned") is not True:
            continue
        word_count = len((bd.get("evidence_text") or "").strip())
        acc[entity]["mentionedCount"] += 1
        acc[entity]["totalWordCount"] += word_count
        if x_max > 0:
            acc[entity]["depthScoreSum"] += (word_count / x_max) * 100
        # sentiment: -1.0~+1.0 → 归一化为 0~100
        s = bd.get("sentiment_score")
        if s is not None:
            acc[entity]["sentimentSum"] += (float(s) + 1.0) / 2.0 * 100
            acc[entity]["sentimentCount"] += 1
        rank = bd.get("physical_rank") or 0
        if rank > 0 and total_brands > 0:
            acc[entity]["competitivenessSum"] += ((total_brands - rank + 1) / total_brands) * 100
            acc[entity]["rankSum"] += rank
            if total_brands > 1:
                acc[entity]["competitiveCount"] += 1
        # A 维度：引用可信度
        ca = bd.get("citation_authority_score")
        if ca is not None:
            acc[entity]["citationAuthoritySum"] += float(ca)
            acc[entity]["citationCount"] += 1


def _compute_ewm_weights(raw_scores: dict, m: int) -> dict:
    """信息熵自适应权重 + 基线混合。raw_scores: {brand: {dim: score_0_100}}"""
    dims = ["V", "D", "R", "C", "A"]
    brands = list(raw_scores.keys())

    if m <= 1:
        return BASELINE_WEIGHTS.copy()

    matrix = [[raw_scores[b][d] for d in dims] for b in brands]

    # 极差归一化
    norm = []
    for j, d in enumerate(dims):
        col = [matrix[i][j] for i in range(m)]
        mn, mx = min(col), max(col)
        if mx == mn:
            norm.append([0.0] * m)
        else:
            norm.append([(col[i] - mn) / (mx - mn) for i in range(m)])

    # 贡献比例 p_ij
    entropy_weights = []
    for j in range(len(dims)):
        col_sum = sum(norm[j])
        if col_sum == 0:
            p = [1.0 / m] * m
        else:
            p = [v / col_sum for v in norm[j]]
        # 信息熵 e_j
        ln_m = math.log(m)
        e = -1.0 / ln_m * sum((pi * math.log(pi) if pi > 0 else 0.0) for pi in p)
        entropy_weights.append(1.0 - e)  # 差异系数 d_j

    d_sum = sum(entropy_weights)
    if d_sum == 0:
        w_entropy = {d: 0.20 for d in dims}
    else:
        w_entropy = {dims[j]: entropy_weights[j] / d_sum for j in range(len(dims))}

    # 自适应 α（m >= 2，因为 m <= 1 已 early return）
    alpha = min(ALPHA_MAX, ALPHA_0 + (ALPHA_MAX - ALPHA_0) * (m - 1) / (N_THRESHOLD - 1))

    # 混合权重
    mixed = {d: alpha * w_entropy[d] + (1 - alpha) * BASELINE_WEIGHTS[d] for d in dims}
    total = sum(mixed.values())
    return {d: mixed[d] / total for d in dims}


def _compute_scores(acc: dict, entities: list[str], Q: int) -> tuple[dict, dict]:
    # 第一步：计算每个品牌的五维原始分
    raw: dict[str, dict] = {}
    for entity in entities:
        d = acc[entity]
        mc = d["mentionedCount"]
        v_score = (mc / Q) * 100 if Q > 0 else 0.0
        d_score = d["depthScoreSum"] / Q if Q > 0 else 0.0
        r_score = d["sentimentSum"] / d["sentimentCount"] if d["sentimentCount"] > 0 else 0.0
        c_score = d["competitivenessSum"] / Q if Q > 0 else 0.0
        a_score = d["citationAuthoritySum"] / d["citationCount"] if d["citationCount"] > 0 else 0.0
        raw[entity] = {"V": v_score, "D": d_score, "R": r_score, "C": c_score, "A": a_score}

    # 第二步：信息熵自适应权重
    m = len(entities)
    weights = _compute_ewm_weights(raw, m)

    # 第三步：计算各维度 CF 并得到最终 G-Power
    result = {}
    for entity in entities:
        d = acc[entity]
        mc = d["mentionedCount"]
        cf = {
            "V": min(1.0, math.sqrt(Q / CF_THRESHOLD)),
            "D": min(1.0, math.sqrt(mc / CF_THRESHOLD)) if mc > 0 else 0.0,
            "R": min(1.0, math.sqrt(mc / CF_THRESHOLD)) if mc > 0 else 0.0,
            "C": min(1.0, math.sqrt(d["competitiveCount"] / CF_THRESHOLD)) if d["competitiveCount"] > 0 else 0.0,
            "A": min(1.0, math.sqrt(mc / CF_THRESHOLD)) if mc > 0 else 0.0,
        }
        scores = raw[entity]
        geo = sum(weights[dim] * cf[dim] * scores[dim] for dim in ["V", "D", "R", "C", "A"])
        result[entity] = {
            "count":                   mc,
            "avgWords":                round(d["totalWordCount"] / mc) if mc > 0 else 0,
            "avgRank":                 round(d["rankSum"] / mc, 1) if mc > 0 else None,
            "visibilityScore":         round(scores["V"], 1),
            "depthScore":              round(scores["D"], 1),
            "recommendationScore":     round(scores["R"], 1),
            "competitivenessScore":    round(scores["C"], 1),
            "citationAuthorityScore":  round(scores["A"], 1),
            "geoScore":                round(min(geo, 100.0), 1),
        }
    return result, weights


def run_geo_analysis(answers: list[str], entities: list[str],
                     dimension: str, category_def: str,
                     progress_bar=None, status_text=None) -> dict:
    Q = len(answers)
    max_tokens = min(128000, max(4000, len(entities) * 400 + 1000))

    acc = {e: {"mentionedCount": 0, "depthScoreSum": 0.0, "sentimentSum": 0.0,
               "sentimentCount": 0, "competitivenessSum": 0.0, "rankSum": 0,
               "totalWordCount": 0, "citationAuthoritySum": 0.0,
               "citationCount": 0, "competitiveCount": 0} for e in entities}
    lock = Lock()
    failed = []

    def process_one(idx: int, answer: str):
        for attempt in range(_MAX_RETRIES + 1):
            try:
                prompt = _prompt_audit(entities, category_def, answer)
                resp = _call_api(prompt, max_tokens=max_tokens)
                result = _parse_audit_json(resp)
                with lock:
                    _accumulate(result, entities, acc)
                break
            except Exception as e:
                if attempt == _MAX_RETRIES:
                    with lock:
                        failed.append(idx + 1)
                else:
                    delay = min(5 * (2 ** attempt), 20)
                    time.sleep(delay)

    # UI 更新必须在主线程（as_completed 循环运行在主线程）
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
        futures = [ex.submit(process_one, i, a) for i, a in enumerate(answers)]
        for i, f in enumerate(as_completed(futures), 1):
            f.result()
            if progress_bar:
                progress_bar.progress(i / Q)
            if status_text:
                status_text.text(f"分析中… {i} / {Q} 条，失败 {len(failed)} 条")

    scores, weights = _compute_scores(acc, entities, Q)
    return scores, failed, weights


# ─── 结果转 DataFrame ──────────────────────────────────────────────────────────
def scores_to_df(scores: dict, total_q: int) -> pd.DataFrame:
    rows = []
    for name, d in scores.items():
        rows.append({
            "品牌/产品":     name,
            "提及次数":     d["count"],
            "提及率(%)":   round(d["count"] / total_q * 100, 1) if total_q else 0,
            "可见度(V)":    d["visibilityScore"],
            "提及深度(D)":  d["depthScore"],
            "推荐倾向(R)":  d["recommendationScore"],
            "竞争力(C)":    d["competitivenessScore"],
            "引用可信度(A)": d["citationAuthorityScore"],
            "平均排名":     d["avgRank"] if d["avgRank"] is not None else "-",
            "GEO得分":      d["geoScore"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("GEO得分", ascending=False).reset_index(drop=True)
        df.insert(0, "排名", range(1, len(df) + 1))
    return df


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEO 大模型品牌表现分析平台",
    page_icon="📊",
    layout="wide",
)

st.title("📊 GEO 大模型品牌表现分析平台")
st.caption("智能分析品牌在AI时代的提及表现")

# 初始化 session state
for key, default in [
    ("step", 1),
    ("dimension", "brand"),
    ("qa_data", []),
    ("source_file_name", ""),
    ("extracted_entities", []),   # [(name, count)]
    ("selected_brands", []),
    ("category_def", ""),
    ("scores", {}),
    ("failed_items", []),
    ("total_q", 0),
    ("ewm_weights", {}),
    ("extract_scope", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 进度指示 ───────────────────────────────────────────────────────────────────
step_labels = ["选择维度", "上传数据", "提取品牌", "GEO分析", "查看结果"]
cols = st.columns(len(step_labels))
for i, (col, label) in enumerate(zip(cols, step_labels), start=1):
    if i < st.session_state.step:
        col.markdown(f"✅ **{label}**")
    elif i == st.session_state.step:
        col.markdown(f"▶️ **:blue[{label}]**")
    else:
        col.markdown(f"⬜ {label}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — 选择维度
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.subheader("第 1 步：选择分析维度")
    dim = st.radio("分析对象", ["品牌（brand）", "产品型号（product）"],
                   index=0 if st.session_state.dimension == "brand" else 1)
    if st.button("下一步", type="primary"):
        st.session_state.dimension = "brand" if "brand" in dim else "product"
        st.session_state.step = 2
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 上传数据
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.subheader("第 2 步：上传问答数据")
    st.info("文件格式：CSV 或 Excel，**第1列 = 问题，第2列 = 回答**（有无表头均可）")

    uploaded = st.file_uploader("选择文件", type=["csv", "xlsx", "xls"])

    if uploaded:
        try:
            qa_data = parse_uploaded_file(uploaded)
            st.success(f"解析成功，共 **{len(qa_data)}** 条问答数据")
            preview = pd.DataFrame(qa_data[:5])
            st.dataframe(preview, width='stretch')

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("下一步", type="primary"):
                    st.session_state.qa_data = qa_data
                    st.session_state.source_file_name = uploaded.name
                    st.session_state.total_q = len(qa_data)
                    st.session_state.step = 3
                    st.rerun()
        except Exception as e:
            st.error(f"文件解析失败：{e}")

    if st.button("← 返回"):
        st.session_state.step = 1
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 自动提取品牌 + 选择
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    dim_text = "品牌" if st.session_state.dimension == "brand" else "产品型号"
    st.subheader(f"第 3 步：提取 & 选择监测{dim_text}")

    qa_data = st.session_state.qa_data
    answers = [q["answer"] for q in qa_data]

    # ── 品类定义 ────────────────────────────────────────────────────────────
    default_cat = st.session_state.category_def or DEFAULT_CATEGORY[st.session_state.dimension]
    category_def = st.text_input(
        "品类定义（例：手机品牌 / 新能源汽车品牌）",
        value=default_cat,
    )
    st.session_state.category_def = category_def

    extract_scope = st.text_input(
        "提取类别说明（选填）",
        value=st.session_state.get("extract_scope", ""),
        placeholder="精确限定提取范围，例：只提取 AI Agent 编排框架，排除通信协议（MCP/A2A）、云平台和底层模型",
    )
    st.session_state.extract_scope = extract_scope

    # ── 提取按钮 ────────────────────────────────────────────────────────────
    if st.button("🔍 从回答中自动提取", type="primary",
                 disabled=len(st.session_state.extracted_entities) > 0):
        progress_bar = st.progress(0.0)
        status_text  = st.empty()
        with st.spinner("正在分析回答，提取品牌…"):
            result = extract_entities(
                answers, st.session_state.dimension,
                extract_scope=st.session_state.extract_scope,
                progress_bar=progress_bar, status_text=status_text,
            )
        st.session_state.extracted_entities = result
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    # ── 重新提取 ────────────────────────────────────────────────────────────
    if st.session_state.extracted_entities:
        if st.button("🔄 重新提取"):
            for name, _ in st.session_state.extracted_entities:
                st.session_state.pop(f"cb_{name}", None)
            st.session_state.extracted_entities = []
            st.session_state.selected_brands = []
            st.rerun()

    # ── 显示勾选列表 ─────────────────────────────────────────────────────────
    if st.session_state.extracted_entities:
        extracted = st.session_state.extracted_entities  # [(name, count)]
        st.markdown(f"**共提取到 {len(extracted)} 个{dim_text}**，请勾选要分析的项：")

        # 默认全选
        if not st.session_state.selected_brands:
            st.session_state.selected_brands = [name for name, _ in extracted]

        col_a, col_b, col_c = st.columns([1, 1, 6])
        with col_a:
            if st.button("全选"):
                st.session_state.selected_brands = [name for name, _ in extracted]
                for name, _ in extracted:
                    st.session_state[f"cb_{name}"] = True
                st.rerun()
        with col_b:
            if st.button("全不选"):
                st.session_state.selected_brands = []
                for name, _ in extracted:
                    st.session_state[f"cb_{name}"] = False
                st.rerun()

        selected_set = set(st.session_state.selected_brands)
        new_selected = []

        # 分两列展示
        n = len(extracted)
        half = math.ceil(n / 2)
        left_col, right_col = st.columns(2)

        for i, (name, count) in enumerate(extracted):
            col = left_col if i < half else right_col
            # 仅在首次渲染时初始化，避免与 session_state 冲突
            if f"cb_{name}" not in st.session_state:
                st.session_state[f"cb_{name}"] = (name in selected_set)
            checked = col.checkbox(f"{name}　`{count} 条`", key=f"cb_{name}")
            if checked:
                new_selected.append(name)

        st.session_state.selected_brands = new_selected

        # 手动添加
        with st.expander("手动添加品牌"):
            manual = st.text_input("品牌名称", key="manual_input")
            if st.button("添加") and manual.strip():
                name = manual.strip()
                if name not in [n for n, _ in st.session_state.extracted_entities]:
                    st.session_state.extracted_entities = [(name, 0)] + list(
                        st.session_state.extracted_entities
                    )
                if name not in st.session_state.selected_brands:
                    st.session_state.selected_brands = [name] + st.session_state.selected_brands
                st.rerun()

        st.divider()
        st.markdown(f"已选 **{len(new_selected)}** 个{dim_text}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("开始分析 →", type="primary", disabled=len(new_selected) == 0):
                st.session_state.step = 4
                st.rerun()
        with col2:
            if st.button("← 返回"):
                st.session_state.step = 2
                st.rerun()

    else:
        if st.button("← 返回"):
            st.session_state.step = 2
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GEO 分析
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    st.subheader("第 4 步：GEO 分析")

    qa_data    = st.session_state.qa_data
    answers    = [q["answer"] for q in qa_data]
    entities   = st.session_state.selected_brands
    dim        = st.session_state.dimension
    cat_def    = st.session_state.category_def or DEFAULT_CATEGORY[dim]

    st.info(f"共 **{len(answers)}** 条回答，监测 **{len(entities)}** 个品牌，并发数 {_CONCURRENCY}")

    if st.button("▶ 开始分析", type="primary"):
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        with st.spinner("GEO 分析中，请稍候…"):
            scores, failed, weights = run_geo_analysis(
                answers, entities, dim, cat_def,
                progress_bar=progress_bar,
                status_text=status_text,
            )

        st.session_state.scores       = scores
        st.session_state.failed_items = failed
        st.session_state.ewm_weights  = weights
        progress_bar.empty()
        status_text.empty()

        if failed:
            st.warning(f"有 {len(failed)} 条分析失败（第 {failed} 条），结果为部分数据。")

        st.session_state.step = 5
        st.rerun()

    if st.button("← 返回"):
        st.session_state.step = 3
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — 查看结果
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    st.subheader("第 5 步：GEO 指数排名结果")

    scores   = st.session_state.scores
    total_q  = st.session_state.total_q
    dim_text = "品牌" if st.session_state.dimension == "brand" else "产品型号"

    if not scores:
        st.warning("暂无分析结果，请返回重新分析。")
    else:
        df = scores_to_df(scores, total_q)

        # 指标卡
        top = df.iloc[0] if len(df) > 0 else None
        c1, c2, c3 = st.columns(3)
        c1.metric("监测数量", f"{len(df)} 个{dim_text}")
        c2.metric("问答数量", f"{total_q} 条")
        if top is not None:
            c3.metric("GEO第一", f"{top['品牌/产品']}  {top['GEO得分']} 分")

        st.divider()

        # 结果表
        st.dataframe(df, width='stretch', height=400)

        # EWM 权重快照
        if st.session_state.ewm_weights:
            with st.expander("本批次 G-Power V2 权重快照（信息熵自适应）"):
                w = st.session_state.ewm_weights
                wc1, wc2, wc3, wc4, wc5 = st.columns(5)
                wc1.metric("可见度 V", f"{w.get('V', 0):.1%}")
                wc2.metric("提及深度 D", f"{w.get('D', 0):.1%}")
                wc3.metric("推荐倾向 R", f"{w.get('R', 0):.1%}")
                wc4.metric("竞争力 C", f"{w.get('C', 0):.1%}")
                wc5.metric("引用可信度 A", f"{w.get('A', 0):.1%}")

        # 下载
        base_name = st.session_state.source_file_name
        if base_name:
            base_name = re.sub(r'\.[^.]+$', '', base_name) + "_GEO分析"
        else:
            base_name = f"{dim_text}提及分析"
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 权重快照 DataFrame（附在导出数据末尾）
        w = st.session_state.ewm_weights
        weight_df = pd.DataFrame([{
            "维度": "可见度(V)", "权重": f"{w.get('V', 0):.4f}",
        }, {
            "维度": "提及深度(D)", "权重": f"{w.get('D', 0):.4f}",
        }, {
            "维度": "推荐倾向(R)", "权重": f"{w.get('R', 0):.4f}",
        }, {
            "维度": "竞争力(C)", "权重": f"{w.get('C', 0):.4f}",
        }, {
            "维度": "引用可信度(A)", "权重": f"{w.get('A', 0):.4f}",
        }]) if w else pd.DataFrame()

        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 5])
        with col_dl1:
            # CSV：品牌结果 + 空行 + 权重快照
            csv_parts = [df.to_csv(index=False, encoding="utf-8-sig")]
            if not weight_df.empty:
                csv_parts += ["\n本批次G-Power V2权重快照\n",
                              weight_df.to_csv(index=False, encoding="utf-8-sig")]
            csv_bytes = "".join(csv_parts).encode("utf-8-sig")
            st.download_button(
                "⬇ 导出 CSV",
                data=csv_bytes,
                file_name=f"{base_name}_{date_str}.csv",
                mime="text/csv",
            )
        with col_dl2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="GEO分析")
                if not weight_df.empty:
                    weight_df.to_excel(writer, index=False, sheet_name="权重快照")
            st.download_button(
                "⬇ 导出 Excel",
                data=buf.getvalue(),
                file_name=f"{base_name}_{date_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.divider()
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            if st.button("🔄 重新分析（换品牌）"):
                st.session_state.step = 3
                st.rerun()
        with col_r2:
            if st.button("🆕 重置，从头开始"):
                for key in ["step", "dimension", "qa_data", "source_file_name",
                            "extracted_entities", "selected_brands", "category_def",
                            "scores", "failed_items", "total_q", "ewm_weights",
                            "extract_scope"]:
                    del st.session_state[key]
                st.rerun()
