"""
G-Power V2 算法单元测试
覆盖：_compute_ewm_weights / _accumulate / _compute_scores 及边界情况
"""
import math
import sys
import os
import types

# ── 屏蔽 streamlit，让 app.py 可在无 UI 环境下 import ──────────────────────────
class _FakeCol:
    def __getattr__(self, name):
        return lambda *a, **kw: _FakeCol()
    def __enter__(self): return self
    def __exit__(self, *a): pass

def _fake_columns(*a, **kw):
    n = a[0] if a else 1
    count = n if isinstance(n, int) else len(n)
    return [_FakeCol() for _ in range(count)]

st_mock = types.ModuleType("streamlit")
for attr in ["set_page_config", "title", "caption", "divider", "subheader",
             "radio", "button", "file_uploader", "info", "success", "error",
             "warning", "progress", "empty", "spinner", "text_input",
             "expander", "checkbox", "markdown", "dataframe", "metric",
             "download_button", "rerun"]:
    setattr(st_mock, attr, lambda *a, **kw: None)
class _SessionState(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

st_mock.columns = _fake_columns
st_mock.session_state = _SessionState()
sys.modules["streamlit"] = st_mock

sys.path.insert(0, os.path.dirname(__file__))
from app import (
    _accumulate,
    _compute_ewm_weights,
    _compute_scores,
    BASELINE_WEIGHTS,
    CF_THRESHOLD,
    ALPHA_0,
    ALPHA_MAX,
    N_THRESHOLD,
)

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# _compute_ewm_weights
# ══════════════════════════════════════════════════════════════════════════════

def _make_raw(brands_scores: dict) -> dict:
    return {b: dict(zip(["V", "D", "R", "C", "A"], s))
            for b, s in brands_scores.items()}


def test_ewm_single_brand_returns_baseline():
    raw = _make_raw({"A": [80, 60, 70, 50, 65]})
    w = _compute_ewm_weights(raw, 1)
    assert w == BASELINE_WEIGHTS


def test_ewm_weights_sum_to_one():
    raw = _make_raw({
        "A": [80, 60, 70, 50, 65],
        "B": [45, 55, 62, 78, 40],
        "C": [20, 35, 68, 30,  0],
    })
    w = _compute_ewm_weights(raw, 3)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_ewm_high_variance_dim_gets_higher_weight():
    """差异大的维度应获得更高权重：构造 V 极度不均、D 完全均匀的场景。"""
    # V: [100, 0, 0] 极度集中；D: [50, 50, 50] 完全均匀（归一化后差异系数=0）
    raw = _make_raw({
        "X": [100, 50, 50],   # V差异大
        "Y": [  0, 50, 50],   # V差异大
        "Z": [ 50, 50, 50],   # V无差异
    })
    # 此处 V 列有差异，D/R/C/A 列全相同 → V 应获得最高权重
    raw = {
        "X": {"V": 100, "D": 50, "R": 50, "C": 50, "A": 50},
        "Y": {"V":   0, "D": 50, "R": 50, "C": 50, "A": 50},
        "Z": {"V":  50, "D": 50, "R": 50, "C": 50, "A": 50},
    }
    w = _compute_ewm_weights(raw, 3)
    # V 有差异，其余维度全相同（差异系数 0），V 应获得最高权重
    assert w["V"] == max(w.values()), f"V should be highest, got {w}"
    # 权重均在合理范围
    for v in w.values():
        assert 0.0 < v < 1.0


def test_ewm_identical_scores_returns_equal_weights():
    """所有品牌同维度分数一样时，权重退回均等。"""
    raw = _make_raw({
        "A": [50, 50, 50, 50, 50],
        "B": [50, 50, 50, 50, 50],
    })
    w = _compute_ewm_weights(raw, 2)
    for v in w.values():
        assert abs(v - 0.20) < 1e-6


def test_ewm_alpha_increases_with_brand_count():
    """品牌数越多，α 越大（最终权重中熵权占比更高）。"""
    def get_alpha(m):
        return min(ALPHA_MAX, ALPHA_0 + (ALPHA_MAX - ALPHA_0) * (m - 1) / (N_THRESHOLD - 1))

    assert get_alpha(1) <= get_alpha(5) <= get_alpha(10) <= get_alpha(20)
    assert get_alpha(50) == ALPHA_MAX


def test_ewm_two_brands_alpha():
    alpha = min(ALPHA_MAX, ALPHA_0 + (ALPHA_MAX - ALPHA_0) * (2 - 1) / (N_THRESHOLD - 1))
    assert ALPHA_0 < alpha < ALPHA_MAX


# ══════════════════════════════════════════════════════════════════════════════
# _accumulate
# ══════════════════════════════════════════════════════════════════════════════

def _empty_acc(entities):
    return {e: {"mentionedCount": 0, "depthScoreSum": 0.0, "sentimentSum": 0.0,
                "sentimentCount": 0, "competitivenessSum": 0.0, "rankSum": 0,
                "totalWordCount": 0, "citationAuthoritySum": 0.0,
                "citationCount": 0, "competitiveCount": 0}
            for e in entities}


def _audit(brand, rank, sentiment, citation, total_brands, evidence="text"):
    return {
        "total_brands_mentioned": total_brands,
        "all_mentioned_brands": [
            {"brand": b, "physical_rank": i + 1, "evidence_text": "x" * 10}
            for i, b in enumerate([brand] + [f"other{k}" for k in range(total_brands - 1)])
        ],
        "brand_analysis": [
            {
                "brand": brand,
                "is_mentioned": True,
                "evidence_text": evidence,
                "physical_rank": rank,
                "sentiment_score": sentiment,
                "citation_authority_score": citation,
            }
        ],
    }


def test_accumulate_sentiment_normalization():
    """sentiment -1~+1 应归一化为 0~100。"""
    acc = _empty_acc(["华为"])
    _accumulate(_audit("华为", 1, 1.0, None, 1, "a" * 20), ["华为"], acc)
    assert acc["华为"]["sentimentSum"] == pytest.approx(100.0)

    acc2 = _empty_acc(["华为"])
    _accumulate(_audit("华为", 1, -1.0, None, 1, "a" * 20), ["华为"], acc2)
    assert acc2["华为"]["sentimentSum"] == pytest.approx(0.0)

    acc3 = _empty_acc(["华为"])
    _accumulate(_audit("华为", 1, 0.0, None, 1, "a" * 20), ["华为"], acc3)
    assert acc3["华为"]["sentimentSum"] == pytest.approx(50.0)


def test_accumulate_citation_count_only_for_non_null():
    """citation_authority_score 为 null 时不计入 citationCount。"""
    acc = _empty_acc(["华为"])
    _accumulate(_audit("华为", 1, 0.5, None, 1), ["华为"], acc)
    assert acc["华为"]["citationCount"] == 0
    assert acc["华为"]["citationAuthoritySum"] == 0.0

    acc2 = _empty_acc(["华为"])
    _accumulate(_audit("华为", 1, 0.5, 70.0, 1), ["华为"], acc2)
    assert acc2["华为"]["citationCount"] == 1
    assert acc2["华为"]["citationAuthoritySum"] == pytest.approx(70.0)


def test_accumulate_competitive_count_requires_multiple_brands():
    """只有 1 个品牌被提及时不计入 competitiveCount。"""
    acc = _empty_acc(["华为"])
    _accumulate(_audit("华为", 1, 0.0, None, 1), ["华为"], acc)
    assert acc["华为"]["competitiveCount"] == 0

    acc2 = _empty_acc(["华为"])
    _accumulate(_audit("华为", 1, 0.0, None, 3), ["华为"], acc2)
    assert acc2["华为"]["competitiveCount"] == 1


def test_accumulate_not_mentioned_skipped():
    """is_mentioned=false 的品牌不影响 acc。"""
    entities = ["华为", "小米"]
    acc = _empty_acc(entities)
    result = {
        "total_brands_mentioned": 0,
        "all_mentioned_brands": [],
        "brand_analysis": [
            {"brand": "华为", "is_mentioned": False, "evidence_text": "",
             "physical_rank": 0, "sentiment_score": None, "citation_authority_score": None},
            {"brand": "小米", "is_mentioned": False, "evidence_text": "",
             "physical_rank": 0, "sentiment_score": None, "citation_authority_score": None},
        ],
    }
    _accumulate(result, entities, acc)
    assert acc["华为"]["mentionedCount"] == 0
    assert acc["小米"]["mentionedCount"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# _compute_scores
# ══════════════════════════════════════════════════════════════════════════════

def _full_acc(entity, mc, sentiment_sum, sentiment_count,
              depth_sum, comp_sum, comp_count, rank_sum,
              citation_sum, citation_count, word_count, Q):
    return {entity: {
        "mentionedCount": mc,
        "depthScoreSum": depth_sum,
        "sentimentSum": sentiment_sum,
        "sentimentCount": sentiment_count,
        "competitivenessSum": comp_sum,
        "competitiveCount": comp_count,
        "rankSum": rank_sum,
        "citationAuthoritySum": citation_sum,
        "citationCount": citation_count,
        "totalWordCount": word_count,
    }}


def test_compute_scores_basic_range():
    """所有得分应在 0~100，geoScore 也应 ≤ 100。"""
    acc = _full_acc("华为", mc=8, sentiment_sum=400, sentiment_count=8,
                    depth_sum=320, comp_sum=480, comp_count=6, rank_sum=8,
                    citation_sum=560, citation_count=8, word_count=800, Q=10)
    scores, weights = _compute_scores(acc, ["华为"], Q=10)
    d = scores["华为"]
    assert 0 <= d["visibilityScore"] <= 100
    assert 0 <= d["depthScore"] <= 100
    assert 0 <= d["recommendationScore"] <= 100
    assert 0 <= d["competitivenessScore"] <= 100
    assert 0 <= d["citationAuthorityScore"] <= 100
    assert 0 <= d["geoScore"] <= 100


def test_compute_scores_never_mentioned_brand():
    """从未被提及的品牌，所有分数为 0，GEO 得分为 0。"""
    acc = _full_acc("冷门品牌", mc=0, sentiment_sum=0, sentiment_count=0,
                    depth_sum=0, comp_sum=0, comp_count=0, rank_sum=0,
                    citation_sum=0, citation_count=0, word_count=0, Q=10)
    scores, _ = _compute_scores(acc, ["冷门品牌"], Q=10)
    d = scores["冷门品牌"]
    assert d["visibilityScore"] == 0.0
    assert d["geoScore"] == 0.0


def test_compute_scores_r_uses_sentiment_count_not_mc():
    """R 分应使用 sentimentCount 作分母，而非 mc。"""
    # mc=5，但只有 3 条有 sentiment
    acc = _full_acc("华为", mc=5, sentiment_sum=150.0, sentiment_count=3,
                    depth_sum=200, comp_sum=300, comp_count=4, rank_sum=5,
                    citation_sum=0, citation_count=0, word_count=300, Q=10)
    scores, _ = _compute_scores(acc, ["华为"], Q=10)
    assert scores["华为"]["recommendationScore"] == pytest.approx(50.0, abs=0.1)


def test_compute_scores_a_uses_citation_count_not_mc():
    """A 分应使用 citationCount 作分母，而非 mc。"""
    # mc=5，但只有 2 条有 citation
    acc = _full_acc("华为", mc=5, sentiment_sum=250, sentiment_count=5,
                    depth_sum=200, comp_sum=300, comp_count=4, rank_sum=5,
                    citation_sum=140.0, citation_count=2, word_count=300, Q=10)
    scores, _ = _compute_scores(acc, ["华为"], Q=10)
    assert scores["华为"]["citationAuthorityScore"] == pytest.approx(70.0, abs=0.1)


def test_compute_scores_weights_sum_to_one():
    """返回的 weights 之和应为 1。"""
    acc1 = _full_acc("A", mc=8, sentiment_sum=400, sentiment_count=8,
                     depth_sum=320, comp_sum=480, comp_count=6, rank_sum=8,
                     citation_sum=480, citation_count=8, word_count=600, Q=10)
    acc2 = _full_acc("B", mc=5, sentiment_sum=250, sentiment_count=5,
                     depth_sum=200, comp_sum=300, comp_count=4, rank_sum=10,
                     citation_sum=200, citation_count=4, word_count=400, Q=10)
    combined = {**acc1, **acc2}
    _, weights = _compute_scores(combined, ["A", "B"], Q=10)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_compute_scores_cf_caps_at_one():
    """CF = min(1, sqrt(n/K))，当 n >= K 时应为 1.0，得分不被压缩。"""
    # mc=10 = CF_THRESHOLD，CF_D=CF_R=CF_A=1.0
    acc = _full_acc("华为", mc=10, sentiment_sum=500, sentiment_count=10,
                    depth_sum=400, comp_sum=600, comp_count=10, rank_sum=10,
                    citation_sum=700, citation_count=10, word_count=1000, Q=10)
    scores, _ = _compute_scores(acc, ["华为"], Q=10)
    # V: Q=10=CF_THRESHOLD → CF_V=1.0，visibility=100%
    assert scores["华为"]["visibilityScore"] == pytest.approx(100.0)
    # GEO 应接近真实加权分（无打折）
    assert scores["华为"]["geoScore"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
