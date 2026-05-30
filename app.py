"""
GNSS Observation Quality Analyzer - Streamlit Web Interface
运行: streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rinex_reader import RinexObs
from gnss_quality import (
    snr_analysis,
    cycle_slip_detect,
    multipath_calc,
    visibility_stats,
    quality_summary,
    compute_quality_metrics,
    quality_evaluation_report,
    snr_statistics,
)
from coord_convert import (
    blh_to_xyz,
    xyz_to_blh,
    gauss_projection,
    gauss_inverse,
    deg_to_dms,
    convert_summary,
    WGS84,
    CGCS2000,
)

st.set_page_config(
    page_title="GNSS Observation Quality Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Visual system: clean web showcase style
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #06101f;
        --bg2: #0b1628;
        --surface: rgba(12, 22, 40, 0.82);
        --surface2: rgba(17, 31, 55, 0.78);
        --line: rgba(148, 163, 184, 0.20);
        --line2: rgba(56, 189, 248, 0.36);
        --text: #eaf2ff;
        --muted: #93a4ba;
        --blue: #38bdf8;
        --cyan: #67e8f9;
        --green: #4ade80;
        --amber: #fbbf24;
        --red: #fb7185;
        --violet: #a78bfa;
    }

    html, body, [class*="css"] { font-family: Inter, "Microsoft YaHei", "PingFang SC", Arial, sans-serif; }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 14% 4%, rgba(56, 189, 248, .24), transparent 26rem),
            radial-gradient(circle at 86% 10%, rgba(167, 139, 250, .18), transparent 24rem),
            radial-gradient(circle at 50% 95%, rgba(34, 197, 94, .10), transparent 34rem),
            linear-gradient(145deg, #040a14 0%, #071427 42%, #101827 100%);
    }

    header, footer, #MainMenu { visibility: hidden; }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.4rem;
        max-width: 1480px;
    }

    /* Streamlit widgets */
    div[data-testid="stFileUploader"] section {
        border: 1px dashed rgba(103, 232, 249, .42);
        background: rgba(8, 18, 34, .65);
        border-radius: 22px;
        padding: 1.05rem;
    }
    div[data-testid="stFileUploader"] button,
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px !important;
        border: 1px solid rgba(56, 189, 248, .55) !important;
        color: #eff6ff !important;
        background: linear-gradient(135deg, rgba(56, 189, 248, .25), rgba(37, 99, 235, .18)) !important;
        font-weight: 760 !important;
        box-shadow: 0 12px 30px rgba(14, 165, 233, .10);
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: rgba(103, 232, 249, .95) !important;
        box-shadow: 0 16px 34px rgba(14, 165, 233, .18);
    }
    div[data-testid="stMetric"] {
        padding: .9rem .95rem;
        background: rgba(12, 22, 40, .76);
        border: 1px solid var(--line);
        border-radius: 18px;
    }
    div[data-testid="stMetric"] label { color: var(--muted) !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: .5rem;
        background: rgba(5, 12, 24, .45);
        padding: .35rem;
        border: 1px solid rgba(148, 163, 184, .14);
        border-radius: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 14px;
        padding: 0 1rem;
        color: #aab8ca;
        font-weight: 720;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56,189,248,.27), rgba(96,165,250,.16));
        color: #ffffff;
        box-shadow: inset 0 0 0 1px rgba(103, 232, 249, .22);
    }

    div[data-testid="stDataFrame"], .stTable, pre {
        border-radius: 18px !important;
        overflow: hidden !important;
    }

    /* Custom components */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: .72rem;
        color: #ffffff;
        font-weight: 850;
        letter-spacing: -.015em;
        font-size: 1.05rem;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(56,189,248,.95), rgba(59,130,246,.68));
        box-shadow: 0 14px 30px rgba(14,165,233,.25);
    }
    .nav-pills { display: flex; gap: .45rem; flex-wrap: wrap; justify-content: flex-end; }
    .nav-pill {
        padding: .42rem .72rem;
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 999px;
        background: rgba(15, 23, 42, .60);
        color: #c8d4e6;
        font-size: .80rem;
    }

    .hero {
        position: relative;
        padding: 1.7rem 1.7rem 1.35rem 1.7rem;
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(14, 165, 233, .18), transparent 42%),
            linear-gradient(180deg, rgba(15, 23, 42, .86), rgba(15, 23, 42, .62));
        box-shadow: 0 26px 70px rgba(0, 0, 0, .30);
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .hero:after {
        content: "";
        position: absolute;
        right: -120px;
        top: -130px;
        width: 360px;
        height: 360px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(103,232,249,.24), transparent 64%);
        pointer-events: none;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .34rem .68rem;
        border-radius: 999px;
        border: 1px solid rgba(103, 232, 249, .26);
        background: rgba(8, 145, 178, .12);
        color: #c6f7ff;
        font-size: .82rem;
        font-weight: 760;
        margin-bottom: .9rem;
    }
    .hero-title {
        color: #ffffff;
        font-size: clamp(2.15rem, 4vw, 4.15rem);
        line-height: 1.02;
        font-weight: 920;
        letter-spacing: -.06em;
        max-width: 920px;
        margin: 0;
    }
    .hero-subtitle {
        max-width: 880px;
        color: #aebdd0;
        font-size: 1.05rem;
        line-height: 1.75;
        margin-top: .95rem;
        margin-bottom: 0;
    }
    .hero-actions {
        display: flex;
        gap: .65rem;
        flex-wrap: wrap;
        margin-top: 1.15rem;
    }
    .tag {
        padding: .42rem .72rem;
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 14px;
        background: rgba(8, 18, 34, .62);
        color: #dbeafe;
        font-size: .84rem;
        font-weight: 650;
    }
    .shell-grid {
        display: grid;
        grid-template-columns: 1.05fr .95fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    @media (max-width: 980px) { .shell-grid { grid-template-columns: 1fr; } }

    .panel {
        padding: 1.05rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: var(--surface);
        box-shadow: 0 20px 52px rgba(0, 0, 0, .22);
    }
    .panel.soft { background: rgba(12, 22, 40, .62); }
    .section-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: .78rem;
    }
    .section-title {
        margin: 0;
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 850;
        letter-spacing: -.01em;
    }
    .section-subtitle {
        color: var(--muted);
        font-size: .86rem;
        margin-top: .18rem;
        line-height: 1.55;
    }
    .status-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: .34rem .66rem;
        font-size: .78rem;
        font-weight: 800;
        white-space: nowrap;
        border: 1px solid rgba(103, 232, 249, .25);
        background: rgba(14, 165, 233, .10);
        color: #c6f7ff;
    }

    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 128px;
        padding: 1.08rem 1.05rem;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, .18);
        background: linear-gradient(180deg, rgba(15, 23, 42, .90), rgba(15, 23, 42, .64));
        box-shadow: 0 18px 42px rgba(0, 0, 0, .20);
    }
    .metric-card:before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 3px;
        background: linear-gradient(90deg, #67e8f9, #38bdf8, #a78bfa);
    }
    .metric-label { color: var(--muted); font-size: .82rem; font-weight: 720; }
    .metric-value { color: #ffffff; font-size: 2.0rem; line-height: 1.08; font-weight: 900; margin-top: .42rem; letter-spacing: -.035em; }
    .metric-note { color: #b8c5d8; font-size: .80rem; margin-top: .35rem; }
    .grade-good { color: #86efac; font-weight: 900; }
    .grade-warn { color: #fcd34d; font-weight: 900; }
    .grade-bad { color: #fda4af; font-weight: 900; }

    .feature-card {
        padding: 1rem;
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, .16);
        background: rgba(15, 23, 42, .62);
        min-height: 128px;
    }
    .feature-icon { font-size: 1.45rem; margin-bottom: .35rem; }
    .feature-title { color: white; font-weight: 840; margin-bottom: .24rem; }
    .feature-desc { color: var(--muted); font-size: .86rem; line-height: 1.58; }
    .divider-space { height: .55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fmt(value, digits: int = 1, suffix: str = "") -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    if digits == 0:
        return f"{int(round(float(value)))}{suffix}"
    return f"{float(value):.{digits}f}{suffix}"


def grade_css(grade: str) -> str:
    if grade in ("A", "B"):
        return "grade-good"
    if grade == "C":
        return "grade-warn"
    return "grade-bad"


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_header(title: str, subtitle: str = "", chip: str | None = None) -> None:
    chip_html = f'<span class="status-chip">{chip}</span>' if chip else ""
    st.markdown(
        f"""
        <div class="section-head">
            <div>
                <h3 class="section-title">{title}</h3>
                <div class="section-subtitle">{subtitle}</div>
            </div>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig: go.Figure, title: str | None = None, height: int = 460) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,18,34,.42)",
        margin=dict(l=42, r=26, t=58 if title else 30, b=42),
        font=dict(family="Inter, Microsoft YaHei, Arial", color="#eaf2ff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.16)", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def parse_rinex_cached(file_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1] or ".obs"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return RinexObs(tmp_path)


def safe_sat_list(rinex: RinexObs) -> list[str]:
    return rinex.get_satellites() if rinex and rinex.get_satellites() else []


def visibility_dataframe(data) -> pd.DataFrame:
    stats = visibility_stats(data)
    rows = []
    for ep, v in stats.items():
        rows.append({"时间": ep, "GPS": v["n_gps"], "BDS": v["n_bds"], "GAL": v["n_gal"], "GLO": v["n_glo"], "总数": v["total"]})
    return pd.DataFrame(rows)


def snr_dataframe(data, obs_types, sat: str | None = None) -> pd.DataFrame:
    snr_keys = [k for k in obs_types if k.startswith("S")]
    sats = [sat] if sat and sat != "全部" else sorted(data.keys())
    rows = []
    for s in sats:
        epochs, d = snr_analysis(data, obs_types, s)
        for key in snr_keys:
            values = d.get(key, [])
            for ep, val in zip(epochs, values):
                if val is not None and np.isfinite(float(val)):
                    rows.append({"时间": ep, "卫星": s, "观测类型": key, "SNR": float(val)})
    return pd.DataFrame(rows)


def multipath_dataframe(data, obs_types, sat: str) -> pd.DataFrame:
    ep, mp1, mp2 = multipath_calc(data, obs_types, sat)
    return pd.DataFrame({"时间": ep, "MP1": mp1, "MP2": mp2})


def slip_dataframe(data, obs_types, sat: str, mw_th: float, gf_th: float):
    slips, mw, gf = cycle_slip_detect(data, obs_types, sat, mw_threshold=mw_th, gf_threshold=gf_th)
    epochs = data.get(sat, {}).get("epoch", [])
    series = pd.DataFrame({"时间": epochs, "MW组合": mw, "GF组合/m": gf})
    slip_rows = []
    for idx, t, mw_jump, gf_jump in slips:
        slip_rows.append({"序号": idx, "时间": t, "MW跳变/周": mw_jump, "GF跳变/m": gf_jump})
    return pd.DataFrame(slip_rows), series


# -----------------------------------------------------------------------------
# Top navigation and hero
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="brand"><div class="brand-icon">🛰️</div><div>GNSS Quality Analyzer</div></div>
        <div class="nav-pills">
            <span class="nav-pill">RINEX OBS</span>
            <span class="nav-pill">SNR</span>
            <span class="nav-pill">Cycle Slip</span>
            <span class="nav-pill">Multipath</span>
            <span class="nav-pill">Coordinate</span>
        </div>
    </div>
    <div class="hero">
        <div class="eyebrow">Real-time Web Dashboard · GNSS Observation Data</div>
        <h1 class="hero-title">GNSS控制网观测数据质量分析</h1>
        <p class="hero-subtitle">
            上传 RINEX 观测文件后，自动完成信噪比统计、MW+GF 周跳探测、MP1/MP2 多路径分析、卫星可见性统计、坐标转换与综合质量评价。
        </p>
        <div class="hero-actions">
            <span class="tag">高清交互图表</span>
            <span class="tag">自动评分</span>
            <span class="tag">报告导出</span>
            <span class="tag">网页端展示</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Upload and parameters
# -----------------------------------------------------------------------------
left, right = st.columns([1.08, .92], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("数据上传", "支持 RINEX 2.x OBS 观测文件。上传后将自动解析文件头和观测数据。", "INPUT")
    uploaded = st.file_uploader("上传 RINEX OBS 文件", type=["obs", "OBS", "23o", "24o", "25o", "26o", "O"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel soft">', unsafe_allow_html=True)
    panel_header("分析参数", "阈值可根据数据采样间隔和观测环境微调。", "CONFIG")
    c1, c2, c3 = st.columns(3)
    with c1:
        mw_threshold = st.number_input("MW阈值/周", value=3.0, min_value=0.1, max_value=20.0, step=0.1)
    with c2:
        gf_threshold = st.number_input("GF阈值/m", value=0.15, min_value=0.01, max_value=5.0, step=0.01)
    with c3:
        max_sats_plot = st.slider("显示卫星数", 4, 20, 10)
    st.markdown('</div>', unsafe_allow_html=True)

rinex = None
if uploaded is not None:
    try:
        with st.spinner("正在解析 RINEX 文件……"):
            rinex = parse_rinex_cached(uploaded.getvalue(), uploaded.name)
    except Exception as exc:
        st.error(f"RINEX 解析失败：{exc}")

if rinex is None:
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("功能展示", "页面将根据上传文件自动生成图表、统计结果与文本报告。", "OVERVIEW")
    f1, f2, f3 = st.columns(3)
    with f1:
        feature_card("📡", "观测数据读取", "读取测站信息、历元数量、观测类型、卫星列表和基础文件头信息。")
    with f2:
        feature_card("📈", "信号质量分析", "统计 SNR 平均值、最大值、最小值，并绘制不同卫星和频段的时间序列。")
    with f3:
        feature_card("🧭", "完整质量评价", "结合 SNR、周跳、多路径、可见卫星数量和观测时长给出评分与结论。")
    g1, g2, g3 = st.columns(3)
    with g1:
        feature_card("⚡", "MW+GF 周跳探测", "基于宽巷组合和无几何组合识别载波相位不连续点。")
    with g2:
        feature_card("🌊", "多路径效应", "计算 MP1/MP2 组合，展示多路径误差的时序变化和 RMS 指标。")
    with g3:
        feature_card("🌐", "坐标转换", "支持 BLH、XYZ、高斯投影之间的常用转换和反算验证。")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# Parsed data
# -----------------------------------------------------------------------------
data = rinex.get_data()
obs_types = rinex.obs_types
sats = safe_sat_list(rinex)
metrics = compute_quality_metrics(data, obs_types)

# KPI strip
st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    metric_card("综合评分", f"{metrics['score']}", f"<span class='{grade_css(metrics['grade'])}'>{metrics['grade']}级 · {metrics['quality']}</span>")
with k2:
    metric_card("平均SNR", fmt(metrics["avg_snr"], 1), "dB-Hz")
with k3:
    metric_card("可疑周跳", str(metrics["cycle_slips"]), "MW+GF")
with k4:
    metric_card("MP1 RMS", fmt(metrics["mp1_rms"], 3), "m")
with k5:
    metric_card("平均卫星数", fmt(metrics["avg_satellites"], 1), "颗")
with k6:
    metric_card("观测时长", fmt(metrics["duration_min"], 1), "分钟")

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
tab_overview, tab_snr, tab_slip, tab_mp, tab_vis, tab_coord, tab_report = st.tabs(
    ["总览", "SNR", "周跳", "多路径", "可见性", "坐标", "报告"]
)

with tab_overview:
    c_left, c_right = st.columns([1.18, .82], gap="large")
    with c_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_header("文件摘要", "从 RINEX 文件头与观测数据中提取的基础信息。", "FILE")
        info_rows = [
            ["文件名", uploaded.name],
            ["RINEX版本", rinex.header.get("version", "?")],
            ["测站名", rinex.header.get("marker_name", "?")],
            ["观测类型", ", ".join(obs_types[:20]) + (" ..." if len(obs_types) > 20 else "")],
            ["卫星数量", len(sats)],
            ["历元数量", len(rinex.get_times())],
            ["卫星列表", ", ".join(sats[:24]) + (" ..." if len(sats) > 24 else "")],
        ]
        st.dataframe(pd.DataFrame(info_rows, columns=["项目", "内容"]), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_header("质量结论", "根据当前文件的主要观测质量指标自动生成。", "RESULT")
        st.markdown(
            f"""
            <div style="font-size:1.05rem;line-height:1.85;color:#dbeafe">
                综合评分 <b style="color:white;font-size:1.25rem">{metrics['score']} / 100</b>，
                等级 <span class="{grade_css(metrics['grade'])}">{metrics['grade']}级 · {metrics['quality']}</span>。<br>
                <span style="color:#aebdd0">{metrics['suggestion']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if metrics["deductions"]:
            st.markdown("**扣分项**")
            for item in metrics["deductions"]:
                st.markdown(f"- {item}")
        else:
            st.success("未触发明显扣分项。")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_header("质量雷达", "五个维度的归一化展示，便于快速判断数据短板。", "RADAR")
        snr_score = 100 if not np.isfinite(metrics["avg_snr"]) else np.clip((metrics["avg_snr"] - 25) / 20 * 100, 0, 100)
        slip_score = np.clip(100 - metrics["cycle_slips"] * 5, 0, 100)
        mp_score = 60 if not np.isfinite(metrics["mp1_rms"]) else np.clip(100 - metrics["mp1_rms"] * 120, 0, 100)
        sat_score = 50 if not np.isfinite(metrics["avg_satellites"]) else np.clip(metrics["avg_satellites"] / 15 * 100, 0, 100)
        dur_score = 50 if not np.isfinite(metrics["duration_min"]) else np.clip(metrics["duration_min"] / 120 * 100, 0, 100)
        cats = ["SNR", "连续性", "多路径", "卫星数", "时长"]
        vals = [snr_score, slip_score, mp_score, sat_score, dur_score]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", name="质量指数",
            line=dict(color="#67e8f9", width=3), fillcolor="rgba(56,189,248,.22)"
        ))
        fig.update_layout(
            template="plotly_dark", height=430, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#eaf2ff"),
            margin=dict(l=22, r=22, t=10, b=20),
            polar=dict(
                bgcolor="rgba(8,18,34,.25)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,.20)"),
                angularaxis=dict(gridcolor="rgba(148,163,184,.20)"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel soft">', unsafe_allow_html=True)
        panel_header("核心指标", "报告中使用的主要数值。", "METRIC")
        core = pd.DataFrame({
            "指标": ["观测时长/min", "平均SNR/dB-Hz", "周跳数", "MP1 RMS/m", "MP2 RMS/m", "平均卫星数"],
            "数值": [
                fmt(metrics["duration_min"], 1), fmt(metrics["avg_snr"], 1), metrics["cycle_slips"],
                fmt(metrics["mp1_rms"], 3), fmt(metrics["mp2_rms"], 3), fmt(metrics["avg_satellites"], 1),
            ],
        })
        st.dataframe(core, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_snr:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("SNR 信噪比分析", "展示不同卫星和频段的信号强度时间序列，并计算平均、最大、最小 SNR。", "SIGNAL")
    sat_choice = st.selectbox("选择卫星", ["全部"] + sats, index=0, key="snr_sat")
    df = snr_dataframe(data, obs_types, sat_choice)
    if df.empty:
        st.warning("该文件没有可用 SNR 观测值。")
    else:
        stat = snr_statistics(data, obs_types) if sat_choice == "全部" else snr_statistics(data, obs_types, sat_choice)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("平均SNR", fmt(stat["avg"], 1, " dB-Hz"))
        c2.metric("最大SNR", fmt(stat["max"], 1, " dB-Hz"))
        c3.metric("最小SNR", fmt(stat["min"], 1, " dB-Hz"))
        c4.metric("有效样本", int(stat["count"]))

        plot_df = df.copy()
        group_cols: list[str]
        if sat_choice == "全部":
            keep = sorted(plot_df["卫星"].unique())[:max_sats_plot]
            plot_df = plot_df[plot_df["卫星"].isin(keep)]
            group_cols = ["卫星", "观测类型"]
        else:
            group_cols = ["观测类型"]

        fig = go.Figure()
        for keys, sub in plot_df.groupby(group_cols):
            name = " ".join(keys) if isinstance(keys, tuple) else str(keys)
            fig.add_trace(go.Scatter(x=sub["时间"], y=sub["SNR"], mode="lines", name=name, line=dict(width=2)))
        plot_layout(fig, "SNR Time Series", 500)
        fig.update_yaxes(title="SNR / dB-Hz")
        st.plotly_chart(fig, use_container_width=True)

        stat_df = df.groupby(["卫星", "观测类型"], as_index=False).agg(
            平均SNR=("SNR", "mean"), 最大SNR=("SNR", "max"), 最小SNR=("SNR", "min"), 样本数=("SNR", "count")
        )
        st.dataframe(stat_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_slip:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("MW+GF 周跳探测", "通过宽巷 MW 组合和无几何 GF 组合识别载波相位不连续。", "INTEGRITY")
    if not sats:
        st.warning("没有卫星数据。")
    else:
        sat_slip = st.selectbox("选择卫星", sats, index=0, key="slip_sat")
        slip_df, series_df = slip_dataframe(data, obs_types, sat_slip, mw_threshold, gf_threshold)
        c1, c2, c3 = st.columns(3)
        c1.metric("可疑周跳", len(slip_df))
        c2.metric("MW阈值", f"{mw_threshold:.1f} 周")
        c3.metric("GF阈值", f"{gf_threshold:.2f} m")

        fig = go.Figure()
        if not series_df.empty:
            fig.add_trace(go.Scatter(x=series_df["时间"], y=series_df["MW组合"], mode="lines", name="MW组合/周", yaxis="y1", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=series_df["时间"], y=series_df["GF组合/m"], mode="lines", name="GF组合/m", yaxis="y2", line=dict(width=2)))
            for _, row in slip_df.iterrows():
                fig.add_vline(x=row["时间"], line_width=1, line_dash="dash", line_color="#fb7185")
        fig.update_layout(yaxis=dict(title="MW / 周"), yaxis2=dict(title="GF / m", overlaying="y", side="right"))
        plot_layout(fig, f"{sat_slip} Cycle Slip Detection", 520)
        st.plotly_chart(fig, use_container_width=True)
        if not slip_df.empty:
            st.dataframe(slip_df, use_container_width=True, hide_index=True)
        else:
            st.success("该卫星未检测到明显周跳。")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_mp:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("多路径效应分析", "计算 MP1/MP2 组合并展示多路径误差随时间变化。", "MULTIPATH")
    if not sats:
        st.warning("没有卫星数据。")
    else:
        sat_mp = st.selectbox("选择卫星", sats, index=0, key="mp_sat")
        mp_df = multipath_dataframe(data, obs_types, sat_mp)
        if mp_df.empty or mp_df[["MP1", "MP2"]].dropna(how="all").empty:
            st.warning("该卫星缺少双频伪距/载波数据，无法计算 MP1/MP2。")
        else:
            mp1_rms = np.sqrt(np.nanmean(mp_df["MP1"] ** 2))
            mp2_rms = np.sqrt(np.nanmean(mp_df["MP2"] ** 2))
            c1, c2, c3 = st.columns(3)
            c1.metric("MP1 RMS", fmt(mp1_rms, 3, " m"))
            c2.metric("MP2 RMS", fmt(mp2_rms, 3, " m"))
            c3.metric("有效历元", int(mp_df[["MP1", "MP2"]].dropna(how="all").shape[0]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mp_df["时间"], y=mp_df["MP1"], mode="lines", name="MP1", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=mp_df["时间"], y=mp_df["MP2"], mode="lines", name="MP2", line=dict(width=2)))
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,.45)")
            plot_layout(fig, f"{sat_mp} Multipath Combination", 520)
            fig.update_yaxes(title="多路径组合 / m")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("参考：MP RMS < 0.3 m 较好，0.3~0.5 m 一般，> 0.5 m 多路径影响偏大。")
            st.dataframe(mp_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_vis:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("卫星可见性", "统计每个历元可见卫星总数及不同系统卫星数量。", "VISIBILITY")
    vis_df = visibility_dataframe(data)
    if vis_df.empty:
        st.warning("没有可见卫星统计。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("平均总数", fmt(vis_df["总数"].mean(), 1, " 颗"))
        c2.metric("最多", f"{int(vis_df['总数'].max())} 颗")
        c3.metric("最少", f"{int(vis_df['总数'].min())} 颗")
        c4.metric("历元数", len(vis_df))
        fig = go.Figure()
        for col in ["总数", "GPS", "BDS", "GAL", "GLO"]:
            if col == "总数" or vis_df[col].sum() > 0:
                width = 3 if col == "总数" else 2
                fig.add_trace(go.Scatter(x=vis_df["时间"], y=vis_df[col], mode="lines", name=col, line=dict(width=width)))
        plot_layout(fig, "Satellite Visibility", 520)
        fig.update_yaxes(title="可见卫星数 / 颗", rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vis_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_coord:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("坐标转换", "支持 BLH、XYZ、高斯投影之间的常用转换。", "COORDINATE")
    mode = st.radio("转换模式", ["BLH → XYZ / 高斯", "XYZ → BLH / 高斯", "高斯反算 → BLH / XYZ"], horizontal=True)
    ell_name = st.radio("椭球", ["WGS84", "CGCS2000"], horizontal=True)
    ell = WGS84 if ell_name == "WGS84" else CGCS2000

    if mode.startswith("BLH"):
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("B 纬度 / °", value=39.9042, format="%.8f")
        lon = c2.number_input("L 经度 / °", value=116.4074, format="%.8f")
        h = c3.number_input("H 大地高 / m", value=50.0, format="%.4f")
        if st.button("开始转换", key="blh_convert"):
            st.code(convert_summary(lat, lon, h, ell_name), language="text")

    elif mode.startswith("XYZ"):
        c1, c2, c3 = st.columns(3)
        X = c1.number_input("X / m", value=-2175712.0, format="%.4f")
        Y = c2.number_input("Y / m", value=4388428.0, format="%.4f")
        Z = c3.number_input("Z / m", value=4076540.0, format="%.4f")
        if st.button("开始转换", key="xyz_convert"):
            lat, lon, h = xyz_to_blh(X, Y, Z, ell)
            cm3 = round(lon / 3) * 3
            x3, y3, _ = gauss_projection(lat, lon, cm3, ell)
            ld, lm, ls = deg_to_dms(abs(lat))
            lod, lom, los = deg_to_dms(abs(lon))
            lines = [
                "=" * 56,
                f"XYZ → BLH / 高斯投影 ({ell_name})",
                "=" * 56,
                f"输入: X={X:.4f} m, Y={Y:.4f} m, Z={Z:.4f} m",
                f"B={ld}°{lm}'{ls:.3f}\" {'N' if lat >= 0 else 'S'} = {lat:.8f}°",
                f"L={lod}°{lom}'{los:.3f}\" {'E' if lon >= 0 else 'W'} = {lon:.8f}°",
                f"H={h:.4f} m", "", f"3°带中央子午线: {cm3}°", f"x(北)={x3:.4f} m", f"y(东)={y3:.4f} m", "=" * 56,
            ]
            st.code("\n".join(lines), language="text")

    else:
        c1, c2, c3 = st.columns(3)
        gx = c1.number_input("x 北坐标 / m", value=4419100.0, format="%.4f")
        gy = c2.number_input("y 东坐标 / m", value=536400.0, format="%.4f")
        cm = c3.number_input("中央子午线 / °", value=117.0, format="%.4f")
        if st.button("开始转换", key="gauss_convert"):
            lat, lon, _ = gauss_inverse(gx, gy, cm, ell)
            X, Y, Z = blh_to_xyz(lat, lon, 0, ell)
            ld, lm, ls = deg_to_dms(abs(lat))
            lod, lom, los = deg_to_dms(abs(lon))
            lines = [
                "=" * 56,
                f"高斯反算 → BLH / XYZ ({ell_name})",
                "=" * 56,
                f"输入: x={gx:.4f} m, y={gy:.4f} m, CM={cm:.4f}°",
                f"B={ld}°{lm}'{ls:.3f}\" {'N' if lat >= 0 else 'S'} = {lat:.8f}°",
                f"L={lod}°{lom}'{los:.3f}\" {'E' if lon >= 0 else 'W'} = {lon:.8f}°",
                f"XYZ: X={X:.4f} m, Y={Y:.4f} m, Z={Z:.4f} m", "=" * 56,
            ]
            st.code("\n".join(lines), language="text")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_report:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_header("报告导出", "导出综合质量评价和完整分析报告。", "EXPORT")
    report_text = quality_evaluation_report(data, obs_types)
    full_report = quality_summary(data, obs_types)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("下载综合质量评价 TXT", data=report_text.encode("utf-8"), file_name="GNSS数据质量评价报告.txt", mime="text/plain")
    with d2:
        st.download_button("下载完整分析报告 TXT", data=full_report.encode("utf-8"), file_name="GNSS观测数据质量分析报告.txt", mime="text/plain")
    st.markdown("#### 综合质量评价")
    st.code(report_text, language="text")
    st.markdown("#### 完整分析报告")
    st.code(full_report, language="text")
    st.markdown('</div>', unsafe_allow_html=True)
