"""
GNSS 控制网观测数据质量检查程序 - Streamlit 网页前端
运行:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rinex_reader import RinexObs
from rinex_nav_reader import RinexNav
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
from satellite_orbit import (
    calc_sat_position,
    calc_sat_azel,
    compute_dop_from_azel,
)
from spp_positioning import spp_solve

st.set_page_config(
    page_title="GNSS控制网观测数据质量检查程序",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS: 参考 RTK/GNSS 分析类网页的卡片式深色前端
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg0: #07111f;
        --bg1: #0f172a;
        --card: rgba(15, 23, 42, 0.86);
        --card2: rgba(30, 41, 59, 0.86);
        --line: rgba(148, 163, 184, 0.22);
        --text: #e5e7eb;
        --muted: #94a3b8;
        --cyan: #38bdf8;
        --blue: #60a5fa;
        --green: #22c55e;
        --amber: #f59e0b;
        --red: #ef4444;
        --purple: #a78bfa;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(56, 189, 248, 0.20), transparent 26%),
            radial-gradient(circle at 88% 8%, rgba(167, 139, 250, 0.18), transparent 28%),
            linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.98));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }
    .hero {
        padding: 1.45rem 1.65rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.72));
        box-shadow: 0 22px 60px rgba(0,0,0,0.24);
        margin-bottom: 1.05rem;
    }
    .hero-title {
        font-size: 2.1rem;
        line-height: 1.15;
        font-weight: 850;
        letter-spacing: -0.02em;
        color: white;
        margin: 0 0 0.35rem 0;
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin: 0;
    }
    .pill-row { margin-top: 0.9rem; }
    .pill {
        display: inline-block;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        border: 1px solid rgba(56, 189, 248, 0.22);
        background: rgba(56, 189, 248, 0.10);
        color: #dff6ff;
        font-size: 0.82rem;
    }
    .metric-card {
        padding: 1.0rem 1.05rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(15, 23, 42, 0.70));
        box-shadow: 0 18px 45px rgba(0,0,0,0.18);
        min-height: 118px;
        position: relative;
        overflow: hidden;
    }
    .metric-card:before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--cyan), var(--purple));
    }
    .metric-title {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: white;
        font-size: 1.72rem;
        font-weight: 830;
        line-height: 1.12;
        margin-bottom: 0.2rem;
    }
    .metric-foot {
        color: #cbd5e1;
        font-size: 0.78rem;
    }
    .glass-panel {
        padding: 1rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.78);
        box-shadow: 0 18px 45px rgba(0,0,0,0.18);
    }
    .section-title {
        color: white;
        font-size: 1.15rem;
        font-weight: 780;
        margin: 0 0 0.65rem 0;
    }
    .small-muted { color: var(--muted); font-size: 0.88rem; }
    .good { color: #86efac; font-weight: 800; }
    .warn { color: #fcd34d; font-weight: 800; }
    .bad { color: #fca5a5; font-weight: 800; }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid var(--line);
        padding: 0.85rem 0.95rem;
        border-radius: 18px;
    }
    div[data-testid="stMetric"] label { color: var(--muted) !important; }
    div[data-testid="stMetricValue"] { color: white !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.45rem; }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 999px;
        padding: 0 1.0rem;
        border: 1px solid var(--line);
        background: rgba(15, 23, 42, 0.72);
        color: var(--muted);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(96,165,250,0.16));
        color: white;
        border-color: rgba(56,189,248,0.45);
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(96,165,250,0.16));
        color: white;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(56, 189, 248, 0.75);
        color: white;
    }
    .stDataFrame, .stTable {
        border-radius: 18px;
        overflow: hidden;
    }
    pre, code {
        border-radius: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def fmt(value, digits=1, suffix=""):
    try:
        if value is None or not np.isfinite(float(value)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    if digits == 0:
        return f"{int(round(float(value)))}{suffix}"
    return f"{float(value):.{digits}f}{suffix}"


def grade_class(grade: str) -> str:
    if grade in ("A", "B"):
        return "good"
    if grade == "C":
        return "warn"
    return "bad"


def metric_card(title: str, value: str, foot: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_start(title: str, subtitle: str | None = None):
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)


def panel_end():
    st.markdown('</div>', unsafe_allow_html=True)


def make_plot_layout(fig: go.Figure, title: str | None = None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        margin=dict(l=35, r=22, t=52 if title else 28, b=35),
        font=dict(family="Microsoft YaHei, Arial", color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


def save_uploaded_rinex(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".obs"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


@st.cache_data(show_spinner=False)
def parse_rinex_cached(file_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1] or ".obs"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    rinex = RinexObs(path)
    return rinex


def get_loaded_rinex():
    uploaded = st.session_state.get("uploaded_rinex")
    if uploaded is None:
        return None
    return parse_rinex_cached(uploaded.getvalue(), uploaded.name)


def safe_sat_list(rinex: RinexObs):
    sats = rinex.get_satellites()
    return sats if sats else []


def visibility_dataframe(data) -> pd.DataFrame:
    stats = visibility_stats(data)
    rows = []
    for ep, v in stats.items():
        rows.append({
            "时间": ep,
            "GPS": v["n_gps"],
            "BDS": v["n_bds"],
            "GAL": v["n_gal"],
            "GLO": v["n_glo"],
            "总数": v["total"],
        })
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
    rows = []
    for t, a, b in zip(ep, mp1, mp2):
        rows.append({"时间": t, "MP1": a, "MP2": b})
    return pd.DataFrame(rows)


def slip_dataframe(data, obs_types, sat: str, mw_th: float, gf_th: float):
    slips, mw, gf = cycle_slip_detect(data, obs_types, sat, mw_threshold=mw_th, gf_threshold=gf_th)
    epochs = data.get(sat, {}).get("epoch", [])
    series = pd.DataFrame({"时间": epochs, "MW组合": mw, "GF组合(m)": gf})
    slip_rows = []
    for idx, t, mw_jump, gf_jump in slips:
        slip_rows.append({
            "序号": idx,
            "时间": t,
            "MW跳变(周)": mw_jump,
            "GF跳变(m)": gf_jump,
        })
    return pd.DataFrame(slip_rows), series


# -----------------------------------------------------------------------------
# NAV / 定位辅助函数
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_nav_cached(file_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1] or ".nav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    return RinexNav(path)


def get_loaded_nav():
    uploaded = st.session_state.get("uploaded_nav")
    if uploaded is None:
        return None
    return parse_nav_cached(uploaded.getvalue(), uploaded.name)


def get_nav_eph_dict(nav):
    if nav is None:
        return {}
    eph_dict = {}
    for prn in nav.get_all_prns():
        ephs = nav._eph.get(prn, [])
        if ephs:
            eph_dict[prn] = ephs
    return eph_dict


def pos_dataframe(pos_result):
    if pos_result is None or not pos_result.get("epochs"):
        return pd.DataFrame()
    rows = []
    for i in range(len(pos_result["epochs"])):
        rows.append({
            "时间": pos_result["epochs"][i],
            "X/m": fmt(pos_result["X"][i], 3),
            "Y/m": fmt(pos_result["Y"][i], 3),
            "Z/m": fmt(pos_result["Z"][i], 3),
            "B/°": fmt(pos_result["lat"][i], 8),
            "L/°": fmt(pos_result["lon"][i], 8),
            "H/m": fmt(pos_result["h"][i], 3),
            "GDOP": fmt(pos_result["GDOP"][i], 1),
            "PDOP": fmt(pos_result["PDOP"][i], 1),
            "卫星数": int(pos_result["nsat"][i]) if i < len(pos_result["nsat"]) else 0,
            "残差RMS/m": fmt(pos_result["rms_residual"][i], 3),
        })
    return pd.DataFrame(rows)


def make_skyplot(sat_azel_list, title_str="卫星天空图"):
    """根据 [(el_deg, az_deg, prn), ...] 生成极坐标天空图。"""
    fig = go.Figure()
    if not sat_azel_list:
        return fig
    for el, az, prn in sat_azel_list:
        fig.add_trace(go.Scatterpolar(
            r=[90 - el],
            theta=[az],
            mode="markers+text",
            text=[prn],
            textposition="top center",
            marker=dict(size=10, symbol="circle"),
            name=prn,
            showlegend=False,
        ))
    fig.update_layout(
        polar=dict(
            angular=dict(rotation=90, direction="clockwise",
                         tickmode="array",
                         tickvals=[0, 90, 180, 270],
                         ticktext=["N", "E", "S", "W"]),
            radial=dict(range=[0, 90],
                        tickvals=[0, 15, 30, 45, 60, 75, 90],
                        ticktext=["0°", "15°", "30°", "45°", "60°", "75°", "90°"]),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title=title_str,
        font=dict(color="#e5e7eb"),
    )
    return fig


# -----------------------------------------------------------------------------
# 页面头部
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛰️ GNSS控制网观测数据质量检查程序</div>
        <p class="hero-subtitle">面向本科课程设计：RINEX读取、SNR分析、MW+GF周跳探测、MP1/MP2多路径、卫星可见性、坐标转换与综合质量评价。</p>
        <div class="pill-row">
            <span class="pill">RINEX OBS</span>
            <span class="pill">SNR Quality</span>
            <span class="pill">Cycle Slip</span>
            <span class="pill">Multipath</span>
            <span class="pill">Visibility</span>
            <span class="pill">Coordinate Transform</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 侧边栏
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 数据输入")
    st.caption("上传 RINEX OBS 观测文件后即可进行质量分析。")
    uploaded = st.file_uploader(
        "选择 RINEX OBS 文件",
        type=["obs", "OBS", "23o", "24o", "25o", "26o", "O"],
        key="uploaded_rinex",
    )

    st.markdown("---")
    st.markdown("### 导航电文 (定位)")
    st.caption("上传 RINEX NAV 导航文件后可进行单点定位解算。")
    uploaded_nav = st.file_uploader(
        "选择 RINEX NAV 文件",
        type=["nav", "NAV", "23n", "24n", "25n", "26n", "N"],
        key="uploaded_nav",
    )

    st.markdown("---")
    st.markdown("### 分析参数")
    mw_threshold = st.number_input("MW周跳阈值 / 周", value=3.0, min_value=0.1, max_value=20.0, step=0.1)
    gf_threshold = st.number_input("GF周跳阈值 / m", value=0.15, min_value=0.01, max_value=5.0, step=0.01)
    max_sats_plot = st.slider("多卫星图最多显示卫星数", 4, 20, 10)
    elev_mask = st.slider("定位截止高度角 / °", 0, 30, 10)
    pos_system = st.selectbox("定位卫星系统", ["自动 (GPS优先)", "GPS", "BDS"], index=0)

    st.markdown("---")
    st.markdown("### 项目信息")
    st.caption("网页端部署入口：app.py")
    st.caption("核心功能：RINEX读取、SNR、周跳、多路径、可见性、坐标转换、质量评分。")

rinex = None
if uploaded is not None:
    try:
        with st.spinner("正在解析 RINEX 文件……"):
            rinex = get_loaded_rinex()
    except Exception as exc:
        st.error(f"RINEX 解析失败：{exc}")
        rinex = None

# NAV 加载
nav = None
if uploaded_nav is not None:
    try:
        with st.spinner("正在解析 NAV 导航文件……"):
            nav = get_loaded_nav()
    except Exception as exc:
        st.error(f"NAV 解析失败：{exc}")
        nav = None

if rinex is None:
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        panel_start("使用说明", "先在左侧上传 RINEX OBS 文件。")
        st.markdown(
            """
            这是网页发布版主界面。上传 RINEX OBS 文件后会自动生成：

            - 文件摘要与综合质量评分
            - SNR 时间序列与统计表
            - MW+GF 周跳探测曲线
            - MP1/MP2 多路径曲线
            - 卫星可见性变化图
            - **单点定位解算（需同时上传 NAV 文件）**
            - **卫星轨道 / 天空图（需同时上传 NAV 文件）**
            - BLH / XYZ / 高斯投影坐标转换
            """
        )
        panel_end()
    with c2:
        metric_card("课程设计题目", "卫星导航定位", "控制网观测与定位解算")
    with c3:
        metric_card("推荐提交形式", "网页程序", "Streamlit + Python 算法模块")
    st.stop()

# -----------------------------------------------------------------------------
# 加载数据与指标
# -----------------------------------------------------------------------------
data = rinex.get_data()
obs_types = rinex.obs_types
sats = safe_sat_list(rinex)
metrics = compute_quality_metrics(data, obs_types, pos_result)

# SPP 定位解算
pos_result = None
if nav is not None and data:
    try:
        with st.spinner("正在进行单点定位解算……"):
            use_iono = False
            iono_params = nav.iono_params if nav.iono_params else None
            if iono_params and iono_params.get("alpha"):
                use_iono = True
            sys_map = {"自动 (GPS优先)": "G", "GPS": "G", "BDS": "C"}
            ps = sys_map.get(pos_system, "G")
            approx_xyz = None
            if rinex.header.get("approx_xyz"):
                approx_xyz = tuple(rinex.header["approx_xyz"])
            pos_result = spp_solve(data, obs_types, get_nav_eph_dict(nav),
                                  approx_xyz, float(elev_mask), use_iono,
                                  iono_params, system=ps)
    except Exception as exc:
        st.warning(f"定位解算失败：{exc}")
        pos_result = None

# KPI 卡片
n_kpi = 7 if pos_result is not None and pos_result.get("n_epochs", 0) > 0 else 6
if n_kpi >= 7:
    cols = st.columns(7)
else:
    cols = st.columns(6)
with cols[0]:
    metric_card("综合评分", f"{metrics['score']}", f"<span class='{grade_class(metrics['grade'])}'>{metrics['grade']}级 · {metrics['quality']}</span>")
with cols[1]:
    metric_card("平均SNR", fmt(metrics["avg_snr"], 1), "dB-Hz")
with cols[2]:
    metric_card("可疑周跳", f"{metrics['cycle_slips']}", "MW+GF")
with cols[3]:
    metric_card("MP1 RMS", fmt(metrics["mp1_rms"], 3), "m")
with cols[4]:
    metric_card("平均卫星数", fmt(metrics["avg_satellites"], 1), "颗")
with cols[5]:
    metric_card("观测时长", fmt(metrics["duration_min"], 1), "分钟")
if n_kpi >= 7:
    gdop_mean = np.nanmean([g for g in pos_result["GDOP"] if np.isfinite(g)]) if pos_result["GDOP"] else float("nan")
    with cols[6]:
        metric_card("平均GDOP", fmt(gdop_mean, 1), "定位精度")

# -----------------------------------------------------------------------------
# 主标签页
# -----------------------------------------------------------------------------
tab_overview, tab_snr, tab_slip, tab_mp, tab_vis, tab_pos, tab_orbit, tab_coord, tab_report = st.tabs(
    ["总览", "SNR分析", "周跳探测", "多路径", "卫星可见性", "定位解算", "卫星轨道", "坐标转换", "报告导出"]
)

with tab_overview:
    left, right = st.columns([1.12, 1])

    with left:
        panel_start("文件信息")
        info_rows = [
            ["文件名", uploaded.name],
            ["RINEX版本", rinex.header.get("version", "?")],
            ["测站名", rinex.header.get("marker_name", "?")],
            ["观测类型", ", ".join(obs_types[:18]) + (" ..." if len(obs_types) > 18 else "")],
            ["卫星数量", len(sats)],
            ["历元数量", len(rinex.get_times())],
            ["卫星列表", ", ".join(sats[:18]) + (" ..." if len(sats) > 18 else "")],
        ]
        st.dataframe(pd.DataFrame(info_rows, columns=["项目", "内容"]), use_container_width=True, hide_index=True)
        panel_end()

        panel_start("综合结论")
        st.markdown(
            f"""
            <div style="font-size:1.05rem;line-height:1.85">
            当前数据综合评分为 <b>{metrics['score']} / 100</b>，质量等级为
            <span class="{grade_class(metrics['grade'])}">{metrics['grade']}级 · {metrics['quality']}</span>。<br>
            <span style="color:#cbd5e1">{metrics['suggestion']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if metrics["deductions"]:
            st.markdown("**扣分项：**")
            for item in metrics["deductions"]:
                st.markdown(f"- {item}")
        else:
            st.success("未触发明显扣分项。")
        panel_end()

    with right:
        panel_start("质量雷达图", "用于直观展示信号、连续性、多路径、卫星数量、定位精度等维度。")
        snr_score = 100 if not np.isfinite(metrics["avg_snr"]) else np.clip((metrics["avg_snr"] - 25) / 20 * 100, 0, 100)
        slip_score = np.clip(100 - metrics["cycle_slips"] * 5, 0, 100)
        mp_score = 60 if not np.isfinite(metrics["mp1_rms"]) else np.clip(100 - metrics["mp1_rms"] * 120, 0, 100)
        sat_score = 50 if not np.isfinite(metrics["avg_satellites"]) else np.clip(metrics["avg_satellites"] / 15 * 100, 0, 100)
        dur_score = 50 if not np.isfinite(metrics["duration_min"]) else np.clip(metrics["duration_min"] / 120 * 100, 0, 100)
        categories = ["SNR", "连续性", "多路径", "卫星数量", "观测时长"]
        values = [snr_score, slip_score, mp_score, sat_score, dur_score]

        # 若有定位结果，加入 DOP 维度
        has_pos = pos_result is not None and pos_result.get("n_epochs", 0) > 0
        if has_pos:
            avg_gdop_val = np.nanmean([g for g in pos_result["GDOP"] if np.isfinite(g)]) if pos_result["GDOP"] else float("nan")
            if np.isfinite(avg_gdop_val):
                dop_score = np.clip(100 - (avg_gdop_val - 1) * 15, 0, 100)
            else:
                dop_score = 50
            categories = categories + ["定位精度"]
            values = values + [dop_score]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="质量得分",
            line=dict(color="#38bdf8"),
            fillcolor="rgba(56,189,248,0.25)",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(15,23,42,0.20)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.22)"),
                angularaxis=dict(gridcolor="rgba(148,163,184,0.22)"),
            ),
            margin=dict(l=25, r=25, t=20, b=20),
            showlegend=False,
            font=dict(color="#e5e7eb"),
        )
        st.plotly_chart(fig, use_container_width=True)
        panel_end()

        panel_start("核心指标表")
        core_data = {
            "指标": ["观测时长/min", "平均SNR/dB-Hz", "周跳数", "MP1 RMS/m", "MP2 RMS/m", "平均卫星数"],
            "数值": [
                fmt(metrics["duration_min"], 1),
                fmt(metrics["avg_snr"], 1),
                metrics["cycle_slips"],
                fmt(metrics["mp1_rms"], 3),
                fmt(metrics["mp2_rms"], 3),
                fmt(metrics["avg_satellites"], 1),
            ],
        }
        if has_pos and np.isfinite(metrics.get("avg_gdop", float("nan"))):
            core_data["指标"].append("平均GDOP")
            core_data["数值"].append(fmt(metrics["avg_gdop"], 1))
        core = pd.DataFrame(core_data)
        st.dataframe(core, use_container_width=True, hide_index=True)
        panel_end()

        if pos_result is not None and pos_result.get("n_epochs", 0) > 0:
            panel_start("定位解算结果", "伪距单点定位 (SPP) 概要")
            cl, cr = st.columns(2)
            with cl:
                stat_df = pd.DataFrame({
                    "项目": ["平均B", "平均L", "平均H", "解算历元", "平均卫星数", "定位模式"],
                    "内容": [
                        fmt(pos_result.get("mean_lat"), 8) + "°",
                        fmt(pos_result.get("mean_lon"), 8) + "°",
                        fmt(pos_result.get("mean_h"), 3) + " m",
                        pos_result.get("n_epochs", 0),
                        fmt(pos_result.get("mean_nsat", 0), 1) + " 颗",
                        "伪距SPP",
                    ],
                })
                st.dataframe(stat_df, use_container_width=True, hide_index=True)
            with cr:
                gdop_mean = np.nanmean([g for g in pos_result["GDOP"] if np.isfinite(g)]) if pos_result["GDOP"] else float("nan")
                pdop_mean = np.nanmean([p for p in pos_result["PDOP"] if np.isfinite(p)]) if pos_result["PDOP"] else float("nan")
                vdop_mean = np.nanmean([v for v in pos_result["VDOP"] if np.isfinite(v)]) if pos_result["VDOP"] else float("nan")
                dop_df = pd.DataFrame({
                    "指标": ["GDOP", "PDOP", "HDOP", "VDOP"],
                    "均值": [fmt(gdop_mean, 2), fmt(pdop_mean, 2),
                             fmt(np.nanmean([h for h in pos_result["HDOP"] if np.isfinite(h)]) if pos_result["HDOP"] else float("nan"), 2),
                             fmt(vdop_mean, 2)],
                })
                st.dataframe(dop_df, use_container_width=True, hide_index=True)
            panel_end()

with tab_snr:
    sat_choice = st.selectbox("选择卫星", ["全部"] + sats, index=0, key="snr_sat")
    df = snr_dataframe(data, obs_types, sat_choice)
    if df.empty:
        st.warning("该文件没有可用 SNR 观测值。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        if sat_choice == "全部":
            stat = snr_statistics(data, obs_types)
        else:
            stat = snr_statistics(data, obs_types, sat_choice)
        c1.metric("平均SNR", fmt(stat["avg"], 1, " dB-Hz"))
        c2.metric("最大SNR", fmt(stat["max"], 1, " dB-Hz"))
        c3.metric("最小SNR", fmt(stat["min"], 1, " dB-Hz"))
        c4.metric("有效样本", stat["count"])

        plot_df = df.copy()
        if sat_choice == "全部":
            keep = sorted(plot_df["卫星"].unique())[:max_sats_plot]
            plot_df = plot_df[plot_df["卫星"].isin(keep)]
            group_cols = ["卫星", "观测类型"]
        else:
            group_cols = ["观测类型"]

        fig = go.Figure()
        for keys, sub in plot_df.groupby(group_cols):
            name = " ".join(keys) if isinstance(keys, tuple) else str(keys)
            fig.add_trace(go.Scatter(x=sub["时间"], y=sub["SNR"], mode="lines+markers", name=name, marker=dict(size=4)))
        make_plot_layout(fig, "SNR 时间序列")
        fig.update_yaxes(title="SNR / dB-Hz")
        st.plotly_chart(fig, use_container_width=True)

        stat_df = df.groupby(["卫星", "观测类型"], as_index=False).agg(
            平均SNR=("SNR", "mean"), 最大SNR=("SNR", "max"), 最小SNR=("SNR", "min"), 样本数=("SNR", "count")
        )
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

with tab_slip:
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
            fig.add_trace(go.Scatter(x=series_df["时间"], y=series_df["MW组合"], mode="lines", name="MW组合/周", yaxis="y1"))
            fig.add_trace(go.Scatter(x=series_df["时间"], y=series_df["GF组合(m)"], mode="lines", name="GF组合/m", yaxis="y2"))
            for _, row in slip_df.iterrows():
                fig.add_vline(x=row["时间"], line_width=1, line_dash="dash", line_color="#ef4444")
        fig.update_layout(
            yaxis=dict(title="MW / 周"),
            yaxis2=dict(title="GF / m", overlaying="y", side="right"),
        )
        make_plot_layout(fig, f"{sat_slip} MW+GF 周跳探测")
        st.plotly_chart(fig, use_container_width=True)
        if not slip_df.empty:
            st.dataframe(slip_df, use_container_width=True, hide_index=True)
        else:
            st.success("该卫星未检测到明显周跳。")

with tab_mp:
    if not sats:
        st.warning("没有卫星数据。")
    else:
        sat_mp = st.selectbox("选择卫星", sats, index=0, key="mp_sat")
        mp_df = multipath_dataframe(data, obs_types, sat_mp)
        if mp_df.empty or (mp_df[["MP1", "MP2"]].dropna(how="all").empty):
            st.warning("该卫星缺少双频伪距/载波数据，无法计算 MP1/MP2。")
        else:
            mp1_rms = np.sqrt(np.nanmean(mp_df["MP1"] ** 2))
            mp2_rms = np.sqrt(np.nanmean(mp_df["MP2"] ** 2))
            c1, c2, c3 = st.columns(3)
            c1.metric("MP1 RMS", fmt(mp1_rms, 3, " m"))
            c2.metric("MP2 RMS", fmt(mp2_rms, 3, " m"))
            c3.metric("有效历元", int(mp_df[["MP1", "MP2"]].dropna(how="all").shape[0]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mp_df["时间"], y=mp_df["MP1"], mode="lines", name="MP1"))
            fig.add_trace(go.Scatter(x=mp_df["时间"], y=mp_df["MP2"], mode="lines", name="MP2"))
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.45)")
            make_plot_layout(fig, f"{sat_mp} 多路径效应")
            fig.update_yaxes(title="多路径组合 / m")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("参考：MP RMS < 0.3 m 较好，0.3~0.5 m 一般，> 0.5 m 多路径影响偏大。")
            st.dataframe(mp_df, use_container_width=True, hide_index=True)

with tab_vis:
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
                fig.add_trace(go.Scatter(x=vis_df["时间"], y=vis_df[col], mode="lines", name=col))
        make_plot_layout(fig, "卫星可见性随时间变化")
        fig.update_yaxes(title="可见卫星数 / 颗", rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vis_df, use_container_width=True, hide_index=True)

with tab_pos:
    if nav is None:
        st.warning("请在左侧上传 RINEX NAV 导航文件以进行定位解算。")
    elif pos_result is None or not pos_result.get("epochs"):
        st.warning("定位解算失败或无有效历元数据。请确认：\n1) OBS 文件含伪距观测值\n2) NAV 文件卫星与 OBS 匹配\n3) 每历元至少有 4 颗可用卫星")
    else:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">定位结果概览</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            metric_card("有效历元", f"{pos_result.get('n_epochs', 0)}", "")
        with c2:
            metric_card("平均卫星数", fmt(pos_result.get("mean_nsat", 0), 1), "颗")
        with c3:
            gdop_mean = np.nanmean([g for g in pos_result["GDOP"] if np.isfinite(g)]) if pos_result["GDOP"] else float("nan")
            metric_card("平均GDOP", fmt(gdop_mean, 1), "")
        with c4:
            pdop_mean = np.nanmean([p for p in pos_result["PDOP"] if np.isfinite(p)]) if pos_result["PDOP"] else float("nan")
            metric_card("平均PDOP", fmt(pdop_mean, 1), "")
        with c5:
            metric_card("残差RMS", fmt(pos_result.get("rms_residual", [float("nan")])[-1] if pos_result.get("rms_residual") else float("nan"), 3), "m")
        with c6:
            metric_card("解算模式", "伪距SPP", pos_system)

        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">平均坐标</div>', unsafe_allow_html=True)
            stat_df = pd.DataFrame({
                "分量": ["B (°)", "L (°)", "H (m)", "X (m)", "Y (m)", "Z (m)"],
                "均值": [
                    fmt(pos_result.get("mean_lat"), 8),
                    fmt(pos_result.get("mean_lon"), 8),
                    fmt(pos_result.get("mean_h"), 3),
                    fmt(pos_result.get("mean_X"), 3),
                    fmt(pos_result.get("mean_Y"), 3),
                    fmt(pos_result.get("mean_Z"), 3),
                ],
                "标准差(1σ)": [
                    fmt(pos_result.get("std_lat"), 6),
                    fmt(pos_result.get("std_lon"), 6),
                    fmt(pos_result.get("std_h"), 3),
                    fmt(pos_result.get("std_X"), 3),
                    fmt(pos_result.get("std_Y"), 3),
                    fmt(pos_result.get("std_Z"), 3),
                ],
            })
            st.dataframe(stat_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">DOP 值统计</div>', unsafe_allow_html=True)
            dop_df = pd.DataFrame({
                "指标": ["GDOP", "PDOP", "HDOP", "VDOP", "TDOP"],
                "均值": [
                    fmt(np.nanmean([g for g in pos_result["GDOP"] if np.isfinite(g)]) if pos_result["GDOP"] else float("nan"), 2),
                    fmt(np.nanmean([p for p in pos_result["PDOP"] if np.isfinite(p)]) if pos_result["PDOP"] else float("nan"), 2),
                    fmt(np.nanmean([h for h in pos_result["HDOP"] if np.isfinite(h)]) if pos_result["HDOP"] else float("nan"), 2),
                    fmt(np.nanmean([v for v in pos_result["VDOP"] if np.isfinite(v)]) if pos_result["VDOP"] else float("nan"), 2),
                    fmt(np.nanmean([t for t in pos_result["TDOP"] if np.isfinite(t)]) if pos_result["TDOP"] else float("nan"), 2),
                ],
            })
            st.dataframe(dop_df, use_container_width=True, hide_index=True)
            st.caption("参考：GDOP < 2 优秀，2~4 良好，4~6 一般，> 6 较差；定位解算需要 GDOP 尽量小。")
            st.markdown('</div>', unsafe_allow_html=True)

        # 坐标时间序列图
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">坐标与 DOP 时间序列</div>', unsafe_allow_html=True)
        fig_pos = go.Figure()
        if pos_result["X"]:
            x_arr = np.array(pos_result["X"])
            y_arr = np.array(pos_result["Y"])
            z_arr = np.array(pos_result["Z"])
            epochs_pos = pos_result["epochs"]
            fig_pos.add_trace(go.Scatter(x=epochs_pos, y=x_arr - np.nanmean(x_arr),
                                         mode="lines+markers", name="ΔX", marker=dict(size=3)))
            fig_pos.add_trace(go.Scatter(x=epochs_pos, y=y_arr - np.nanmean(y_arr),
                                         mode="lines+markers", name="ΔY", marker=dict(size=3)))
            fig_pos.add_trace(go.Scatter(x=epochs_pos, y=z_arr - np.nanmean(z_arr),
                                         mode="lines+markers", name="ΔZ", marker=dict(size=3)))
        make_plot_layout(fig_pos, "坐标变化 (去均值)")
        fig_pos.update_yaxes(title="Δ / m")
        st.plotly_chart(fig_pos, use_container_width=True)

        fig_dop = go.Figure()
        if pos_result["GDOP"]:
            fig_dop.add_trace(go.Scatter(x=pos_result["epochs"], y=pos_result["GDOP"],
                                         mode="lines", name="GDOP"))
            fig_dop.add_trace(go.Scatter(x=pos_result["epochs"], y=pos_result["PDOP"],
                                         mode="lines", name="PDOP"))
            fig_dop.add_trace(go.Scatter(x=pos_result["epochs"], y=pos_result["HDOP"],
                                         mode="lines", name="HDOP"))
            fig_dop.add_trace(go.Scatter(x=pos_result["epochs"], y=pos_result["VDOP"],
                                         mode="lines", name="VDOP"))
        make_plot_layout(fig_dop, "DOP 值时间序列")
        fig_dop.update_yaxes(title="DOP")
        st.plotly_chart(fig_dop, use_container_width=True)

        # 卫星数
        fig_nsat = go.Figure()
        fig_nsat.add_trace(go.Scatter(x=pos_result["epochs"], y=pos_result["nsat"],
                                      mode="lines", name="可用卫星数",
                                      fill="tozeroy", line=dict(color="#38bdf8")))
        make_plot_layout(fig_nsat, "每历元可用卫星数")
        fig_nsat.update_yaxes(title="卫星数 / 颗", rangemode="tozero")
        st.plotly_chart(fig_nsat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 坐标数据表
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">逐历元定位结果</div>', unsafe_allow_html=True)
        pdf = pos_dataframe(pos_result)
        if not pdf.empty:
            st.dataframe(pdf, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


with tab_orbit:
    if nav is None:
        st.warning("请在左侧上传 RINEX NAV 导航文件。")
    else:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">导航星历概览</div>', unsafe_allow_html=True)
        nav_prns = nav.get_all_prns()
        total_eph = sum(len(nav._eph.get(p, [])) for p in nav_prns)

        c1, c2, c3 = st.columns(3)
        c1.metric("星历卫星数", len(nav_prns))
        c2.metric("星历总数", total_eph)
        c3.metric("卫星系统", "GPS" if nav.system == "G" else ("BDS" if nav.system == "C" else nav.system))

        if nav.iono_params:
            st.caption(f"电离层参数 (Klobuchar): α={nav.iono_params.get('alpha', [])}, β={nav.iono_params.get('beta', [])}")
        else:
            st.caption("未包含电离层改正参数")
        st.markdown('</div>', unsafe_allow_html=True)

        # 选择历元和卫星做天空图
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">卫星天空图</div>', unsafe_allow_html=True)
        if not data or not sats:
            st.warning("需要先加载 OBS 数据。")
        elif pos_result is not None and pos_result.get("epochs"):
            epoch_choice = st.selectbox("选择历元", pos_result["epochs"], key="orbit_epoch",
                                        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            eph_dict = get_nav_eph_dict(nav)
            azel_list = []
            for prn in sats:
                eph = None
                for e in eph_dict.get(prn, []):
                    toe = e.get('toe', 0.0)
                    if toe > 0:
                        eph = e
                        break
                if eph is None:
                    continue
                sat_pos = calc_sat_position(prn, eph, epoch_choice, 'C' if prn.startswith('C') else 'G')
                if sat_pos is None:
                    continue
                # 使用 SPP 坐标或近似坐标
                rec_x = pos_result.get("mean_X", 0)
                rec_y = pos_result.get("mean_Y", 0)
                rec_z = pos_result.get("mean_Z", 0)
                if not np.isfinite(rec_x):
                    rec_x, rec_y, rec_z = 0, 0, 0
                elev, az, _ = calc_sat_azel((sat_pos["X"], sat_pos["Y"], sat_pos["Z"]), (rec_x, rec_y, rec_z))
                if elev >= 0:
                    azel_list.append((elev, az, prn))
            if azel_list:
                sky = make_skyplot(azel_list, f"天空图 {epoch_choice.strftime('%Y-%m-%d %H:%M:%S')}")
                st.plotly_chart(sky, use_container_width=True)
            else:
                st.warning("该历元无有效卫星。")
        else:
            st.warning("需先完成定位解算才能展示天空图。")
        st.markdown('</div>', unsafe_allow_html=True)

        # 卫星星历详情
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">广播星历参数详情</div>', unsafe_allow_html=True)
        nav_sel_prn = st.selectbox("选择卫星查看星历", nav_prns, key="nav_prn_detail")
        if nav_sel_prn:
            ephs = nav._eph.get(nav_sel_prn, [])
            if ephs:
                eph = ephs[0]
                eph_rows = [
                    ["卫星", nav_sel_prn],
                    ["GPS周", eph.get("gps_week", "")],
                    ["Toe (周内秒)", fmt(eph.get("toe", 0), 0)],
                    ["√A (m^0.5)", fmt(eph.get("sqrt_a", 0), 3)],
                    ["e (偏心率)", fmt(eph.get("e", 0), 8)],
                    ["i0 (°)", fmt(eph.get("i0", 0), 6)],
                    ["Ω0 (°)", fmt(eph.get("omega0", 0), 6)],
                    ["ω (°)", fmt(eph.get("omega", 0), 6)],
                    ["M0 (°)", fmt(eph.get("m0", 0), 6)],
                    ["Δn (rad/s)", fmt(eph.get("delta_n", 0), 9)],
                    ["Ω̇ (rad/s)", fmt(eph.get("omega_dot", 0), 9)],
                    ["IDOT (rad/s)", fmt(eph.get("idot", 0), 9)],
                    ["Cuc (rad)", fmt(eph.get("cuc", 0), 8)],
                    ["Cus (rad)", fmt(eph.get("cus", 0), 8)],
                    ["Crc (m)", fmt(eph.get("crc", 0), 3)],
                    ["Crs (m)", fmt(eph.get("crs", 0), 3)],
                    ["Cic (rad)", fmt(eph.get("cic", 0), 8)],
                    ["Cis (rad)", fmt(eph.get("cis", 0), 8)],
                    ["TGD (s)", fmt(eph.get("tgd", 0), 9)],
                    ["钟偏 (s)", fmt(eph.get("sv_clock_bias", 0), 9)],
                    ["钟漂 (s/s)", fmt(eph.get("sv_clock_drift", 0), 9)],
                ]
                st.dataframe(pd.DataFrame(eph_rows, columns=["参数", "值"]), use_container_width=True, hide_index=True)
                st.caption("仅显示该卫星第一组星历，用于参考。")
        st.markdown('</div>', unsafe_allow_html=True)


with tab_coord:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">坐标转换</div>', unsafe_allow_html=True)
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
                f"H={h:.4f} m",
                "",
                f"3°带中央子午线: {cm3}°",
                f"x(北)={x3:.4f} m",
                f"y(东)={y3:.4f} m",
                "=" * 56,
            ]
            st.code("\n".join(lines), language="text")

    else:
        c1, c2, c3 = st.columns(3)
        gx = c1.number_input("x 北坐标 / m", value=4419100.0, format="%.4f")
        gy = c2.number_input("y 东坐标 / m", value=536400.0, format="%.4f")
        cm = c3.number_input("中央子午线 / °", value=117.0, format="%.4f")
        if st.button("开始转换", key="gauss_convert"):
            lat, lon, h = gauss_inverse(gx, gy, cm, ell)
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
                f"XYZ: X={X:.4f} m, Y={Y:.4f} m, Z={Z:.4f} m",
                "=" * 56,
            ]
            st.code("\n".join(lines), language="text")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_report:
    report_text = quality_evaluation_report(data, obs_types, pos_result)
    full_report = quality_summary(data, obs_types, pos_result)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "下载综合质量评价 TXT",
            data=report_text.encode("utf-8"),
            file_name="GNSS数据质量评价报告.txt",
            mime="text/plain",
        )
    with c2:
        st.download_button(
            "下载完整分析报告 TXT",
            data=full_report.encode("utf-8"),
            file_name="GNSS观测数据质量分析报告.txt",
            mime="text/plain",
        )

    st.markdown("#### 综合质量评价")
    st.code(report_text, language="text")
    st.markdown("#### 完整分析报告")
    st.code(full_report, language="text")
