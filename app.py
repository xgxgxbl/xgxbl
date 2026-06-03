"""
GNSS 控制网观测数据质量检查程序 — Streamlit 前端
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
from gns_reader import GnsFile
from gnss_quality import cycle_slip_detect, station_slip_summary
from sync_loop import SyncLoopChecker, generate_example_baseline

st.set_page_config(
    page_title="GNSS控制网观测数据质量检查",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 大理石渐变配色 CSS
# =============================================================================
st.markdown(
    """
    <style>
    :root {
        --warm-white: #faf7f2;
        --cream: #f3ede3;
        --sand: #ede4d3;
        --text-dark: #3d2e1f;
        --text-medium: #7a6b5c;
        --accent-gold: #b8935c;
        --accent-amber: #c8843c;
        --accent-rose: #b87a6a;
        --border-warm: #ddd0bf;
        --success: #5a8f4a;
        --danger: #b85450;
        --warning: #c4963a;
    }
    .stApp {
        background: linear-gradient(135deg, #faf7f2 0%, #f3ede3 40%, #efe6d8 100%);
        color: #3d2e1f;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5efe4 0%, #ede4d3 100%);
        border-right: 1px solid #ddd0bf;
    }
    [data-testid="stSidebar"] * { color: #3d2e1f; }
    [data-testid="stSidebar"] .stCaption { color: #7a6b5c; }
    .marble-card {
        padding: 1.2rem 1.4rem;
        border: 1px solid #ddd0bf;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,252,248,0.95), rgba(248,242,233,0.85));
        box-shadow: 0 8px 30px rgba(139,115,85,0.08);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 0.9rem 1rem;
        border: 1px solid #ddd0bf;
        border-radius: 16px;
        background: linear-gradient(180deg, #ffffff, #faf6ef);
        box-shadow: 0 6px 24px rgba(139,115,85,0.06);
        text-align: center;
        min-height: 100px;
    }
    .metric-card .value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #3d2e1f;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: #7a6b5c;
        margin-top: 0.2rem;
    }
    .hero {
        padding: 1.2rem 1.5rem;
        border: 1px solid #ddd0bf;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(200,165,120,0.10), rgba(250,247,242,0.92) 40%, rgba(220,185,145,0.06));
        box-shadow: 0 12px 40px rgba(139,115,85,0.06);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 850;
        color: #2d2015;
        letter-spacing: -0.02em;
        margin: 0 0 0.3rem 0;
    }
    .hero-subtitle {
        color: #7a6b5c;
        font-size: 0.95rem;
        margin: 0;
    }
    .pill-row { margin-top: 0.7rem; }
    .pill {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(200,132,60,0.25);
        background: rgba(200,132,60,0.08);
        color: #8b6d4b;
        font-size: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 999px;
        padding: 0 1.1rem;
        border: 1px solid #ddd0bf;
        background: rgba(255,252,248,0.7);
        color: #7a6b5c;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(200,132,60,0.15), rgba(184,122,106,0.10));
        color: #c8843c;
        border-color: rgba(200,132,60,0.35);
        font-weight: 700;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(200,132,60,0.35);
        background: linear-gradient(135deg, rgba(200,132,60,0.12), rgba(184,122,106,0.08));
        color: #b87a4a;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(200,132,60,0.7);
        color: #c8843c;
    }

    .closure-pass { color: #5a8f4a; font-weight: 800; }
    .closure-fail { color: #b85450; font-weight: 800; }
    .closure-warn { color: #c4963a; font-weight: 800; }

    .section-title {
        font-size: 1.1rem;
        font-weight: 750;
        color: #2d2015;
        margin-bottom: 0.5rem;
    }
    .small-muted { color: #7a6b5c; font-size: 0.85rem; }
    .alert-card {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(184,84,80,0.30);
        background: rgba(255,240,238,0.7);
        color: #b85450;
        font-size: 0.9rem;
        margin: 0.6rem 0;
    }
    .info-card {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(90,143,74,0.25);
        background: rgba(240,248,238,0.7);
        color: #4a7a3e;
        font-size: 0.9rem;
        margin: 0.6rem 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,252,248,0.8);
        border: 1px solid #ddd0bf;
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
    }
    div[data-testid="stMetric"] label { color: #7a6b5c !important; }
    div[data-testid="stMetricValue"] { color: #3d2e1f !important; }
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 工具函数
# =============================================================================
SYSTEM_NAMES = {'G': 'GPS', 'C': 'BDS', 'E': 'GAL', 'R': 'GLO'}


def fmt(value, digits=1, suffix=""):
    try:
        if value is None or not np.isfinite(float(value)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    if digits == 0:
        return f"{int(round(float(value)))}{suffix}"
    return f"{float(value):.{digits}f}{suffix}"


def system_label(code: str) -> str:
    return SYSTEM_NAMES.get(code, code)


def make_plot_layout(fig: go.Figure, title: str | None = None):
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,240,235,0.30)",
        margin=dict(l=35, r=22, t=52 if title else 28, b=35),
        font=dict(family="Microsoft YaHei, Arial", color="#3d2e1f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        colorway=["#c8843c", "#b8935c", "#b87a6a", "#5a8f4a", "#8b7355", "#c4963a"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,165,145,0.20)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,165,145,0.20)")
    return fig


@st.cache_data(show_spinner=False)
def parse_rinex_cached(file_bytes: bytes, filename: str) -> RinexObs:
    suffix = os.path.splitext(filename)[1] or ".obs"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    return RinexObs(path)


# =============================================================================
# 页面头部
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛰️ GNSS 控制网观测数据质量检查</div>
        <p class="hero-subtitle">多测站管理 · RINEX 观测统计 · 同步环检核 · 周跳探测</p>
        <div class="pill-row">
            <span class="pill">多测站</span>
            <span class="pill">观测统计</span>
            <span class="pill">同步环检核</span>
            <span class="pill">MW+GF 周跳</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 侧边栏
# =============================================================================
with st.sidebar:
    st.markdown("### 📡 数据输入")

    st.caption("上传多个 RINEX OBS / GNS 观测文件")
    uploaded_files = st.file_uploader(
        "选择观测文件",
        type=["obs","OBS","23o","24o","25o","26o","o","O",
              "gns","GNS",
              "rnx","RNX","crx","CRX"],
        accept_multiple_files=True,
        key="uploaded_rinex",
    )

    st.markdown("---")
    st.caption("上传基线处理成果文件（可选，用于同步环检核）")
    uploaded_baseline = st.file_uploader(
        "选择基线成果文件",
        type=["txt", "csv", "dat"],
        key="uploaded_baseline",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 下载示例基线", key="dl_example"):
            example = generate_example_baseline()
            st.download_button(
                "点击下载示例基线.txt",
                data=example,
                file_name="示例基线成果.txt",
                mime="text/plain",
            )

    st.markdown("---")
    st.markdown("### ⚙️ 周跳探测参数")
    mw_threshold = st.number_input("MW阈值 / 周", value=3.0, min_value=0.1, max_value=20.0, step=0.1)
    gf_threshold = st.number_input("GF阈值 / m", value=0.15, min_value=0.01, max_value=5.0, step=0.01)

    st.markdown("---")
    st.markdown("### 📋 项目信息")
    st.caption("卫星导航定位课程设计")
    st.caption("控制网观测数据处理与质量检查")

# =============================================================================
# 数据加载
# =============================================================================
rinex_list: list[tuple[str, object]] = []   # (测站名, RinexObs | GnsFile)
gns_only_stations: list[str] = []            # 仅有GNS、无法周跳探测的测站

if uploaded_files:
    with st.spinner("正在解析观测文件……"):
        for uf in uploaded_files:
            try:
                raw_bytes = uf.getvalue()
                # 智能格式检测：检查文件头
                is_gns = (raw_bytes[:30].find(b"ZHD COLLECTED") >= 0 or
                          raw_bytes[:30].find(b"HITRTK") >= 0 or
                          uf.name.lower().endswith(".gns"))

                if is_gns:
                    # GNS 文件：保存临时文件后用 GnsFile 解析
                    suffix = os.path.splitext(uf.name)[1] or ".GNS"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    gobj = GnsFile(tmp_path)
                    stn = gobj.station_name
                    rinex_list.append((stn, gobj))
                    if not gobj.has_obs_data:
                        gns_only_stations.append(stn)
                else:
                    # RINEX 文件
                    robj = parse_rinex_cached(raw_bytes, uf.name)
                    stn = robj.header.get("marker_name", uf.name.rsplit(".", 1)[0])
                    rinex_list.append((stn, robj))
            except Exception as exc:
                st.error(f"解析 {uf.name} 失败：{exc}")

checker = None
if uploaded_baseline:
    try:
        text = uploaded_baseline.getvalue().decode("utf-8", errors="replace")
        checker = SyncLoopChecker(text)
    except Exception as exc:
        st.error(f"解析基线文件失败：{exc}")

# =============================================================================
# 未上传文件时的空状态
# =============================================================================
if not rinex_list:
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        st.markdown('<div class="marble-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">使用说明</div>', unsafe_allow_html=True)
        st.markdown(
            """
            在左侧上传 **多个 RINEX OBS 观测文件** 即可开始分析：

            1. **观测统计** — 自动识别测站，统计历元数、卫星系统组成、采样间隔
            2. **同步环检核** — 上传基线成果文件后，自动提取同步环并检核闭合差
            3. **周跳探测** — MW+GF 双组合探测，关联同步环质量分析

            > 无基线文件？点击侧边栏"下载示例基线"获取测试文件。
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="value">📡</div><div class="label">多测站管理</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="value">🔍</div><div class="label">同步环检核</div></div>', unsafe_allow_html=True)
    st.stop()

# =============================================================================
# 标签页
# =============================================================================
tab_stats, tab_sync, tab_slip = st.tabs(["📊 观测统计", "🔗 同步环检核", "📈 周跳探测"])

# =============================================================================
# 标签页1: 观测统计
# =============================================================================
with tab_stats:
    st.markdown('<div class="marble-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">多测站观测数据汇总</div>', unsafe_allow_html=True)

    # 汇总表
    rows = []
    for stn, robj in rinex_list:
        comp = robj.get_system_composition()
        interval = robj.get_sampling_interval()
        is_gns = isinstance(robj, GnsFile)

        if is_gns:
            sats = robj.get_satellites()
            rows.append({
                "测站名": stn,
                "文件名": os.path.basename(robj.filepath),
                "格式": "GNS",
                "历元数": robj.n_epochs,
                "卫星数": len(sats),
                "GPS": comp.get('G', 0),
                "BDS": comp.get('C', 0),
                "GAL": comp.get('E', 0),
                "GLO": comp.get('R', 0),
                "采样间隔": "?" if interval is None else f"{interval:.0f}s",
                "观测时长": "?",
                "起始时间": robj.header.get("date", "?"),
                "结束时间": robj.header.get("date", "?"),
            })
        else:
            times = robj.get_times()
            sats = robj.get_satellites()
            start_str = times[0].strftime("%Y-%m-%d %H:%M:%S") if times else "?"
            end_str = times[-1].strftime("%Y-%m-%d %H:%M:%S") if times else "?"
            dur_min = (times[-1] - times[0]).total_seconds() / 60.0 if len(times) >= 2 else 0

            rows.append({
                "测站名": stn,
                "文件名": os.path.basename(robj.filepath),
                "格式": "RINEX",
                "历元数": len(times),
                "卫星数": len(sats),
                "GPS": comp.get('G', 0),
                "BDS": comp.get('C', 0),
                "GAL": comp.get('E', 0),
                "GLO": comp.get('R', 0),
                "采样间隔": f"{interval:.0f}s" if interval else "?",
                "观测时长": f"{dur_min:.0f}min",
                "起始时间": start_str,
                "结束时间": end_str,
            })

    df_stats = pd.DataFrame(rows)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # KPI 卡片行
    total_stations = len(rinex_list)
    total_epochs_sum = 0
    all_sats_union = set()
    for _, robj in rinex_list:
        if isinstance(robj, GnsFile):
            total_epochs_sum += robj.n_epochs
        else:
            total_epochs_sum += len(robj.get_times())
        all_sats_union.update(robj.get_satellites())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("测站数", total_stations)
    c2.metric("总历元数", total_epochs_sum)
    c3.metric("不同卫星总数", len(all_sats_union))
    if checker:
        c4.metric("基线数", checker.n_baselines)
    else:
        c4.metric("基线数", "未上传")

    # 卫星系统组成柱状图
    if len(rinex_list) > 1:
        fig_bar = go.Figure()
        sys_types = ['G', 'C', 'E', 'R']
        for s in sys_types:
            vals = []
            for _, robj in rinex_list:
                comp = robj.get_system_composition()
                vals.append(comp.get(s, 0))
            if any(v > 0 for v in vals):
                fig_bar.add_trace(go.Bar(
                    name=system_label(s),
                    x=[stn for stn, _ in rinex_list],
                    y=vals,
                ))
        fig_bar.update_layout(
            barmode="stack",
            title="各测站卫星系统组成",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(245,240,235,0.30)",
            font=dict(family="Microsoft YaHei, Arial", color="#3d2e1f"),
            yaxis=dict(title="卫星数 / 颗", gridcolor="rgba(180,165,145,0.20)"),
            xaxis=dict(gridcolor="rgba(180,165,145,0.20)"),
            colorway=["#c8843c", "#b8935c", "#b87a6a", "#5a8f4a"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 标签页2: 同步环检核
# =============================================================================
with tab_sync:
    if checker is None:
        st.markdown('<div class="marble-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">同步环检核</div>', unsafe_allow_html=True)
        st.info("请先在左侧上传**基线处理成果文件**以进行同步环检核。\n\n"
                "基线文件格式示例：\n```\n"
                "# 起点 终点 dX(m) dY(m) dZ(m) sigma(mm)\n"
                "BJFS SHAO   1234567.890  -234567.890   345678.901  3.0\n"
                "BJFS WUHN   -987654.321   876543.210  -765432.109  3.5\n"
                "SHAO WUHN  -2222222.211  1111110.100  1111111.010  3.2\n```\n\n"
                "也可点击侧边栏 **\"下载示例基线\"** 获取测试文件。")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        summary = checker.summary()

        # KPI 卡片
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="value">{summary["n_stations"]}</div><div class="label">测站数</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="value">{summary["n_baselines"]}</div><div class="label">基线数</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="value">{summary["n_loops"]}</div><div class="label">同步环数</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="value">{summary["n_pass"]}</div><div class="label">合格</div></div>', unsafe_allow_html=True)
        with c5:
            fail_count = summary["n_fail"]
            color = "#5a8f4a" if fail_count == 0 else "#b85450"
            st.markdown(f'<div class="metric-card"><div class="value" style="color:{color}">{fail_count}</div><div class="label">不合格</div></div>', unsafe_allow_html=True)

        # 合格率
        pass_rate = summary["pass_rate"]
        if pass_rate == 100:
            st.markdown(f'<div class="info-card">✅ 全部 {summary["n_loops"]} 个同步环均通过检核，闭合差合格率 100%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-card">⚠️ {fail_count} 个同步环闭合差超限，合格率 {pass_rate:.1f}%，请检查对应基线</div>', unsafe_allow_html=True)

        # 闭合差汇总表
        st.markdown('<div class="marble-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">同步环闭合差汇总</div>', unsafe_allow_html=True)

        table_rows = []
        for i, r in enumerate(summary["results"], 1):
            table_rows.append({
                "环号": i,
                "测站组合": f"{r['loop'][0]} → {r['loop'][1]} → {r['loop'][2]}",
                "ΔX/mm": fmt(r['dX_closure_mm'], 1),
                "ΔY/mm": fmt(r['dY_closure_mm'], 1),
                "ΔZ/mm": fmt(r['dZ_closure_mm'], 1),
                "3D闭合差/mm": fmt(r['closure_3d_mm'], 2),
                "限差/mm": fmt(r['tolerance_mm'], 1),
                "判定": "✅ 合格" if r['pass'] else "❌ 不合格",
            })

        df_sync = pd.DataFrame(table_rows)
        st.dataframe(df_sync, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 单个环详情
        if summary["results"]:
            st.markdown('<div class="marble-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">单环详情</div>', unsafe_allow_html=True)

            loop_choices = [f"环{i+1}: {r['loop'][0]}→{r['loop'][1]}→{r['loop'][2]}" for i, r in enumerate(summary["results"])]
            selected_loop = st.selectbox("选择同步环查看详情", loop_choices, key="loop_detail")
            sel_idx = loop_choices.index(selected_loop)
            result = summary["results"][sel_idx]

            # 闭合差分量子图
            c1, c2 = st.columns([1, 1])
            with c1:
                comp_fig = go.Figure()
                comps = ['ΔX', 'ΔY', 'ΔZ']
                vals = [result['dX_closure_mm'], result['dY_closure_mm'], result['dZ_closure_mm']]
                colors = ['#5a8f4a' if abs(v) <= result['tolerance_mm'] / 3 else '#c4963a' if abs(v) <= result['tolerance_mm'] else '#b85450' for v in vals]
                comp_fig.add_trace(go.Bar(x=comps, y=vals, marker_color=colors, name="闭合差分量"))
                comp_fig.add_hline(y=result['tolerance_mm'] / 3, line_dash="dash", line_color="rgba(180,165,145,0.5)", annotation_text="限差")
                comp_fig.add_hline(y=-result['tolerance_mm'] / 3, line_dash="dash", line_color="rgba(180,165,145,0.5)")
                make_plot_layout(comp_fig, f"闭合差分量 ({result['loop'][0]}→{result['loop'][1]}→{result['loop'][2]})")
                comp_fig.update_layout(yaxis=dict(title="闭合差 / mm"))
                st.plotly_chart(comp_fig, use_container_width=True)

            with c2:
                # 测站三角拓扑
                A, B, C = result['loop']
                # 简易布局：等边三角形
                angles = np.linspace(0, 2 * np.pi, 4)[:3]
                radius = 1
                pos_x = radius * np.cos(angles)
                pos_y = radius * np.sin(angles)
                stations = [A, B, C]

                topo_fig = go.Figure()
                # 三条边
                for i, j in [(0, 1), (1, 2), (2, 0)]:
                    topo_fig.add_trace(go.Scatter(
                        x=[pos_x[i], pos_x[j]], y=[pos_y[i], pos_y[j]],
                        mode="lines+markers",
                        line=dict(color="#b8935c", width=2),
                        marker=dict(size=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                # 三个节点
                topo_fig.add_trace(go.Scatter(
                    x=pos_x, y=pos_y,
                    mode="markers+text",
                    marker=dict(size=28, color="#faf6ef", line=dict(color="#c8843c", width=2)),
                    text=stations,
                    textposition="top center",
                    textfont=dict(color="#3d2e1f", size=13),
                    showlegend=False,
                ))
                # 标注闭合差
                mid_x = np.mean(pos_x)
                mid_y = np.mean(pos_y)
                status_text = "✓ 合格" if result['pass'] else "✗ 不合格"
                topo_fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"{status_text}<br>3D={result['closure_3d_mm']:.1f}mm",
                    showarrow=False,
                    font=dict(color="#5a8f4a" if result['pass'] else "#b85450", size=13),
                )
                topo_fig.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(topo_fig, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 标签页3: 周跳探测
# =============================================================================
with tab_slip:
    # 测站选择
    station_choices = [stn for stn, _ in rinex_list]
    if len(station_choices) == 0:
        st.warning("请先上传观测文件")
    else:
        selected_stn = st.selectbox("选择测站", station_choices, key="slip_station")

        # 找到对应的解析对象
        stn_to_obj = {stn: robj for stn, robj in rinex_list}
        robj = stn_to_obj[selected_stn]

        # GNS 文件不支持周跳探测
        if isinstance(robj, GnsFile):
            st.info(
                f"📋 **{selected_stn}** 是 GNS 原始二进制格式，"
                "暂不支持直接进行周跳探测。\n\n"
                "请使用 **RTKLIB convbin** 工具将 GNS 转换为 RINEX OBS 格式后重新上传。\n\n"
                "```bash\nconvbin -r gps -v 2.11 -od -os input.GNS\n```\n\n"
                f"该文件基本信息：{robj.n_epochs} 个历元，{len(robj.satellites)} 颗卫星，"
                f"观测日期 {robj.header.get('date', '?')}。"
            )
        else:
            data = robj.get_data()
            obs_types = robj.obs_types
            sats = robj.get_satellites()

            if not sats:
                st.warning("该文件没有卫星数据")
            else:
                # 测站周跳汇总
                slip_summary = station_slip_summary(data, obs_types, selected_stn, mw_threshold, gf_threshold)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("卫星数", slip_summary["sat_count"])
                c2.metric("总周跳数", slip_summary["total_slips"])
                c3.metric("存在周跳的卫星数", slip_summary["sat_with_slips"])
                c4.metric("MW阈值", f"{mw_threshold:.1f} 周")

                # 与同步环的关联分析
                if checker and slip_summary["total_slips"] > 0:
                    sloops = checker.station_loops(selected_stn)
                    if sloops:
                        failed = [r for r in sloops if not r['pass']]
                        if failed:
                            failed_ids = [f"环 {r['loop'][0]}→{r['loop'][1]}→{r['loop'][2]}" for r in failed]
                            st.markdown(
                                f'<div class="alert-card">'
                                f'⚠️ <b>{selected_stn}</b> 共检测到 <b>{slip_summary["total_slips"]}</b> 处可疑周跳，'
                                f'且该测站参与了 <b>{len(failed)}</b> 个不合格同步环：<br>'
                                f'{"<br>".join(failed_ids)}<br>'
                                f'<small>周跳可能导致载波相位观测值不连续，建议检查对应基线处理结果。</small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="info-card">'
                                f'✅ {selected_stn} 虽有 {slip_summary["total_slips"]} 处周跳，但参与的所有同步环均合格。'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # 选择卫星查看详细周跳曲线
                sat_choice = st.selectbox("选择卫星", sats, key="slip_sat_detail")

                if sat_choice:
                    slips, mw_series, gf_series = cycle_slip_detect(data, obs_types, sat_choice, mw_threshold, gf_threshold)
                    epochs = data.get(sat_choice, {}).get("epoch", [])

                c1, c2, c3 = st.columns(3)
                c1.metric(f"{sat_choice} 可疑周跳", len(slips))
                c2.metric("MW阈值", f"{mw_threshold:.1f} 周")
                c3.metric("GF阈值", f"{gf_threshold:.2f} m")

                # MW+GF 曲线图
                fig = go.Figure()
                if epochs:
                    fig.add_trace(go.Scatter(
                        x=epochs, y=mw_series,
                        mode="lines", name="MW组合 / 周",
                        line=dict(color="#c8843c"),
                        yaxis="y1",
                    ))
                    fig.add_trace(go.Scatter(
                        x=epochs, y=gf_series,
                        mode="lines", name="GF组合 / m",
                        line=dict(color="#b87a6a"),
                        yaxis="y2",
                    ))
                    for idx, t, mw_jump, gf_jump in slips:
                        fig.add_vline(x=t, line_width=1, line_dash="dash", line_color="#b85450")
                fig.update_layout(
                    yaxis=dict(title="MW / 周", titlefont=dict(color="#c8843c")),
                    yaxis2=dict(title="GF / m", overlaying="y", side="right", titlefont=dict(color="#b87a6a")),
                )
                make_plot_layout(fig, f"{sat_choice} MW+GF 周跳探测")
                st.plotly_chart(fig, use_container_width=True)

                # 周跳列表
                if slips:
                    slip_df = pd.DataFrame([
                        {"序号": idx, "历元": t.strftime("%Y-%m-%d %H:%M:%S"), "MW跳变/周": fmt(mw, 2), "GF跳变/m": fmt(gf, 3)}
                        for idx, t, mw, gf in slips
                    ])
                    st.dataframe(slip_df, use_container_width=True, hide_index=True)
                else:
                    st.success(f"{sat_choice} 未检测到明显周跳。")
