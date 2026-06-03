"""
GNSS 观测数据质量分析模块
- SNR 分析
- MW+GF 周跳探测
- 多路径效应计算
- 卫星可见性统计
- 综合数据质量评价
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ------------------------------------------------------------
# 频率常数 (MHz)
# ------------------------------------------------------------
FREQ = {
    'G': {'L1': 1575.42, 'L2': 1227.60, 'L5': 1176.45},
    'C': {'B1': 1561.098, 'B2': 1207.140, 'B3': 1268.520, 'L1': 1561.098, 'L2': 1207.140},
    'E': {'E1': 1575.42, 'E5a': 1176.45, 'E5b': 1207.140, 'L1': 1575.42, 'L2': 1207.140},
    'R': {'L1': 1602.00, 'L2': 1246.00},  # 简化处理，未考虑 GLONASS FDMA 频点号
}

# 光速 m/s
C = 299792458.0


def _get_freq(sat: str, bands: Tuple[str, str] = ('L1', 'L2')) -> Tuple[float, float]:
    """根据卫星系统获取双频频率，返回 Hz。"""
    sys = sat[0] if sat and sat[0] in FREQ else 'G'
    fmap = FREQ.get(sys, FREQ['G'])
    f1 = fmap.get(bands[0], fmap.get('L1', 1575.42))
    f2 = fmap.get(bands[1], fmap.get('L2', 1227.60))
    return f1 * 1e6, f2 * 1e6


def _as_float_array(values: Iterable[Any], length: int | None = None) -> np.ndarray:
    """安全转换为 float 数组，并可按 length 补齐/截断。"""
    arr = np.array(list(values) if values is not None else [], dtype=float)
    if length is None:
        return arr
    if len(arr) == length:
        return arr
    out = np.full(length, np.nan)
    n = min(len(arr), length)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.array(list(values) if values is not None else [], dtype=float)
    return arr[np.isfinite(arr)]


def _rms(values: Iterable[Any]) -> float:
    arr = _finite(values)
    if len(arr) == 0:
        return float('nan')
    return float(np.sqrt(np.mean(arr ** 2)))


def _std(values: Iterable[Any]) -> float:
    arr = _finite(values)
    if len(arr) == 0:
        return float('nan')
    return float(np.std(arr))


def _select_obs_pair(obs_types: List[str], sat: str) -> Tuple[List[str], List[str]]:
    """
    为不同 RINEX 写法选择近似的 L1/L2 与 C1/C2 观测值。
    兼容常见 RINEX2: L1/L2/C1/P1/C2/P2，以及部分 RINEX3 简写。
    """
    ot = set(obs_types)
    sys = sat[0] if sat else 'G'

    # 载波相位候选
    l1_candidates = ['L1', 'L1C', 'L1W', 'L1P', 'L1I', 'L1X']
    l2_candidates = ['L2', 'L2W', 'L2P', 'L2C', 'L2I', 'L2X']
    if sys == 'C':
        l1_candidates = ['L1', 'L1I', 'L1X', 'L2I', 'L2X']
        l2_candidates = ['L2', 'L2I', 'L2X', 'L7I', 'L7X']

    # 伪距候选
    c1_candidates = ['C1', 'P1', 'C1C', 'C1W', 'C1P', 'C1I', 'C1X']
    c2_candidates = ['C2', 'P2', 'C2W', 'C2P', 'C2C', 'C2I', 'C2X']
    if sys == 'C':
        c1_candidates = ['C1', 'C1I', 'C1X', 'C2I', 'C2X']
        c2_candidates = ['C2', 'C2I', 'C2X', 'C7I', 'C7X']

    def pick(candidates):
        for key in candidates:
            if key in ot:
                return key
        return None

    l1 = pick(l1_candidates)
    l2 = pick(l2_candidates)
    c1 = pick(c1_candidates)
    c2 = pick(c2_candidates)
    return [l1, l2], [c1, c2]


def snr_analysis(data: Dict[str, Dict[str, Any]], obs_types: List[str], sat: str):
    """
    SNR 分析：提取指定卫星的信噪比时间序列。

    返回:
        epochs, result
        result 形如 {'epoch': [...], 'S1': [...], 'S2': [...]}
    """
    if sat not in data:
        return [], {}

    sat_data = data[sat]
    snr_keys = [ot for ot in obs_types if ot.startswith('S')]
    result = {'epoch': sat_data.get('epoch', [])}
    for sk in snr_keys:
        result[sk] = sat_data.get(sk, [])
    return result.get('epoch', []), result


def snr_statistics(data: Dict[str, Dict[str, Any]], obs_types: List[str], sat: str | None = None) -> Dict[str, float]:
    """计算 SNR 平均值、最大值、最小值和有效样本数。"""
    sats = [sat] if sat and sat in data else sorted(data.keys())
    snr_keys = [ot for ot in obs_types if ot.startswith('S')]
    values = []
    for s in sats:
        for key in snr_keys:
            values.extend(data.get(s, {}).get(key, []))
    arr = _finite(values)
    if len(arr) == 0:
        return {'avg': np.nan, 'max': np.nan, 'min': np.nan, 'count': 0}
    return {
        'avg': float(np.mean(arr)),
        'max': float(np.max(arr)),
        'min': float(np.min(arr)),
        'count': int(len(arr)),
    }


def cycle_slip_detect(data: Dict[str, Dict[str, Any]], obs_types: List[str], sat: str,
                      mw_threshold: float = 3.0, gf_threshold: float = 0.15):
    """
    MW + GF 组合周跳探测（TurboEdit 简化版）。

    参数:
        mw_threshold: MW 宽巷组合跳变阈值（周），默认 3.0
        gf_threshold: GF 几何无关组合跳变阈值（米），默认 0.15

    返回:
        slips: [(epoch_index, epoch_time, mw_diff, gf_diff), ...]
        mw_series: MW 组合时间序列
        gf_series: GF 组合时间序列
    """
    if sat not in data:
        return [], [], []

    sat_data = data[sat]
    epochs = sat_data.get('epoch', [])
    n = len(epochs)
    if n < 2:
        return [], [], []

    phase_keys, code_keys = _select_obs_pair(obs_types, sat)
    has_phase = all(k is not None for k in phase_keys)
    has_code = all(k is not None for k in code_keys)
    if not has_phase:
        return [], [], []

    f1, f2 = _get_freq(sat, bands=('L1', 'L2'))
    lam1 = C / f1
    lam2 = C / f2

    L1 = _as_float_array(sat_data.get(phase_keys[0], []), n) * lam1
    L2 = _as_float_array(sat_data.get(phase_keys[1], []), n) * lam2

    if has_code:
        P1 = _as_float_array(sat_data.get(code_keys[0], []), n)
        P2 = _as_float_array(sat_data.get(code_keys[1], []), n)
    else:
        P1 = np.full(n, np.nan)
        P2 = np.full(n, np.nan)

    valid = ~(np.isnan(L1) | np.isnan(L2) | np.isnan(P1) | np.isnan(P2))
    if np.sum(valid) < 2:
        # 没有码观测时，仍可给出 GF 序列，但 MW 不参与判别
        valid_gf = ~(np.isnan(L1) | np.isnan(L2))
        gf_series = np.full(n, np.nan)
        gf_series[valid_gf] = L1[valid_gf] - L2[valid_gf]
        slips = []
        valid_idx = np.where(valid_gf)[0]
        for j in range(1, len(valid_idx)):
            idx = valid_idx[j]
            prev_idx = valid_idx[j - 1]
            gf_diff = abs(gf_series[idx] - gf_series[prev_idx])
            if gf_diff > gf_threshold:
                slips.append((int(idx), epochs[idx], float('nan'), float(gf_diff)))
        return slips, list(np.full(n, np.nan)), list(gf_series)

    wl_lam = C / abs(f1 - f2)  # 宽巷波长
    mw_series = np.full(n, np.nan)
    gf_series = np.full(n, np.nan)

    mw_series[valid] = ((f1 * L1[valid] - f2 * L2[valid]) / (f1 - f2) -
                        (f1 * P1[valid] + f2 * P2[valid]) / (f1 + f2)) / wl_lam
    gf_series[valid] = L1[valid] - L2[valid]

    slips = []
    valid_idx = np.where(valid)[0]
    for j in range(1, len(valid_idx)):
        idx = valid_idx[j]
        prev_idx = valid_idx[j - 1]
        mw_diff = abs(mw_series[idx] - mw_series[prev_idx])
        gf_diff = abs(gf_series[idx] - gf_series[prev_idx])
        if mw_diff > mw_threshold or gf_diff > gf_threshold:
            slips.append((int(idx), epochs[idx], float(mw_diff), float(gf_diff)))

    return slips, list(mw_series), list(gf_series)


def multipath_calc(data: Dict[str, Dict[str, Any]], obs_types: List[str], sat: str):
    """
    多路径效应计算。

    公式:
        MP1 = P1 - (α+1)/(α-1)·L1 + 2/(α-1)·L2
        MP2 = P2 - 2α/(α-1)·L1 + (2α/(α-1)-1)·L2
        其中 α = (f1/f2)^2

    返回:
        epochs, mp1_values, mp2_values
    """
    if sat not in data:
        return [], [], []

    sat_data = data[sat]
    epochs = sat_data.get('epoch', [])
    n = len(epochs)
    if n == 0:
        return [], [], []

    phase_keys, code_keys = _select_obs_pair(obs_types, sat)
    has_phase = all(k is not None for k in phase_keys)
    has_code = all(k is not None for k in code_keys)
    if not (has_phase and has_code):
        return [], [], []

    f1, f2 = _get_freq(sat, bands=('L1', 'L2'))
    alpha = (f1 / f2) ** 2
    lam1 = C / f1
    lam2 = C / f2

    a1 = (alpha + 1) / (alpha - 1)
    a2 = 2 / (alpha - 1)
    b1 = 2 * alpha / (alpha - 1)
    b2 = (2 * alpha / (alpha - 1)) - 1

    L1 = _as_float_array(sat_data.get(phase_keys[0], []), n) * lam1
    L2 = _as_float_array(sat_data.get(phase_keys[1], []), n) * lam2
    P1 = _as_float_array(sat_data.get(code_keys[0], []), n)
    P2 = _as_float_array(sat_data.get(code_keys[1], []), n)

    valid = ~(np.isnan(L1) | np.isnan(L2) | np.isnan(P1) | np.isnan(P2))
    if np.sum(valid) < 1:
        return [], [], []

    mp1 = np.full(n, np.nan)
    mp2 = np.full(n, np.nan)
    mp1[valid] = P1[valid] - a1 * L1[valid] + a2 * L2[valid]
    mp2[valid] = P2[valid] - b1 * L1[valid] + b2 * L2[valid]

    # 去掉每颗卫星的常数偏置，使图形更适合观察多路径变化
    if np.sum(np.isfinite(mp1)) > 0:
        mp1[np.isfinite(mp1)] -= np.nanmean(mp1)
    if np.sum(np.isfinite(mp2)) > 0:
        mp2[np.isfinite(mp2)] -= np.nanmean(mp2)

    return epochs, list(mp1), list(mp2)


def visibility_stats(data: Dict[str, Dict[str, Any]]):
    """
    卫星可见性统计。

    返回:
        {epoch: {'n_gps': N, 'n_bds': N, 'n_gal': N, 'n_glo': N, 'total': N}}
    """
    epoch_map = defaultdict(lambda: {'G': 0, 'C': 0, 'E': 0, 'R': 0, 'total': 0, 'sats': []})

    for sat, sat_data in data.items():
        for ep in sat_data.get('epoch', []):
            sys = sat[0] if sat and sat[0] in 'GCER' else 'G'
            epoch_map[ep][sys] += 1
            epoch_map[ep]['total'] += 1
            epoch_map[ep]['sats'].append(sat)

    result = {}
    for ep in sorted(epoch_map.keys()):
        entry = epoch_map[ep]
        result[ep] = {
            'n_gps': entry['G'],
            'n_bds': entry['C'],
            'n_gal': entry['E'],
            'n_glo': entry['R'],
            'total': entry['total'],
        }
    return result


def compute_dop(data, obs_types, station_xyz=None):
    """
    基于卫星方位角/高度角计算 DOP 值。
    不依赖 NAV 星历，而是从观测数据中假设的卫星位置计算。
    若无法确定卫星-接收机几何关系，返回 None。

    在实际使用中建议优先使用 spp_positioning 中的 compute_dop_from_azel，
    该函数基于定位解算过程中已知的卫星高度角和方位角计算 DOP。
    """
    try:
        from satellite_orbit import compute_dop_from_azel
    except ImportError:
        return None

    # 尝试从 header 获取测站近似坐标
    rec_xyz = station_xyz
    if rec_xyz is None:
        return None

    # 由于没有卫星星历，这里需要外部提供卫星方位角/高度角
    # 返回 None 表示需要定位解算完成后才能计算
    return None


def compute_dop_for_epoch(sat_azel_list):
    """
    根据卫星高度角/方位角列表计算 DOP。

    参数:
        sat_azel_list: [(elev_deg, azim_deg), ...]

    返回:
        {'GDOP': val, 'PDOP': val, 'HDOP': val, 'VDOP': val, 'TDOP': val}
    """
    from satellite_orbit import compute_dop_from_azel
    return compute_dop_from_azel(sat_azel_list)


def compute_sat_elevation(sat_xyz, rec_xyz):
    """计算卫星高度角，返回度。"""
    sat = np.array(sat_xyz, dtype=float)
    rec = np.array(rec_xyz, dtype=float)
    d = sat - rec
    r = np.linalg.norm(d)
    if r == 0:
        return float('nan')

    x, y, z = rec
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    u = cos_lat * cos_lon * d[0] + cos_lat * sin_lon * d[1] + sin_lat * d[2]
    elev = np.arcsin(u / r)
    return float(np.degrees(elev))


def compute_quality_metrics(data: Dict[str, Dict[str, Any]], obs_types: List[str],
                            pos_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """计算课程设计用的综合质量评价指标，可选传入定位结果以纳入定位精度评分。"""
    stats = visibility_stats(data)
    all_sats = sorted(data.keys())
    epochs = sorted(stats.keys()) if stats else []

    total_epochs = len(epochs)
    duration_min = float('nan')
    if len(epochs) >= 2:
        duration_min = (epochs[-1] - epochs[0]).total_seconds() / 60.0

    avg_total = float('nan')
    min_total = float('nan')
    max_total = float('nan')
    if epochs:
        totals = np.array([stats[e]['total'] for e in epochs], dtype=float)
        avg_total = float(np.mean(totals))
        min_total = float(np.min(totals))
        max_total = float(np.max(totals))

    snr_stat = snr_statistics(data, obs_types)

    total_slips = 0
    slip_by_sat = {}
    for sat in all_sats:
        slips, _, _ = cycle_slip_detect(data, obs_types, sat)
        slip_by_sat[sat] = len(slips)
        total_slips += len(slips)

    mp1_values = []
    mp2_values = []
    mp_by_sat = {}
    for sat in all_sats:
        _, mp1, mp2 = multipath_calc(data, obs_types, sat)
        mp1_arr = _finite(mp1)
        mp2_arr = _finite(mp2)
        if len(mp1_arr) > 0:
            mp1_values.extend(mp1_arr)
        if len(mp2_arr) > 0:
            mp2_values.extend(mp2_arr)
        mp_by_sat[sat] = {
            'mp1_rms': _rms(mp1),
            'mp2_rms': _rms(mp2),
            'mp1_std': _std(mp1),
            'mp2_std': _std(mp2),
        }

    mp1_rms = _rms(mp1_values)
    mp2_rms = _rms(mp2_values)

    # 提取定位相关指标
    avg_gdop = float('nan')
    avg_pdop = float('nan')
    pos_std_3d = float('nan')
    if pos_result is not None and pos_result.get("n_epochs", 0) > 0:
        gdop_vals = [g for g in pos_result.get("GDOP", []) if np.isfinite(g)]
        if gdop_vals:
            avg_gdop = float(np.mean(gdop_vals))
        pdop_vals = [p for p in pos_result.get("PDOP", []) if np.isfinite(p)]
        if pdop_vals:
            avg_pdop = float(np.mean(pdop_vals))
        if pos_result.get("std_X") is not None and np.isfinite(pos_result["std_X"]):
            pos_std_3d = float(np.sqrt(pos_result["std_X"]**2 + pos_result["std_Y"]**2 + pos_result["std_Z"]**2))

    # 简洁评分规则，适合课程作业说明
    score = 100
    deductions = []

    if np.isfinite(snr_stat['avg']):
        if snr_stat['avg'] < 30:
            score -= 25
            deductions.append('平均 SNR < 30 dB-Hz，信号质量较差，扣 25 分')
        elif snr_stat['avg'] < 35:
            score -= 15
            deductions.append('平均 SNR < 35 dB-Hz，信号偏弱，扣 15 分')
        elif snr_stat['avg'] < 40:
            score -= 8
            deductions.append('平均 SNR < 40 dB-Hz，信号一般，扣 8 分')
    else:
        score -= 8
        deductions.append('缺少 SNR 观测值，无法评价信噪比，扣 8 分')

    if np.isfinite(mp1_rms):
        if mp1_rms > 1.0:
            score -= 25
            deductions.append('MP1 RMS > 1.0 m，多路径影响明显，扣 25 分')
        elif mp1_rms > 0.5:
            score -= 15
            deductions.append('MP1 RMS > 0.5 m，多路径影响偏大，扣 15 分')
        elif mp1_rms > 0.3:
            score -= 8
            deductions.append('MP1 RMS > 0.3 m，多路径影响一般，扣 8 分')
    else:
        score -= 5
        deductions.append('缺少双频伪距/载波数据，无法计算 MP1，扣 5 分')

    if total_slips > 20:
        score -= 25
        deductions.append('可疑周跳数 > 20，观测连续性较差，扣 25 分')
    elif total_slips > 10:
        score -= 15
        deductions.append('可疑周跳数 > 10，观测连续性一般，扣 15 分')
    elif total_slips > 3:
        score -= 8
        deductions.append('存在多处可疑周跳，扣 8 分')

    if np.isfinite(avg_total):
        if avg_total < 6:
            score -= 20
            deductions.append('平均可见卫星数 < 6，几何条件较差，扣 20 分')
        elif avg_total < 10:
            score -= 8
            deductions.append('平均可见卫星数 < 10，卫星数量一般，扣 8 分')

    # 定位精度评分（若有定位结果）
    if np.isfinite(avg_gdop):
        if avg_gdop > 6:
            score -= 15
            deductions.append(f'平均 GDOP={avg_gdop:.1f} > 6，定位几何条件较差，扣 15 分')
        elif avg_gdop > 4:
            score -= 8
            deductions.append(f'平均 GDOP={avg_gdop:.1f} > 4，定位几何条件一般，扣 8 分')
        elif avg_gdop < 2:
            deductions.append(f'平均 GDOP={avg_gdop:.1f} < 2，定位几何条件优秀 (+5 分奖励)')
            score += 5  # 奖励分

    if np.isfinite(pos_std_3d):
        if pos_std_3d > 10:
            score -= 10
            deductions.append(f'定位标准差 3D={pos_std_3d:.1f} m > 10 m，定位精度较差，扣 10 分')
        elif pos_std_3d > 5:
            score -= 5
            deductions.append(f'定位标准差 3D={pos_std_3d:.1f} m > 5 m，定位精度一般，扣 5 分')

    score = max(0, min(105, int(round(score))))
    if score >= 90:
        grade = 'A'
        quality = '优秀'
        suggestion = '适合静态控制测量与定位解算；观测数据整体稳定，可作为课程设计优质样例。'
    elif score >= 80:
        grade = 'B'
        quality = '良好'
        suggestion = '基本适合静态控制测量与定位解算；建议重点检查少量周跳或多路径较大的卫星弧段。'
    elif score >= 70:
        grade = 'C'
        quality = '一般'
        suggestion = '可用于普通数据分析练习；正式控制测量前建议延长观测或剔除低质量卫星。'
    else:
        grade = 'D'
        quality = '较差'
        suggestion = '不建议直接用于静态控制测量；应检查遮挡、多路径环境、接收机设置和观测时长。'

    return {
        'score': score,
        'grade': grade,
        'quality': quality,
        'suggestion': suggestion,
        'deductions': deductions,
        'total_epochs': total_epochs,
        'duration_min': duration_min,
        'satellite_count': len(all_sats),
        'avg_satellites': avg_total,
        'min_satellites': min_total,
        'max_satellites': max_total,
        'avg_snr': snr_stat['avg'],
        'max_snr': snr_stat['max'],
        'min_snr': snr_stat['min'],
        'snr_count': snr_stat['count'],
        'cycle_slips': total_slips,
        'slip_by_sat': slip_by_sat,
        'mp1_rms': mp1_rms,
        'mp2_rms': mp2_rms,
        'mp_by_sat': mp_by_sat,
        'avg_gdop': avg_gdop,
        'avg_pdop': avg_pdop,
        'pos_std_3d': pos_std_3d,
    }


def quality_evaluation_report(data: Dict[str, Dict[str, Any]], obs_types: List[str],
                               pos_result: Dict[str, Any] | None = None) -> str:
    """生成综合数据质量评价报告。"""
    m = compute_quality_metrics(data, obs_types, pos_result)

    def fmt(value, digits=1, suffix=''):
        try:
            if value is None or not np.isfinite(float(value)):
                return '无有效数据'
        except (TypeError, ValueError):
            return '无有效数据'
        return f"{float(value):.{digits}f}{suffix}"

    lines = [
        '=' * 64,
        '  GNSS 控制网观测数据质量评价',
        '=' * 64,
        '',
        '【核心指标】',
        f"  观测时长:     {fmt(m['duration_min'], 1, ' 分钟')}",
        f"  历元数:       {m['total_epochs']}",
        f"  卫星数量:     {m['satellite_count']} 颗",
        f"  平均卫星数:   {fmt(m['avg_satellites'], 1, ' 颗')}",
        f"  平均 SNR:     {fmt(m['avg_snr'], 1, ' dB-Hz')}",
        f"  最大 SNR:     {fmt(m['max_snr'], 1, ' dB-Hz')}",
        f"  最小 SNR:     {fmt(m['min_snr'], 1, ' dB-Hz')}",
        f"  可疑周跳数:   {m['cycle_slips']} 处",
        f"  MP1 RMS:      {fmt(m['mp1_rms'], 3, ' m')}",
        f"  MP2 RMS:      {fmt(m['mp2_rms'], 3, ' m')}",
        '',
        '【综合评价】',
        f"  综合评分:     {m['score']} / 100",
        f"  质量等级:     {m['grade']} 级",
        f"  数据质量:     {m['quality']}",
        f"  使用建议:     {m['suggestion']}",
        '',
        '【扣分说明】',
    ]

    if m['deductions']:
        for item in m['deductions']:
            lines.append(f'  - {item}')
    else:
        lines.append('  - 未触发明显扣分项')

    lines += [
        '',
        '【评分规则】',
        '  初始分数 100 分。',
        '  平均 SNR < 35 dB-Hz 扣 20 分；MP1 RMS > 0.5 m 扣 20 分；',
        '  可疑周跳数 > 10 扣 20 分；平均卫星数过少按程度扣分。',
        '  90~100: A；80~89: B；70~79: C；<70: D。',
        '=' * 64,
    ]
    return '\n'.join(lines)


def quality_summary(data: Dict[str, Dict[str, Any]], obs_types: List[str],
                     pos_result: Dict[str, Any] | None = None) -> str:
    """生成完整文本报告，含可见性、周跳、多路径、定位与综合评价。"""
    stats = visibility_stats(data)
    all_sats = sorted(data.keys())

    lines = []
    lines.append('=' * 64)
    lines.append('  GNSS 观测数据质量分析报告')
    lines.append('=' * 64)

    if stats:
        epochs = sorted(stats.keys())
        total_epochs = len(epochs)
        avg_gps = np.mean([stats[e]['n_gps'] for e in epochs])
        avg_bds = np.mean([stats[e]['n_bds'] for e in epochs])
        avg_gal = np.mean([stats[e]['n_gal'] for e in epochs])
        avg_glo = np.mean([stats[e]['n_glo'] for e in epochs])
        avg_total = np.mean([stats[e]['total'] for e in epochs])
        duration = (epochs[-1] - epochs[0]).total_seconds() / 60.0 if len(epochs) > 1 else 0

        lines.append(f"\n  总历元数:      {total_epochs}")
        lines.append(f"  观测时长:      {duration:.1f} 分钟")
        lines.append(f"  卫星总数:      {len(all_sats)} 颗")
        lines.append(f"  平均可见 GPS:  {avg_gps:.1f} 颗")
        if avg_bds > 0:
            lines.append(f"  平均可见 BDS:  {avg_bds:.1f} 颗")
        if avg_gal > 0:
            lines.append(f"  平均可见 GAL:  {avg_gal:.1f} 颗")
        if avg_glo > 0:
            lines.append(f"  平均可见 GLO:  {avg_glo:.1f} 颗")
        lines.append(f"  平均可见总数:  {avg_total:.1f} 颗")

    snr_stat = snr_statistics(data, obs_types)
    lines.append(f"\n{'─' * 64}")
    lines.append('  SNR 信噪比统计')
    lines.append(f"{'─' * 64}")
    if snr_stat['count'] > 0:
        lines.append(f"  平均 SNR: {snr_stat['avg']:.1f} dB-Hz")
        lines.append(f"  最大 SNR: {snr_stat['max']:.1f} dB-Hz")
        lines.append(f"  最小 SNR: {snr_stat['min']:.1f} dB-Hz")
    else:
        lines.append('  无 SNR 观测值')

    lines.append(f"\n{'─' * 64}")
    lines.append('  周跳探测结果 (MW+GF)')
    lines.append(f"{'─' * 64}")

    has_slips = False
    for sat in all_sats[:30]:
        slips, _, _ = cycle_slip_detect(data, obs_types, sat)
        if slips:
            has_slips = True
            lines.append(f"  {sat}: 检测到 {len(slips)} 处可疑周跳")
            for idx, t, mw, gf in slips[:5]:
                tstr = t.strftime('%H:%M:%S') if hasattr(t, 'strftime') else str(t)
                mw_str = 'NA' if not np.isfinite(mw) else f'{mw:.2f}周'
                lines.append(f"      {tstr}  MW跳变={mw_str}  GF跳变={gf:.3f}m")

    if not has_slips:
        lines.append('  未检测到明显周跳')

    lines.append(f"\n{'─' * 64}")
    lines.append('  多路径效应统计 (RMS)')
    lines.append(f"{'─' * 64}")

    shown = 0
    for sat in all_sats:
        _, mp1, mp2 = multipath_calc(data, obs_types, sat)
        mp1_r = _rms(mp1)
        mp2_r = _rms(mp2)
        if np.isfinite(mp1_r) or np.isfinite(mp2_r):
            lines.append(f"  {sat}: MP1 RMS={mp1_r:.3f}m  MP2 RMS={mp2_r:.3f}m")
            shown += 1
        if shown >= 15:
            break
    if shown == 0:
        lines.append('  无有效多路径数据')

    # 定位结果（若有）
    if pos_result is not None and pos_result.get("n_epochs", 0) > 0:
        lines.append(f"\n{'─' * 64}")
        lines.append('  定位解算结果 (SPP)')
        lines.append(f"{'─' * 64}")
        lines.append(f"  解算历元数:   {pos_result.get('n_epochs', 0)}")
        lines.append(f"  平均可用卫星: {pos_result.get('mean_nsat', 0):.1f} 颗")
        if np.isfinite(pos_result.get('mean_lat', float('nan'))):
            from coord_convert import deg_to_dms
            lat = pos_result['mean_lat']
            lon = pos_result['mean_lon']
            h = pos_result.get('mean_h', 0)
            ld, lm, ls = deg_to_dms(abs(lat))
            lod, lom, los = deg_to_dms(abs(lon))
            lat_hemi = 'N' if lat >= 0 else 'S'
            lon_hemi = 'E' if lon >= 0 else 'W'
            lines.append(f"  平均坐标:     B={ld}°{lm:02d}'{ls:06.3f}\" {lat_hemi}")
            lines.append(f"                L={lod}°{lom:02d}'{los:06.3f}\" {lon_hemi}")
            lines.append(f"                H={h:.4f} m")
            lines.append(f"                X={pos_result.get('mean_X', 0):.4f} m")
            lines.append(f"                Y={pos_result.get('mean_Y', 0):.4f} m")
            lines.append(f"                Z={pos_result.get('mean_Z', 0):.4f} m")
        gdop_mean = np.nanmean([g for g in pos_result.get("GDOP", []) if np.isfinite(g)])
        pdop_mean = np.nanmean([p for p in pos_result.get("PDOP", []) if np.isfinite(p)])
        if np.isfinite(gdop_mean):
            lines.append(f"  平均 GDOP:     {gdop_mean:.2f}")
            lines.append(f"  平均 PDOP:     {pdop_mean:.2f}")

    lines.append(f"\n{'─' * 64}")
    lines.append('  综合质量评价')
    lines.append(f"{'─' * 64}")
    m = compute_quality_metrics(data, obs_types, pos_result)
    lines.append(f"  综合评分: {m['score']} / 100")
    lines.append(f"  质量等级: {m['grade']} 级")
    lines.append(f"  数据质量: {m['quality']}")
    lines.append(f"  建议: {m['suggestion']}")

    lines.append('\n' + '=' * 64)
    return '\n'.join(lines)
