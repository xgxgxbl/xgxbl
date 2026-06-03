"""
GNSS 观测数据质量分析模块
- MW+GF 周跳探测 (TurboEdit 简化版)
- 测站周跳汇总
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ------------------------------------------------------------
# 频率常数 (MHz)
# ------------------------------------------------------------
FREQ = {
    'G': {'L1': 1575.42, 'L2': 1227.60, 'L5': 1176.45},
    'C': {'B1': 1561.098, 'B2': 1207.140, 'B3': 1268.520, 'L1': 1561.098, 'L2': 1207.140},
    'E': {'E1': 1575.42, 'E5a': 1176.45, 'E5b': 1207.140, 'L1': 1575.42, 'L2': 1207.140},
    'R': {'L1': 1602.00, 'L2': 1246.00},
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


def _select_obs_pair(obs_types: List[str], sat: str) -> Tuple[List[str], List[str]]:
    """
    为不同 RINEX 写法选择近似的 L1/L2 与 C1/C2 观测值。
    兼容常见 RINEX2: L1/L2/C1/P1/C2/P2，以及部分 RINEX3 简写。
    """
    ot = set(obs_types)
    sys = sat[0] if sat else 'G'

    l1_candidates = ['L1', 'L1C', 'L1W', 'L1P', 'L1I', 'L1X']
    l2_candidates = ['L2', 'L2W', 'L2P', 'L2C', 'L2I', 'L2X']
    if sys == 'C':
        l1_candidates = ['L1', 'L1I', 'L1X', 'L2I', 'L2X']
        l2_candidates = ['L2', 'L2I', 'L2X', 'L7I', 'L7X']

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

    wl_lam = C / abs(f1 - f2)
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


def station_slip_summary(data: Dict[str, Dict[str, Any]], obs_types: List[str],
                          station_name: str = '',
                          mw_th: float = 3.0, gf_th: float = 0.15) -> Dict[str, Any]:
    """
    对单个测站的所有卫星执行周跳探测，返回汇总结果。

    返回:
        {
            'station': str,
            'total_slips': int,
            'by_sat': {sat: count, ...},
            'slip_details': {sat: [(epoch, mw_jump, gf_jump), ...], ...},
            'sat_count': int,
            'sat_with_slips': int,
            'slip_epochs': {sat: [epoch_time, ...], ...},
        }
    """
    all_sats = sorted(data.keys())
    total = 0
    by_sat = {}
    slip_details = {}
    slip_epochs = {}

    for sat in all_sats:
        slips, _, _ = cycle_slip_detect(data, obs_types, sat, mw_th, gf_th)
        by_sat[sat] = len(slips)
        total += len(slips)
        if slips:
            slip_details[sat] = [(t, mw, gf) for (_, t, mw, gf) in slips]
            slip_epochs[sat] = [t for (_, t, _, _) in slips]

    sat_with = sum(1 for v in by_sat.values() if v > 0)

    return {
        'station': station_name,
        'total_slips': total,
        'by_sat': by_sat,
        'slip_details': slip_details,
        'sat_count': len(all_sats),
        'sat_with_slips': sat_with,
        'slip_epochs': slip_epochs,
    }
