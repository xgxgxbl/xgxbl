"""
卫星轨道计算模块
基于广播星历计算卫星位置 (GPS ICD-200 / BDS ICD 标准算法)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# 物理常数
# ------------------------------------------------------------------
C = 299792458.0                     # 光速 m/s

# GPS (WGS-84)
GPS_GM = 3.986005e14                # 地球引力常数 m³/s²
GPS_OMEGA_E = 7.2921151467e-5       # 地球自转角速度 rad/s

# BDS (CGCS2000)
BDS_GM = 3.986004418e14
BDS_OMEGA_E = 7.2921150e-5

# 地球半径（赤道）
EARTH_R = 6378137.0


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def calc_sat_position(prn: str, eph: Dict[str, float], t: datetime,
                      system: str = 'G') -> Optional[Dict[str, Any]]:
    """
    计算卫星在时刻 t 的 ECEF 坐标和钟差。

    参数:
        prn:  卫星 PRN，如 'G01'、'C01'
        eph:  广播星历参数字典
        t:    观测时刻 (UTC datetime)
        system: 'G' 或 'C'

    返回:
        {'X': m, 'Y': m, 'Z': m, 'dt_sv': s, 'vX': m/s, 'vY': m/s, 'vZ': m/s}
        若星历无效则返回 None。
    """
    if system.startswith('C'):
        return _calc_bds_position(prn, eph, t)
    return _calc_gps_position(prn, eph, t)


def calc_sat_clock_correction(eph: Dict[str, float], t: float) -> float:
    """
    卫星钟差改正（秒）。

    参数:
        eph: 广播星历字典
        t:   信号发射时刻 (GPS seconds of week 或 BDS seconds of week)
            实际上是 (观测时刻 - 近似传播时间) 对应的周内秒

    返回:
        钟差改正（秒），可直接乘 C 转为米
    """
    toc = eph.get('toc', 0.0)  # 星历参考时刻 (sow)
    dt = t - toc
    # 处理周边界
    if abs(dt) > 302400:
        if dt > 0:
            dt -= 604800
        else:
            dt += 604800
    a0 = eph.get('sv_clock_bias', 0.0)
    a1 = eph.get('sv_clock_drift', 0.0)
    a2 = eph.get('sv_clock_drift_rate', 0.0)
    # 相对论修正
    ecc = eph.get('e', 0.0)
    sqrt_a = eph.get('sqrt_a', 0.0)
    rel = 0.0
    if sqrt_a > 0:
        n0 = np.sqrt(GPS_GM / (sqrt_a ** 6))
        M = eph.get('m0', 0.0) + eph.get('delta_n', 0.0)
        Ek = _solve_kepler(M + n0 * dt, ecc)
        rel = -4.442807633e-10 * ecc * sqrt_a * np.sin(Ek)
    return a0 + a1 * dt + a2 * dt * dt + rel


def calc_sat_azel(sat_xyz: Tuple[float, float, float],
                  rec_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    计算卫星高度角、方位角和距离。

    返回:
        (elev_deg, azim_deg, range_m)
    """
    sat = np.array(sat_xyz, dtype=float)
    rec = np.array(rec_xyz, dtype=float)
    d = sat - rec
    rng = np.linalg.norm(d)

    # 接收机大地坐标
    x, y, z = rec
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

    # ENU 变换矩阵
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # 站心坐标系向量
    e = -sin_lon * d[0] + cos_lon * d[1]
    n = -sin_lat * cos_lon * d[0] - sin_lat * sin_lon * d[1] + cos_lat * d[2]
    u = cos_lat * cos_lon * d[0] + cos_lat * sin_lon * d[1] + sin_lat * d[2]

    elev = np.arcsin(u / rng)
    azim = np.arctan2(e, n)
    if azim < 0:
        azim += 2 * np.pi

    return float(np.degrees(elev)), float(np.degrees(azim)), float(rng)


# ------------------------------------------------------------------
# GPS 轨道计算 (ICD-GPS-200)
# ------------------------------------------------------------------
def _calc_gps_position(prn: str, eph: Dict[str, float], t: datetime) -> Optional[Dict[str, Any]]:
    """GPS 卫星 ECEF 坐标计算。"""
    GM = GPS_GM
    OMEGA_E = GPS_OMEGA_E

    sqrt_a = eph.get('sqrt_a', 0.0)
    if sqrt_a <= 0:
        return None

    A = sqrt_a ** 2
    n0 = np.sqrt(GM / A ** 3)

    # 信号发射时刻
    toe = eph.get('toe', 0.0)
    if toe <= 0:
        return None

    # 观测时间的 GPS 周内秒
    gps_week = int(eph.get('gps_week', 0))
    t_sow = _datetime_to_gps_sow(t, gps_week)
    if t_sow is None:
        return None

    # 信号传播时间初始估计
    travel_time = 0.075  # ~22500 km / c
    t_transmit = t_sow - travel_time

    # 迭代 2 次
    for _ in range(2):
        sat_result = _compute_sat_pos_gps(eph, t_transmit, A, n0, toe, GM, OMEGA_E)
        if sat_result is None:
            return None
        dist = np.sqrt((sat_result['X']) ** 2 + (sat_result['Y']) ** 2 + (sat_result['Z']) ** 2)
        travel_time = dist / C
        t_transmit = t_sow - travel_time

    sat_result = _compute_sat_pos_gps(eph, t_transmit, A, n0, toe, GM, OMEGA_E)
    if sat_result is None:
        return None

    # 地球自转改正 (Sagnac)
    Rot = travel_time * OMEGA_E
    X = sat_result['X'] * np.cos(Rot) + sat_result['Y'] * np.sin(Rot)
    Y = -sat_result['X'] * np.sin(Rot) + sat_result['Y'] * np.cos(Rot)
    Z = sat_result['Z']

    # 卫星钟差
    dt_sv = calc_sat_clock_correction(eph, t_transmit)

    return {
        'X': float(X), 'Y': float(Y), 'Z': float(Z),
        'dt_sv': float(dt_sv),
        'vX': float(sat_result.get('vX', 0)),
        'vY': float(sat_result.get('vY', 0)),
        'vZ': float(sat_result.get('vZ', 0)),
    }


def _compute_sat_pos_gps(eph: Dict[str, float], t: float, A: float, n0: float,
                          toe: float, GM: float, OMEGA_E: float) -> Optional[Dict[str, float]]:
    """GPS 轨道位置计算核心。"""
    delta_n = eph.get('delta_n', 0.0)
    m0 = eph.get('m0', 0.0)
    e = eph.get('e', 0.0)
    omega = eph.get('omega', 0.0)
    cuc = eph.get('cuc', 0.0)
    cus = eph.get('cus', 0.0)
    crs = eph.get('crs', 0.0)
    crc = eph.get('crc', 0.0)
    cic = eph.get('cic', 0.0)
    cis = eph.get('cis', 0.0)
    i0 = eph.get('i0', 0.0)
    idot = eph.get('idot', 0.0)
    omega0 = eph.get('omega0', 0.0)
    omega_dot = eph.get('omega_dot', 0.0)

    tk = t - toe
    # 处理周边界 (±302400 秒)
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800

    n = n0 + delta_n
    Mk = m0 + n * tk
    Ek = _solve_kepler(Mk, e)

    sin_Ek = np.sin(Ek)
    cos_Ek = np.cos(Ek)

    # 真近点角
    nu_k = np.arctan2(np.sqrt(1 - e ** 2) * sin_Ek, cos_Ek - e)

    # 升交角距
    Phi_k = nu_k + omega

    # 二阶调和改正
    sin_2Phi = np.sin(2 * Phi_k)
    cos_2Phi = np.cos(2 * Phi_k)

    du_k = cus * sin_2Phi + cuc * cos_2Phi
    dr_k = crs * sin_2Phi + crc * cos_2Phi
    di_k = cis * sin_2Phi + cic * cos_2Phi

    u_k = Phi_k + du_k
    r_k = A * (1 - e * cos_Ek) + dr_k
    i_k = i0 + di_k + idot * tk

    # 轨道平面坐标
    x_k = r_k * np.cos(u_k)
    y_k = r_k * np.sin(u_k)

    # 升交点经度
    Omega_k = omega0 + (omega_dot - OMEGA_E) * tk - OMEGA_E * toe

    cos_Omega = np.cos(Omega_k)
    sin_Omega = np.sin(Omega_k)
    cos_i = np.cos(i_k)
    sin_i = np.sin(i_k)

    # ECEF 坐标
    X = x_k * cos_Omega - y_k * cos_i * sin_Omega
    Y = x_k * sin_Omega + y_k * cos_i * cos_Omega
    Z = y_k * sin_i

    return {'X': X, 'Y': Y, 'Z': Z, 'vX': 0.0, 'vY': 0.0, 'vZ': 0.0}


# ------------------------------------------------------------------
# BDS 轨道计算 (BDS ICD)
# ------------------------------------------------------------------
def _calc_bds_position(prn: str, eph: Dict[str, float], t: datetime) -> Optional[Dict[str, Any]]:
    """BDS 卫星 ECEF 坐标计算。"""
    GM = BDS_GM
    OMEGA_E = BDS_OMEGA_E

    sqrt_a = eph.get('sqrt_a', 0.0)
    if sqrt_a <= 0:
        return None

    A = sqrt_a ** 2
    n0 = np.sqrt(GM / A ** 3)

    toe = eph.get('toe', 0.0)
    if toe <= 0:
        return None

    # BDT 周内秒
    sow = _datetime_to_bds_sow(t)
    if sow is None:
        return None

    travel_time = 0.075
    t_transmit = sow - travel_time

    for _ in range(2):
        sat_result = _compute_sat_pos_gps(eph, t_transmit, A, n0, toe, GM, OMEGA_E)
        if sat_result is None:
            return None
        dist = np.sqrt(sat_result['X'] ** 2 + sat_result['Y'] ** 2 + sat_result['Z'] ** 2)
        travel_time = dist / C
        t_transmit = sow - travel_time

    sat_result = _compute_sat_pos_gps(eph, t_transmit, A, n0, toe, GM, OMEGA_E)
    if sat_result is None:
        return None

    # 地球自转改正
    Rot = travel_time * OMEGA_E
    X = sat_result['X'] * np.cos(Rot) + sat_result['Y'] * np.sin(Rot)
    Y = -sat_result['X'] * np.sin(Rot) + sat_result['Y'] * np.cos(Rot)
    Z = sat_result['Z']

    # BDS GEO 卫星特殊处理 (PRN C01-C05)
    prn_num = int(prn[1:]) if len(prn) > 1 else 0
    if prn_num <= 5:
        X, Y, Z = _bds_geo_rotation(X, Y, Z, OMEGA_E, t_transmit)

    dt_sv = calc_sat_clock_correction(eph, t_transmit)

    return {
        'X': float(X), 'Y': float(Y), 'Z': float(Z),
        'dt_sv': float(dt_sv),
        'vX': 0.0, 'vY': 0.0, 'vZ': 0.0,
    }


def _bds_geo_rotation(X: float, Y: float, Z: float, OMEGA_E: float,
                      tk: float) -> Tuple[float, float, float]:
    """BDS GEO 卫星坐标系旋转 (CGCS2000 惯性系 → 地固系)。"""
    omega_tk = OMEGA_E * tk
    cos_w = np.cos(omega_tk)
    sin_w = np.sin(omega_tk)
    # 绕 Z 轴旋转 -5°
    angle = np.radians(-5.0)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    X_rot = X * cos_w + Y * sin_w
    Y_rot = -X * sin_w + Y * cos_w
    Z_rot = Z

    X_final = X_rot * cos_a + Y_rot * sin_a
    Y_final = -X_rot * sin_a + Y_rot * cos_a
    Z_final = Z_rot

    return X_final, Y_final, Z_final


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------
def _solve_kepler(M: float, e: float, max_iter: int = 10) -> float:
    """迭代求解开普勒方程 E - e*sin(E) = M。"""
    E = M
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E


def _datetime_to_gps_sow(t: datetime, ref_week: int) -> Optional[float]:
    """UTC datetime → GPS 周内秒。"""
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    dt = (t - gps_epoch).total_seconds()
    week = int(dt / 604800)
    sow = dt - week * 604800
    # 确保在 ref_week 附近
    diff_week = week - ref_week
    if abs(diff_week) <= 1:
        return sow + diff_week * 604800
    return sow  # 返回原始值


def _datetime_to_bds_sow(t: datetime) -> Optional[float]:
    """UTC datetime → BDS 周内秒。"""
    bds_epoch = datetime(2006, 1, 1, 0, 0, 0)
    dt = (t - bds_epoch).total_seconds()
    week = int(dt / 604800)
    sow = dt - week * 604800
    return sow


def compute_dop_from_azel(sat_azel_list: list) -> Dict[str, float]:
    """
    从卫星方位角/高度角列表计算 DOP 值。

    参数:
        sat_azel_list: [(elev_deg, azim_deg), ...]

    返回:
        {'GDOP': val, 'PDOP': val, 'HDOP': val, 'VDOP': val, 'TDOP': val}
    """
    n = len(sat_azel_list)
    if n < 4:
        return {'GDOP': float('nan'), 'PDOP': float('nan'),
                'HDOP': float('nan'), 'VDOP': float('nan'), 'TDOP': float('nan')}

    A = np.zeros((n, 4))
    for i, (elev, azim) in enumerate(sat_azel_list):
        el_rad = np.radians(elev)
        az_rad = np.radians(azim)
        cos_el = np.cos(el_rad)
        sin_el = np.sin(el_rad)
        A[i, 0] = cos_el * np.cos(az_rad)
        A[i, 1] = cos_el * np.sin(az_rad)
        A[i, 2] = sin_el
        A[i, 3] = 1.0

    try:
        Q = np.linalg.inv(A.T @ A)
        gdop = np.sqrt(np.trace(Q))
        pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
        hdop = np.sqrt(Q[0, 0] + Q[1, 1])
        vdop = np.sqrt(Q[2, 2])
        tdop = np.sqrt(Q[3, 3])
        return {
            'GDOP': float(gdop), 'PDOP': float(pdop),
            'HDOP': float(hdop), 'VDOP': float(vdop), 'TDOP': float(tdop),
        }
    except np.linalg.LinAlgError:
        return {'GDOP': float('nan'), 'PDOP': float('nan'),
                'HDOP': float('nan'), 'VDOP': float('nan'), 'TDOP': float('nan')}
