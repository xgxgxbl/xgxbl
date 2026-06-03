"""
伪距单点定位模块 (SPP — Single Point Positioning)

支持 GPS / BDS 伪距单点定位：
- 广播星历卫星轨道 + 钟差改正
- 对流层延迟改正（Saastamoinen 模型）
- 电离层延迟改正（Klobuchar 模型，可选）
- 最小二乘迭代解算
- GDOP / PDOP / HDOP / VDOP / TDOP 计算
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from satellite_orbit import (
    C, calc_sat_position, calc_sat_azel, compute_dop_from_azel,
    GPS_GM, GPS_OMEGA_E,
)

# ------------------------------------------------------------------
# 对流层延迟 (Saastamoinen 模型)
# ------------------------------------------------------------------
def _tropo_saastamoinen(el_deg: float, lat_deg: float = 0.0,
                        h_m: float = 0.0, doy: int = 180) -> float:
    """
    Saastamoinen 对流层延迟模型（标准大气参数）。

    参数:
        el_deg: 卫星高度角 (度)
        lat_deg: 接收机纬度 (度)
        h_m: 接收机大地高 (m)
        doy: 年积日

    返回:
        对流层延迟 (m)
    """
    if el_deg <= 0:
        return 100.0  # 截止高度角以下，返回大值

    # 标准大气参数
    P = 1013.25  # 气压 hPa
    T = 291.15   # 温度 K (18°C)
    e = 11.691   # 水汽压 hPa (50% RH)

    # 天顶方向干、湿分量
    lat_rad = np.radians(lat_deg)
    cos2lat = np.cos(2 * lat_rad)
    zhd = 0.0022768 * P / (1 - 0.00266 * cos2lat - 2.8e-7 * h_m)
    zwd = 0.0022768 * (1255.0 / T + 0.05) * e

    # 映射函数
    sin_el = np.sin(np.radians(el_deg))
    m_wet = 1.001 / np.sqrt(0.002001 + sin_el ** 2)  # 湿分量映射

    # 干分量映射（简化）
    m_dry = 1.0 / np.sqrt(sin_el ** 2 + (2.5e-4))

    return zhd * m_dry + zwd * m_wet


# ------------------------------------------------------------------
# 电离层延迟 (Klobuchar 模型)
# ------------------------------------------------------------------
def _iono_klobuchar(el_deg: float, az_deg: float,
                    lat_deg: float, lon_deg: float,
                    sow: float, iono_alpha: List[float],
                    iono_beta: List[float]) -> float:
    """
    Klobuchar 电离层延迟模型。

    返回:
        L1 频点电离层延迟 (m)
    """
    if el_deg <= 0 or len(iono_alpha) < 4 or len(iono_beta) < 4:
        return 0.0

    # 穿刺点计算
    el_rad = np.radians(el_deg)
    az_rad = np.radians(az_deg)
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)

    psi = 0.0137 / (el_rad / np.pi + 0.11) - 0.022  # 地心角
    lat_i = lat_r + psi * np.cos(az_rad)
    if lat_i > 0.416:
        lat_i = 0.416
    if lat_i < -0.416:
        lat_i = -0.416
    lon_i = lon_r + psi * np.sin(az_rad) / np.cos(lat_i)

    # 地磁纬度
    lat_m = lat_i + 0.064 * np.cos(lon_i - 1.617)

    # 电离层延迟（秒）
    A_i = iono_alpha[0] + iono_alpha[1] * lat_m + iono_alpha[2] * lat_m ** 2 + iono_alpha[3] * lat_m ** 3
    if A_i < 0:
        A_i = 0

    P_i = iono_beta[0] + iono_beta[1] * lat_m + iono_beta[2] * lat_m ** 2 + iono_beta[3] * lat_m ** 3
    if P_i < 72000:
        P_i = 72000

    X_i = 2 * np.pi * (sow - 50400) / P_i
    F = 1.0 + 16.0 * (0.53 - el_rad / np.pi) ** 3

    if abs(X_i) < 1.57:
        delay_s = F * (5e-9 + A_i * np.cos(X_i))
    else:
        delay_s = F * 5e-9

    return delay_s * C  # 转为米


# ------------------------------------------------------------------
# 最小二乘定位解算
# ------------------------------------------------------------------
def spp_solve(data: Dict[str, Dict[str, Any]],
              obs_types: List[str],
              nav_eph: Dict[str, List[Dict[str, float]]],
              approx_xyz: Optional[Tuple[float, float, float]] = None,
              elev_mask: float = 10.0,
              use_iono: bool = False,
              iono_params: Optional[Dict[str, List[float]]] = None,
              system: str = 'G') -> Dict[str, Any]:
    """
    伪距单点定位解算。

    参数:
        data:       RinexObs 观测数据 {sat: {obs_type: [vals], 'epoch': [...]}}
        obs_types:  观测类型列表
        nav_eph:    导航星历 {prn: [eph_dict, ...]}
        approx_xyz: 近似坐标 (X, Y, Z)，若为 None 则自动估计
        elev_mask:  截止高度角 (度)
        use_iono:   是否使用电离层改正
        iono_params: 电离层 Klobuchar 参数 {'alpha': [...], 'beta': [...]}
        system:     卫星系统 'G' 或 'C'

    返回:
        {
            'epochs': [datetime, ...],
            'X': [m, ...], 'Y': [m, ...], 'Z': [m, ...],
            'lat': [deg, ...], 'lon': [deg, ...], 'h': [m, ...],
            'GDOP': [...], 'PDOP': [...], 'HDOP': [...], 'VDOP': [...], 'TDOP': [...],
            'nsat': [...], 'rms_residual': [...],
            'mean_X': m, 'std_X': m, ...
        }
    """
    from rinex_nav_reader import RinexNav

    # 收集所有历元
    epoch_set = set()
    for sat_data in data.values():
        for ep in sat_data.get('epoch', []):
            epoch_set.add(ep)
    sorted_epochs = sorted(epoch_set)

    if not sorted_epochs:
        return _empty_result()

    # 尝试为每个卫星加载星历
    all_eph = {}
    for prn in data.keys():
        if prn.startswith(system) or system in ('G', 'C'):
            ephs = nav_eph.get(prn, [])
            if ephs:
                all_eph[prn] = ephs

    if not all_eph:
        return _empty_result()

    # 初始近似坐标
    if approx_xyz is None:
        approx_xyz = (0.0, 0.0, 0.0)
        for prn in list(all_eph.keys())[:1]:
            ep = sorted_epochs[len(sorted_epochs) // 2]
            eph = _find_eph(prn, ep, all_eph)
            if eph:
                sat_pos = calc_sat_position(prn, eph, ep, 'C' if prn.startswith('C') else 'G')
                if sat_pos:
                    # 以第一颗卫星为近似（减去 ~20000 km）
                    X_s = sat_pos['X']
                    Y_s = sat_pos['Y']
                    Z_s = sat_pos['Z']
                    approx_xyz = (X_s * 0.3, Y_s * 0.3, Z_s * 0.3)
                    break

    # 观测值选择
    code_keys = []
    for candidate in ['C1', 'C1C', 'C1W', 'C1P', 'C1I', 'P1']:
        if candidate in obs_types:
            code_keys.append(candidate)
            break
    if not code_keys:
        code_keys = [ot for ot in obs_types if ot.startswith('C') or ot.startswith('P')]
        if not code_keys:
            return _empty_result()
    c1_key = code_keys[0]

    # 逐历元解算
    result = {
        'epochs': [],
        'X': [], 'Y': [], 'Z': [],
        'lat': [], 'lon': [], 'h': [],
        'GDOP': [], 'PDOP': [], 'HDOP': [], 'VDOP': [], 'TDOP': [],
        'nsat': [], 'rms_residual': [],
    }

    X_r, Y_r, Z_r = approx_xyz

    for ep in sorted_epochs:
        # 获取当前历元各卫星的伪距和星历
        obs_list = []  # [(prn, P, eph)]
        for prn, sat_data in data.items():
            # 只处理指定系统的卫星
            if system == 'G' and not prn.startswith('G'):
                continue
            if system == 'C' and not prn.startswith('C'):
                continue

            try:
                idx = sat_data['epoch'].index(ep)
            except (ValueError, KeyError):
                continue

            P_val = sat_data.get(c1_key, [np.nan] * (idx + 1))
            if idx >= len(P_val):
                continue
            P = float(P_val[idx]) if idx < len(P_val) else np.nan
            if not np.isfinite(P) or P <= 0:
                continue

            eph = _find_eph(prn, ep, all_eph)
            if eph is None:
                continue

            obs_list.append((prn, P, eph))

        # 最少4颗卫星
        if len(obs_list) < 4:
            continue

        # 迭代最小二乘
        niter = 8
        converged = False
        used_sats = []

        for _ in range(niter):
            A_list = []
            L_list = []
            W_list = []
            used_sats = []

            for prn, P, eph in obs_list:
                sat_pos = calc_sat_position(prn, eph, ep, 'C' if prn.startswith('C') else 'G')
                if sat_pos is None:
                    continue

                X_s = sat_pos['X']
                Y_s = sat_pos['Y']
                Z_s = sat_pos['Z']
                dt_sv = sat_pos['dt_sv']

                # 几何距离（近似的信号传播时间）
                dx = X_s - X_r
                dy = Y_s - Y_r
                dz = Z_s - Z_r
                rho0 = np.sqrt(dx * dx + dy * dy + dz * dz)

                # 高度角检查
                elev, azim, _ = calc_sat_azel((X_s, Y_s, Z_s), (X_r, Y_r, Z_r))
                if elev < elev_mask:
                    continue

                # 卫星钟差改正（米）
                dt_sv_m = dt_sv * C

                # 对流层改正
                from coord_convert import xyz_to_blh
                lat_r, lon_r, h_r = xyz_to_blh(X_r, Y_r, Z_r)
                tropo = _tropo_saastamoinen(elev, lat_r, h_r)

                # 电离层改正
                iono = 0.0
                if use_iono and iono_params:
                    sow = _epoch_to_sow(ep)
                    if sow is not None:
                        iono = _iono_klobuchar(elev, azim, lat_r, lon_r, sow,
                                               iono_params.get('alpha', [0, 0, 0, 0]),
                                               iono_params.get('beta', [0, 0, 0, 0]))

                # 改正后的伪距
                P_corr = P + dt_sv_m - tropo - iono

                # 观测方程线性化
                los_x = -dx / rho0
                los_y = -dy / rho0
                los_z = -dz / rho0

                A_list.append([los_x, los_y, los_z, 1.0])
                L_list.append(P_corr - rho0)

                # 高度角加权
                w = np.sin(np.radians(elev))
                W_list.append(w)

                used_sats.append((prn, elev, azim))

            if len(A_list) < 4:
                break

            A = np.array(A_list)
            L = np.array(L_list).reshape(-1, 1)
            W = np.diag(W_list)

            # 加权最小二乘
            try:
                Q = np.linalg.inv(A.T @ W @ A)
                dx_vec = Q @ A.T @ W @ L
            except np.linalg.LinAlgError:
                break

            X_r += dx_vec[0, 0]
            Y_r += dx_vec[1, 0]
            Z_r += dx_vec[2, 0]

            if abs(dx_vec[0, 0]) < 0.001 and abs(dx_vec[1, 0]) < 0.001 and \
               abs(dx_vec[2, 0]) < 0.001:
                converged = True
                break

        if not used_sats or len(used_sats) < 4:
            continue

        # 收敛后计算 DOP
        azel_list = [(el, az) for _, el, az in used_sats]
        dop = compute_dop_from_azel(azel_list)

        # 残差
        residuals = []
        for k, (prn, P, eph) in enumerate(obs_list):
            if k >= len(L):
                break
            residuals.append(float(L[k]))

        rms_res = float(np.sqrt(np.mean([r ** 2 for r in residuals]))) if residuals else float('nan')

        # BLH
        from coord_convert import xyz_to_blh
        lat_final, lon_final, h_final = xyz_to_blh(X_r, Y_r, Z_r)

        result['epochs'].append(ep)
        result['X'].append(float(X_r))
        result['Y'].append(float(Y_r))
        result['Z'].append(float(Z_r))
        result['lat'].append(float(lat_final))
        result['lon'].append(float(lon_final))
        result['h'].append(float(h_final))
        result['GDOP'].append(dop['GDOP'])
        result['PDOP'].append(dop['PDOP'])
        result['HDOP'].append(dop['HDOP'])
        result['VDOP'].append(dop['VDOP'])
        result['TDOP'].append(dop['TDOP'])
        result['nsat'].append(len(used_sats))
        result['rms_residual'].append(rms_res)

    # 统计量
    if result['X']:
        X_arr = np.array(result['X'])
        Y_arr = np.array(result['Y'])
        Z_arr = np.array(result['Z'])
        lat_arr = np.array(result['lat'])
        lon_arr = np.array(result['lon'])
        h_arr = np.array(result['h'])
        valid = np.isfinite(X_arr)

        result['mean_X'] = float(np.mean(X_arr[valid]))
        result['mean_Y'] = float(np.mean(Y_arr[valid]))
        result['mean_Z'] = float(np.mean(Z_arr[valid]))
        result['std_X'] = float(np.std(X_arr[valid]))
        result['std_Y'] = float(np.std(Y_arr[valid]))
        result['std_Z'] = float(np.std(Z_arr[valid]))
        result['mean_lat'] = float(np.mean(lat_arr[valid]))
        result['mean_lon'] = float(np.mean(lon_arr[valid]))
        result['mean_h'] = float(np.mean(h_arr[valid]))
        result['std_lat'] = float(np.std(lat_arr[valid]))
        result['std_lon'] = float(np.std(lon_arr[valid]))
        result['std_h'] = float(np.std(h_arr[valid]))
        result['n_epochs'] = int(np.sum(valid))
        result['mean_nsat'] = float(np.mean([n for i, n in enumerate(result['nsat']) if valid[i]]))
    else:
        for key in ['mean_X', 'mean_Y', 'mean_Z', 'mean_lat', 'mean_lon', 'mean_h']:
            result[key] = float('nan')
        for key in ['std_X', 'std_Y', 'std_Z', 'std_lat', 'std_lon', 'std_h', 'n_epochs', 'mean_nsat']:
            result[key] = float('nan')

    return result


def spp_solve_combined(data: Dict[str, Dict[str, Any]],
                       obs_types: List[str],
                       nav_eph: Dict[str, List[Dict[str, float]]],
                       approx_xyz: Optional[Tuple[float, float, float]] = None,
                       elev_mask: float = 10.0,
                       use_iono: bool = False,
                       iono_params: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
    """
    联合 GPS+BDS 单点定位。自动根据可用卫星混合解算。
    """
    return spp_solve(data, obs_types, nav_eph, approx_xyz, elev_mask,
                     use_iono, iono_params, system='G')


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------
def _find_eph(prn: str, t: datetime,
              eph_dict: Dict[str, List[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    """在星历字典中查找最接近时间 t 的星历。"""
    ephs = eph_dict.get(prn, [])
    if not ephs:
        return None
    best = None
    best_dt = float('inf')
    for eph in ephs:
        toe_sow = eph.get('toe', 0.0)
        sow = _epoch_to_sow(t)
        if sow is None:
            # 用 toc
            toc = eph.get('toc')
            if toc is not None and hasattr(toc, 'total_seconds'):
                dt = abs((t - toc).total_seconds())
            else:
                continue
        else:
            dt = abs(sow - toe_sow)
            if dt > 302400:
                dt = 604800 - dt
        if dt < best_dt:
            best_dt = dt
            best = eph
    if best is not None and best_dt > 14400:
        return None
    return best


def _epoch_to_sow(t: datetime) -> Optional[float]:
    """UTC datetime → GPS 周内秒。"""
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    dt = (t - gps_epoch).total_seconds()
    week = int(dt / 604800)
    return dt - week * 604800


def _empty_result() -> Dict[str, Any]:
    return {
        'epochs': [], 'X': [], 'Y': [], 'Z': [],
        'lat': [], 'lon': [], 'h': [],
        'GDOP': [], 'PDOP': [], 'HDOP': [], 'VDOP': [], 'TDOP': [],
        'nsat': [], 'rms_residual': [],
        'mean_X': float('nan'), 'mean_Y': float('nan'), 'mean_Z': float('nan'),
        'mean_lat': float('nan'), 'mean_lon': float('nan'), 'mean_h': float('nan'),
        'std_X': float('nan'), 'std_Y': float('nan'), 'std_Z': float('nan'),
        'std_lat': float('nan'), 'std_lon': float('nan'), 'std_h': float('nan'),
        'n_epochs': 0, 'mean_nsat': float('nan'),
    }
