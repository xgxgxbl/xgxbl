"""
坐标转换模块
- XYZ ↔ BLH (空间直角坐标 ↔ 大地坐标)
- 高斯-克吕格投影
- UTM 投影
"""

import numpy as np

# WGS84 椭球参数
WGS84 = {
    'a': 6378137.0,
    'f': 1.0 / 298.257223563,
    'GM': 3.986005e14,
    'omega': 7.292115e-5,
}
WGS84['b'] = WGS84['a'] * (1 - WGS84['f'])
WGS84['e2'] = 2 * WGS84['f'] - WGS84['f']**2
WGS84['e2_'] = WGS84['e2'] / (1 - WGS84['e2'])


# CGCS2000 椭球参数（与WGS84非常接近，a相同，f略有差异）
CGCS2000 = {
    'a': 6378137.0,
    'f': 1.0 / 298.257222101,
}
CGCS2000['b'] = CGCS2000['a'] * (1 - CGCS2000['f'])
CGCS2000['e2'] = 2 * CGCS2000['f'] - CGCS2000['f']**2
CGCS2000['e2_'] = CGCS2000['e2'] / (1 - CGCS2000['e2'])


def blh_to_xyz(lat, lon, h, ellipsoid=WGS84):
    """
    大地坐标 (B, L, H) → 空间直角坐标 (X, Y, Z)

    输入:
        lat, lon: 度（十进制）
        h: 大地高（米）
        ellipsoid: 椭球参数字典

    返回:
        (X, Y, Z) 米
    """
    a = ellipsoid['a']
    e2 = ellipsoid['e2']

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    X = (N + h) * cos_lat * np.cos(lon_rad)
    Y = (N + h) * cos_lat * np.sin(lon_rad)
    Z = (N * (1 - e2) + h) * sin_lat

    return X, Y, Z


def xyz_to_blh(X, Y, Z, ellipsoid=WGS84, max_iter=10):
    """
    空间直角坐标 (X, Y, Z) → 大地坐标 (B, L, H)

    返回:
        (lat, lon, h) 度, 度, 米
    """
    a = ellipsoid['a']
    e2 = ellipsoid['e2']

    lon = np.degrees(np.arctan2(Y, X))

    # 迭代法
    p = np.sqrt(X**2 + Y**2)
    lat = np.arctan2(Z, p * (1 - e2))  # 初始值

    for _ in range(max_iter):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        lat_new = np.arctan2(Z + N * e2 * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    h = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), h


def gauss_projection(lat, lon, central_meridian, ellipsoid=WGS84):
    """
    高斯-克吕格投影（3° 或 6° 带）

    输入:
        lat, lon: 度
        central_meridian: 中央子午线（度）

    返回:
        (x, y) 米, 以及带号
    """
    a = ellipsoid['a']
    e2 = ellipsoid['e2']
    e2_ = ellipsoid['e2_']

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lon0_rad = np.radians(central_meridian)

    dl = lon_rad - lon0_rad

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    tan_lat = np.tan(lat_rad)

    N = a / np.sqrt(1 - e2 * sin_lat**2)
    eta2 = e2_ * cos_lat**2

    # 子午线弧长（近似）
    A0 = 1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256
    A1 = 3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024
    A2 = 15 * e2**2 / 256 + 45 * e2**3 / 1024
    A3 = 35 * e2**3 / 3072

    X0 = a * (A0 * lat_rad - A1 * np.sin(2 * lat_rad) +
              A2 * np.sin(4 * lat_rad) - A3 * np.sin(6 * lat_rad))

    t = tan_lat
    c = eta2

    x = X0 + N * t * cos_lat**2 * dl**2 / 2 + \
        N * t * (5 - t**2 + 9 * c + 4 * c**2) * cos_lat**4 * dl**4 / 24 + \
        N * t * (61 - 58 * t**2 + t**4 + 270 * c - 330 * t**2 * c) * \
        cos_lat**6 * dl**6 / 720

    y = N * cos_lat * dl + \
        N * (1 - t**2 + c) * cos_lat**3 * dl**3 / 6 + \
        N * (5 - 18 * t**2 + t**4 + 14 * c - 58 * t**2 * c) * \
        cos_lat**5 * dl**5 / 120

    y += 500000  # 东偏移

    # 带号: 3度带 = round((lon - 1.5) / 3), 6度带 = round(lon / 6)
    zone_6 = int(np.ceil((lon + 180) / 6))

    return x, y, zone_6


def gauss_inverse(x, y, central_meridian, ellipsoid=WGS84, max_iter=10):
    """
    高斯-克吕格投影反算 → (lat, lon, h=0)
    """
    a = ellipsoid['a']
    e2 = ellipsoid['e2']
    e2_ = ellipsoid['e2_']

    y -= 500000

    # 底点纬度（迭代）
    A0 = 1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256
    A1 = 3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024
    A2 = 15 * e2**2 / 256 + 45 * e2**3 / 1024
    A3 = 35 * e2**3 / 3072

    Bf = x / a / A0  # 初始值
    for _ in range(max_iter):
        dB = -(A0 * Bf - A1 * np.sin(2 * Bf) + A2 * np.sin(4 * Bf) -
               A3 * np.sin(6 * Bf) - x / a) / A0
        Bf += dB
        if abs(dB) < 1e-12:
            break

    sin_Bf = np.sin(Bf)
    cos_Bf = np.cos(Bf)
    tan_Bf = np.tan(Bf)
    Nf = a / np.sqrt(1 - e2 * sin_Bf**2)
    eta2f = e2_ * cos_Bf**2

    lat_rad = Bf - Nf * tan_Bf * (y / Nf)**2 / 2 + \
        Nf * tan_Bf * (5 + 3 * tan_Bf**2 + eta2f - 9 * eta2f * tan_Bf**2) * \
        (y / Nf)**4 / 24 - \
        Nf * tan_Bf * (61 + 90 * tan_Bf**2 + 45 * tan_Bf**4) * (y / Nf)**6 / 720

    lon_rad = np.radians(central_meridian) + \
        (y / Nf) / cos_Bf - \
        (1 + 2 * tan_Bf**2 + eta2f) * (y / Nf)**3 / (6 * cos_Bf) + \
        (5 + 28 * tan_Bf**2 + 24 * tan_Bf**4 + 6 * eta2f + 8 * eta2f * tan_Bf**2) * \
        (y / Nf)**5 / (120 * cos_Bf)

    return np.degrees(lat_rad), np.degrees(lon_rad), 0


def dms_to_deg(d, m, s):
    """度分秒 → 十进制度"""
    return d + m / 60.0 + s / 3600.0


def deg_to_dms(deg):
    """十进制度 → 度分秒"""
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    if abs(s - 60) < 1e-8:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d += 1
    return d, m, s


def convert_summary(lat, lon, h=0, ellipsoid_name='WGS84'):
    """生成坐标转换汇总"""
    ell = WGS84 if ellipsoid_name.upper() == 'WGS84' else CGCS2000

    X, Y, Z = blh_to_xyz(lat, lon, h, ell)
    lat_c, lon_c, h_c = xyz_to_blh(X, Y, Z, ell)

    # 3度带高斯投影
    cm_3 = round(lon / 3) * 3
    x3, y3, z3 = gauss_projection(lat, lon, cm_3, ell)

    # 6度带高斯投影
    zone_6 = int(np.ceil((lon + 180) / 6))
    cm_6 = zone_6 * 6 - 3
    x6, y6, z6_g = gauss_projection(lat, lon, cm_6, ell)

    d, m, s = deg_to_dms(abs(lat))
    lat_hemi = 'N' if lat >= 0 else 'S'
    ld, lm, ls = deg_to_dms(abs(lon))
    lon_hemi = 'E' if lon >= 0 else 'W'

    lines = [
        "=" * 60,
        f"  坐标转换结果 ({ellipsoid_name})",
        "=" * 60,
        "",
        "── 大地坐标 ──",
        f"  B (纬度): {d}°{m:02d}′{s:06.3f}″ {lat_hemi}  = {lat:.8f}°",
        f"  L (经度): {ld}°{lm:02d}′{ls:06.3f}″ {lon_hemi}  = {lon:.8f}°",
        f"  H (大地高): {h:.4f} m",
        "",
        "── 空间直角坐标 ──",
        f"  X: {X:15.4f} m",
        f"  Y: {Y:15.4f} m",
        f"  Z: {Z:15.4f} m",
        "",
        "── 高斯-克吕格投影 (3°带) ──",
        f"  中央子午线: {cm_3}°",
        f"  x (北): {x3:12.4f} m",
        f"  y (东): {y3:12.4f} m",
        "",
        "── 高斯-克吕格投影 (6°带) ──",
        f"  带号: {zone_6}",
        f"  中央子午线: {cm_6}°",
        f"  x (北): {x6:12.4f} m",
        f"  y (东): {y6:12.4f} m",
        "",
        "── 反算验证 ──",
        f"  B: {lat_c:.8f}°  (Δ = {abs(lat_c-lat)*3600:.6f}″)",
        f"  L: {lon_c:.8f}°  (Δ = {abs(lon_c-lon)*3600:.6f}″)",
        f"  H: {h_c:.4f} m  (Δ = {abs(h_c-h):.4f} m)",
        "=" * 60,
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    # 测试：北京附近坐标
    print(convert_summary(39.9042, 116.4074, 50, 'WGS84'))
