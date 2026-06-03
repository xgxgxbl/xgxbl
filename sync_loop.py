"""
同步环检核模块 — GNSS 控制网数据质量检查核心

功能:
1. 解析基线处理成果文件（极简文本格式）
2. 构建测站无向图
3. 枚举三角形同步环
4. 计算各环闭合差（ΔX, ΔY, ΔZ, 3D）
5. 与限差比较，判定合格/不合格
6. 生成检核报告与汇总统计
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_baseline_text(text: str) -> Tuple[Dict, set]:
    """
    解析基线成果文本。

    支持格式（空格或制表符分隔）:
        起点名 终点名 dX(m) dY(m) dZ(m) [σX(mm)] [σY(mm)] [σZ(mm)]

    忽略空行和 # 开头的注释行。

    返回:
        baselines: {(A, B): (dX, dY, dZ, sigma_mm_or_None), ...}
        stations: 测站名集合
    """
    baselines = {}
    stations = set()

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        sta_a, sta_b = parts[0], parts[1]
        try:
            dx = float(parts[2])
            dy = float(parts[3])
            dz = float(parts[4])
        except ValueError:
            continue

        sigma = float(parts[5]) if len(parts) >= 6 else None

        stations.add(sta_a)
        stations.add(sta_b)
        # 双向存储
        baselines[(sta_a, sta_b)] = (dx, dy, dz, sigma)
        baselines[(sta_b, sta_a)] = (-dx, -dy, -dz, sigma)

    return baselines, stations


class SyncLoopChecker:
    """
    同步环检核器。

    使用示例:
        checker = SyncLoopChecker(baseline_text)
        summary = checker.summary()
        for r in summary['results']:
            print(r['loop'], r['closure_3d_mm'], '合格' if r['pass'] else '不合格')
    """

    def __init__(self, baseline_text: str):
        self.baselines, self.stations = _parse_baseline_text(baseline_text)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_baselines(self) -> int:
        """唯一基线数（不重复计算方向）"""
        return len(self.baselines) // 2

    def find_sync_loops(self) -> List[Tuple[str, str, str]]:
        """
        枚举所有三角形同步环（3个测站、3条基线均存在的环）。

        返回: [(sta1, sta2, sta3), ...]
        """
        stations = sorted(self.stations)
        loops = []
        n = len(stations)
        for i in range(n):
            for j in range(i + 1, n):
                if (stations[i], stations[j]) not in self.baselines:
                    continue
                for k in range(j + 1, n):
                    if ((stations[i], stations[k]) in self.baselines and
                            (stations[j], stations[k]) in self.baselines):
                        loops.append((stations[i], stations[j], stations[k]))
        return loops

    def check_closure(self, loop: Tuple[str, str, str],
                      tolerance_mm: float | None = None) -> Dict[str, Any]:
        """
        检核单个同步环的闭合差。

        环方向: A → B → C → A
        闭合差 = 基线AB + 基线BC + 基线CA（矢量叠加，理论值应为0）

        返回:
            {
                'loop': (A, B, C),
                'dX_closure_mm': float,
                'dY_closure_mm': float,
                'dZ_closure_mm': float,
                'closure_3d_mm': float,
                'tolerance_mm': float,
                'pass': bool,
                'baselines_used': [(A,B), (B,C), (C,A)],
            }
        """
        A, B, C = loop
        dx_ab, dy_ab, dz_ab, _ = self.baselines.get((A, B), (0.0, 0.0, 0.0, None))
        dx_bc, dy_bc, dz_bc, _ = self.baselines.get((B, C), (0.0, 0.0, 0.0, None))
        dx_ca, dy_ca, dz_ca, _ = self.baselines.get((C, A), (0.0, 0.0, 0.0, None))

        dx_close = dx_ab + dx_bc + dx_ca
        dy_close = dy_ab + dy_bc + dy_ca
        dz_close = dz_ab + dz_bc + dz_ca
        close_3d = np.sqrt(dx_close ** 2 + dy_close ** 2 + dz_close ** 2)

        if tolerance_mm is None:
            tolerance_mm = self._default_tolerance(loop)

        return {
            'loop': loop,
            'dX_closure_mm': dx_close * 1000.0,
            'dY_closure_mm': dy_close * 1000.0,
            'dZ_closure_mm': dz_close * 1000.0,
            'closure_3d_mm': close_3d * 1000.0,
            'tolerance_mm': tolerance_mm,
            'pass': close_3d * 1000.0 <= tolerance_mm,
            'baselines_used': [(A, B), (B, C), (C, A)],
        }

    def _default_tolerance(self, loop: Tuple[str, str, str]) -> float:
        """
        根据基线平均长度估算默认限差（D级网标准）。
        限差 = 2·√3·σ，其中 σ = 10mm + 基线长度·1ppm
        """
        A, B, C = loop
        lengths = []
        for pair in [(A, B), (B, C), (C, A)]:
            dx, dy, dz, _ = self.baselines.get(pair, (0.0, 0.0, 0.0, None))
            lengths.append(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        avg_len_m = float(np.mean(lengths)) if lengths else 0.0
        sigma_mm = 10.0 + avg_len_m * 1e-6 * 1000.0
        return 2.0 * np.sqrt(3) * sigma_mm

    def check_all(self, tolerance_mm: float | None = None) -> List[Dict[str, Any]]:
        """对所有同步环执行检核，返回结果列表。"""
        loops = self.find_sync_loops()
        return [self.check_closure(loop, tolerance_mm) for loop in loops]

    def summary(self, tolerance_mm: float | None = None) -> Dict[str, Any]:
        """汇总统计。"""
        results = self.check_all(tolerance_mm)
        n_pass = sum(1 for r in results if r['pass'])
        n_total = len(results)
        return {
            'n_stations': self.n_stations,
            'n_baselines': self.n_baselines,
            'n_loops': n_total,
            'n_pass': n_pass,
            'n_fail': n_total - n_pass,
            'pass_rate': (n_pass / n_total * 100.0) if n_total > 0 else 0.0,
            'results': results,
        }

    def station_loops(self, station_name: str) -> List[Dict[str, Any]]:
        """获取某测站参与的所有同步环及其检核结果。"""
        results = self.check_all()
        return [r for r in results if station_name in r['loop']]


def generate_example_baseline() -> str:
    """生成示例基线成果文件内容，含合格和不合格环，供测试使用。

    模拟4个测站的短基线控制网（基线约5~15km），
    在一条基线上人为引入15cm偏差，产生2个不合格同步环。
    """
    # 测站局部坐标 (X, Y, Z) 单位 m，模拟约5~15km的短基线网
    coords = {
        'A001': (0.0, 0.0, 0.0),
        'A002': (8234.567, -3456.789, 5123.456),
        'A003': (-4567.890, 7890.123, -2345.678),
        'A004': (10234.111, 5678.222, 3456.333),
    }

    def baseline(a, b):
        ax, ay, az = coords[a]
        bx, by, bz = coords[b]
        return (bx - ax, by - ay, bz - az)

    lines = [
        "# GNSS 基线处理成果（示例）",
        "# 格式: 起点 终点 dX(m) dY(m) dZ(m) sigma(mm)",
        "# 坐标参考系: 局部坐标系",
        "#",
        "# 共4个测站、6条基线、4个同步环",
        "# 环1 (A001-A002-A003): 严格闭合 → 合格",
        "# 环2 (A001-A002-A004): 基线A002→A004 人为引入15cm偏差 → 不合格",
        "# 环3 (A001-A003-A004): 严格闭合 → 合格",
        "# 环4 (A002-A003-A004): 涉及含偏差的基线 → 不合格",
        "#",
    ]

    for pair in [('A001', 'A002'), ('A001', 'A003'), ('A002', 'A003')]:
        dx, dy, dz = baseline(*pair)
        lines.append(f"{pair[0]:<6s} {pair[1]:<6s}  {dx:>14.4f} {dy:>14.4f} {dz:>14.4f}  3.0")

    # A001→A004 (严格)
    dx, dy, dz = baseline('A001', 'A004')
    lines.append(f"{'A001':<6s} {'A004':<6s}  {dx:>14.4f} {dy:>14.4f} {dz:>14.4f}  3.0")

    # A002→A004 (引入15cm X偏差)
    dx, dy, dz = baseline('A002', 'A004')
    dx_bad = dx + 0.15  # +15cm偏差，远大于限差
    lines.append(f"{'A002':<6s} {'A004':<6s}  {dx_bad:>14.4f} {dy:>14.4f} {dz:>14.4f}  3.0")

    # A003→A004 (严格)
    dx, dy, dz = baseline('A003', 'A004')
    lines.append(f"{'A003':<6s} {'A004':<6s}  {dx:>14.4f} {dy:>14.4f} {dz:>14.4f}  3.0")

    return '\n'.join(lines)
