"""
RINEX Navigation File Reader
支持 RINEX 2.xx 导航电文格式（GPS / BDS 广播星历）
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 常量
C = 299792458.0  # 光速 m/s

# GPS 导航电文字段（RINEX 2 标准 8 行/每星历）
GPS_EPH_FIELDS = [
    # line 1: PRN YY MM DD HH MM SS SV_clock_bias SV_clock_drift SV_clock_drift_rate
    ['prn', 'year', 'month', 'day', 'hour', 'minute', 'second',
     'sv_clock_bias', 'sv_clock_drift', 'sv_clock_drift_rate'],
    # line 2: IODE Crs Delta_n M0
    ['iode', 'crs', 'delta_n', 'm0'],
    # line 3: Cuc e Cus sqrt_A
    ['cuc', 'e', 'cus', 'sqrt_a'],
    # line 4: Toe Cic Omega0 Cis
    ['toe', 'cic', 'omega0', 'cis'],
    # line 5: i0 Crc omega OmegaDot
    ['i0', 'crc', 'omega', 'omega_dot'],
    # line 6: IDOT codes_on_L2 GPS_week L2_P_flag
    ['idot', 'codes_l2', 'gps_week', 'l2_p_flag'],
    # line 7: SV_accuracy SV_health TGD IODC
    ['sv_accuracy', 'sv_health', 'tgd', 'iodc'],
    # line 8: transmission_time fit_interval spare1 spare2
    ['transmission_time', 'fit_interval', 'spare1', 'spare2'],
]


class RinexNav:
    """RINEX 2.xx 导航电文读取器，支持 GPS 和 BDS 广播星历。"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header: Dict[str, Any] = {}
        self.iono_params: Dict[str, List[float]] = {}  # 电离层 Klobuchar 参数
        self._eph: Dict[str, List[Dict[str, float]]] = {}  # {PRN: [ephemeris_dict, ...]}
        self.system = 'G'  # 默认 GPS

        self._parse()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def get_ephemeris(self, prn: str, t: datetime) -> Optional[Dict[str, float]]:
        """获取指定卫星在时间 t 附近的有效星历（基于 Toe）。"""
        ephs = self._eph.get(prn, [])
        if not ephs:
            return None
        best = None
        best_dt = float('inf')
        for eph in ephs:
            toe = self._toe_to_datetime(prn, eph)
            dt = abs((t - toe).total_seconds())
            if dt < best_dt:
                best_dt = dt
                best = eph
        # 有效期通常在 Toe 附近 ±2 小时
        if best is not None and best_dt > 86400:
            return None
        return best

    def get_all_prns(self) -> List[str]:
        return sorted(self._eph.keys())

    def get_iono_params(self) -> Optional[Dict[str, List[float]]]:
        """返回电离层参数，若无则返回 None。"""
        if not self.iono_params:
            return None
        return self.iono_params

    def summary(self) -> str:
        lines = [
            f"文件: {self.filepath}",
            f"系统: {self.system}",
            f"卫星数: {len(self._eph)}",
            f"星历总数: {sum(len(v) for v in self._eph.values())}",
            f"卫星列表: {', '.join(sorted(self._eph.keys()))}",
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    def _parse(self):
        with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
        lines = raw.split('\n')

        header_end = self._parse_header(lines)

        if self.system == 'C':
            self._parse_eph_records(lines, header_end + 1, is_bds=True)
        else:
            self._parse_eph_records(lines, header_end + 1, is_bds=False)

    def _parse_header(self, lines: List[str]) -> int:
        """解析 NAV 文件头，返回 END OF HEADER 的行号。"""
        iono_alpha = []
        iono_beta = []

        for i, line in enumerate(lines):
            label = line[60:].strip()

            if 'RINEX VERSION / TYPE' in label:
                self.header['version'] = float(line[:9].strip())
                self.header['file_type'] = line[20:40].strip()
                sat_sys = line[40:41].strip()
                if sat_sys:
                    self.system = sat_sys

            elif 'PGM / RUN BY / DATE' in label:
                self.header['program'] = line[:20].strip()
                self.header['date'] = line[20:60].strip()

            elif 'ION ALPHA' in label:
                # Klobuchar alpha 参数 (4个)
                for k in range(4):
                    start = 2 + k * 12
                    try:
                        val = line[start:start+12].replace('D', 'E').strip()
                        iono_alpha.append(float(val))
                    except (ValueError, IndexError):
                        pass

            elif 'ION BETA' in label:
                for k in range(4):
                    start = 2 + k * 12
                    try:
                        val = line[start:start+12].replace('D', 'E').strip()
                        iono_beta.append(float(val))
                    except (ValueError, IndexError):
                        pass

            elif 'IONOSPHERIC CORR' in label:
                # 兼容某些 RINEX 写法：IONOSPHERIC CORR 后面可能跟 GPSA/GPSB
                ion_type = line[:60].strip()
                vals = self._extract_d_values(line[:60])
                if 'GPSA' in ion_type or 'ALPHA' in ion_type.upper():
                    iono_alpha = vals
                elif 'GPSB' in ion_type or 'BETA' in ion_type.upper():
                    iono_beta = vals

            elif 'END OF HEADER' in label:
                if len(iono_alpha) >= 4:
                    self.iono_params['alpha'] = iono_alpha[:4]
                if len(iono_beta) >= 4:
                    self.iono_params['beta'] = iono_beta[:4]
                return i

        return len(lines)

    def _parse_eph_records(self, lines: List[str], start: int, is_bds: bool):
        """解析广播星历数据记录。"""
        i = start
        while i < len(lines):
            line = lines[i]
            if len(line) < 60:
                i += 1
                continue

            # 检查是否是星历首行（PRN 开头）
            prn_match = re.match(r'\s*(\d{2})\s+', line)
            if not prn_match:
                i += 1
                continue

            prn_num = int(prn_match.group(1))
            sys_char = 'C' if is_bds else 'G'
            if prn_num > 32:
                sys_char = 'C'
            prn = f"{sys_char}{prn_num:02d}"

            # 收集 8 行数据
            eph_block = []
            for k in range(8):
                if i + k >= len(lines):
                    break
                eph_block.append(lines[i + k])

            if len(eph_block) < 8:
                i += 8
                continue

            eph = self._parse_eph_block(prn, eph_block, is_bds)
            if eph is not None:
                if prn not in self._eph:
                    self._eph[prn] = []
                self._eph[prn].append(eph)

            i += 8

    def _parse_eph_block(self, prn: str, block: List[str], is_bds: bool) -> Optional[Dict[str, float]]:
        """将 8 行星历数据块解析为字典。"""
        eph = {}
        try:
            # 第1行：PRN + 时间 + 钟差参数
            line0 = block[0]
            eph['prn'] = prn
            yy = int(line0[3:6].strip())
            mm = int(line0[6:9].strip())
            dd = int(line0[9:12].strip())
            hh = int(line0[12:15].strip())
            minute = int(line0[15:18].strip())
            sec = float(line0[18:22].strip())

            if yy >= 80:
                yy += 1900
            else:
                yy += 2000

            eph['toc'] = datetime(yy, mm, dd, hh, minute, int(sec), int((sec - int(sec)) * 1e6))
            eph['year'] = yy
            eph['month'] = mm
            eph['day'] = dd
            eph['hour'] = hh
            eph['minute'] = minute
            eph['second'] = sec

            # 钟差参数
            eph['sv_clock_bias'] = float(line0[22:41].replace('D', 'E').strip())
            eph['sv_clock_drift'] = float(line0[41:60].replace('D', 'E').strip())
            eph['sv_clock_drift_rate'] = float(line0[60:79].replace('D', 'E').strip()) if len(line0) > 60 else 0.0

            # 后续7行：每行4个参数
            line2_field_map = ['iode', 'crs', 'delta_n', 'm0']
            line3_field_map = ['cuc', 'e', 'cus', 'sqrt_a']
            line4_field_map = ['toe', 'cic', 'omega0', 'cis']
            line5_field_map = ['i0', 'crc', 'omega', 'omega_dot']
            line6_field_map = ['idot', 'codes_l2', 'gps_week', 'l2_p_flag']
            line7_field_map = ['sv_accuracy', 'sv_health', 'tgd', 'iodc']
            line8_field_map = ['transmission_time', 'fit_interval', 'spare1', 'spare2']

            all_maps = [line2_field_map, line3_field_map, line4_field_map,
                        line5_field_map, line6_field_map, line7_field_map, line8_field_map]

            for row_idx, field_map in enumerate(all_maps):
                line = block[row_idx + 1]
                for k, key in enumerate(field_map):
                    start = 3 + k * 19
                    end = min(start + 19, len(line))
                    val_str = line[start:end].replace('D', 'E').strip()
                    if val_str:
                        eph[key] = float(val_str)
                    else:
                        eph[key] = 0.0

        except (ValueError, IndexError) as ex:
            print(f"Warning: failed to parse ephemeris for {prn}: {ex}")
            return None

        return eph

    @staticmethod
    def _extract_d_values(text: str, count: int = 4) -> List[float]:
        """从文本中提取 D 格式的数值。"""
        vals = []
        pattern = re.compile(r'([+-]?\d+\.\d+[DE][+-]\d+)')
        matches = pattern.findall(text)
        for m in matches[:count]:
            vals.append(float(m.replace('D', 'E')))
        return vals

    def _toe_to_datetime(self, prn: str, eph: Dict[str, float]) -> datetime:
        """将星历的 Toe (GPS seconds of week) 转换为 datetime。"""
        toe_sow = eph.get('toe', 0.0)
        gps_week = eph.get('gps_week', 0)
        if toe_sow > 0 and gps_week > 0:
            return self._gps_sow_to_datetime(gps_week, toe_sow, prn.startswith('C'))
        # 回退：使用 toc 时刻
        return eph.get('toc', datetime(2000, 1, 1))

    @staticmethod
    def _gps_sow_to_datetime(gps_week: float, sow: float, is_bds: bool = False) -> datetime:
        """GPS week + seconds of week → datetime。"""
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        if is_bds:
            gps_epoch = datetime(2006, 1, 1, 0, 0, 0)  # BDT epoch
        total_seconds = timedelta(weeks=int(gps_week), seconds=sow)
        return gps_epoch + total_seconds


def timedelta(weeks: int = 0, days: int = 0, seconds: float = 0.0) -> 'datetime.timedelta':
    import datetime as _dt
    return _dt.timedelta(weeks=weeks, days=days, seconds=seconds)
