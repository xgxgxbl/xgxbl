"""
RINEX Observation File Reader
支持 RINEX 2.11 OBS 格式
"""

import re
import numpy as np
from datetime import datetime


class RinexObs:
    """RINEX 2.xx 观测文件读取器"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.header = {}
        self.obs_types = []
        self.system = 'G'
        self._data = {}
        self.times = []

        self._parse_header()
        self._read_data()

    def _parse_header(self):
        """解析 RINEX OBS 头文件"""
        with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()

        lines = raw.split('\n')

        header_end = 0
        obs_type_lines = []

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

            elif 'MARKER NAME' in label:
                self.header['marker_name'] = line[:60].strip()

            elif 'OBSERVER / AGENCY' in label:
                self.header['observer'] = line[:20].strip()
                self.header['agency'] = line[20:60].strip()

            elif 'REC # / TYPE / VERS' in label:
                self.header['receiver'] = line[:60].strip()

            elif 'ANT # / TYPE' in label:
                self.header['antenna'] = line[:60].strip()

            elif 'APPROX POSITION XYZ' in label:
                try:
                    self.header['approx_xyz'] = [
                        float(line[:14]), float(line[14:28]), float(line[28:42])
                    ]
                except ValueError:
                    pass

            elif 'ANTENNA: DELTA H/E/N' in label:
                try:
                    self.header['antenna_offset'] = [
                        float(line[:14]), float(line[14:28]), float(line[28:42])
                    ]
                except ValueError:
                    pass

            elif '# / TYPES OF OBSERV' in label:
                ntypes = int(line[:6].strip())
                obs_type_lines.append(line[6:60])

            elif 'INTERVAL' in label:
                try:
                    self.header['interval'] = float(line[:10].strip())
                except ValueError:
                    pass

            elif 'END OF HEADER' in label:
                header_end = i
                break
            else:
                # 可能是观测类型的续行
                if obs_type_lines and line[60:].strip() == '' and 'COMMENT' not in line[60:]:
                    obs_type_lines.append(line[6:60])

        # 解析观测类型
        obs_str = ''.join(obs_type_lines)
        self.obs_types = []
        for j in range(0, len(obs_str), 6):
            ot = obs_str[j:j+6].strip()
            if ot:
                self.obs_types.append(ot)

        self._header_end = header_end
        self._lines = lines

    def _read_data(self):
        """读取全部观测数据"""
        lines = self._lines[self._header_end + 1:]
        ntypes = len(self.obs_types)
        if ntypes == 0:
            return

        data = {}
        times = []
        i = 0

        while i < len(lines):
            line = lines[i]
            if len(line) < 30:
                i += 1
                continue

            # 尝试匹配历元行: YY MM DD HH MM SS.sssssss
            epoch_match = re.match(
                r'\s*(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s*([\d.]+)',
                line
            )

            if not epoch_match:
                i += 1
                continue

            yy, mm, dd, hh, minute, sec = epoch_match.groups()
            yy = int(yy); mm = int(mm); dd = int(dd)
            hh = int(hh); minute = int(minute)
            sec = float(sec)

            if yy >= 80:
                yy += 1900
            else:
                yy += 2000

            try:
                epoch = datetime(yy, mm, dd, hh, minute,
                                 int(sec), int((sec - int(sec)) * 1e6))
            except ValueError:
                i += 1
                continue

            # 解析 epoch flag 和卫星数
            flag_str = line[28:29].strip()
            flag = int(flag_str) if flag_str else 0
            nsat_str = line[29:32].strip()
            nsat = int(nsat_str) if nsat_str else 0

            if flag > 0:
                i += 1
                continue

            # 读取卫星列表
            sats = []
            for j in range(nsat):
                pos = 32 + j * 3
                if pos + 3 <= len(line):
                    prn = line[pos:pos+3].strip()
                    if prn:
                        sats.append(prn)

            # 如果卫星数超过12，读续行
            if nsat > 12:
                remaining = nsat - 12
                while remaining > 0:
                    i += 1
                    cline = lines[i]
                    for j in range(min(remaining, 12)):
                        pos = 32 + j * 3
                        if pos + 3 <= len(cline):
                            prn = cline[pos:pos+3].strip()
                            if prn:
                                sats.append(prn)
                    remaining -= 12

            i += 1
            times.append(epoch)

            # 读取每颗卫星的观测数据
            for sat in sats:
                if sat not in data:
                    data[sat] = {ot: [] for ot in self.obs_types}
                    data[sat]['epoch'] = []

                data[sat]['epoch'].append(epoch)

                # 计算需要几行来存 ntypes 个观测值（每行5个）
                nlines = (ntypes + 4) // 5
                obs_values = []

                for ln in range(nlines):
                    if i >= len(lines):
                        break
                    oline = lines[i]

                    for k in range(5):
                        start = 3 + k * 16
                        read_len = min(14, max(0, len(oline) - start))
                        if read_len > 0:
                            val_str = oline[start:start+read_len].strip()
                            try:
                                obs_values.append(float(val_str))
                            except ValueError:
                                obs_values.append(np.nan)
                        else:
                            obs_values.append(np.nan)

                    i += 1

                # 填到对应观测类型
                for k, ot in enumerate(self.obs_types):
                    if k < len(obs_values):
                        data[sat][ot].append(obs_values[k])
                    else:
                        data[sat][ot].append(np.nan)

        self._data = data
        self.times = sorted(set(times))
        self._satellites = sorted(data.keys())

    def get_data(self):
        """返回 {卫星: {观测类型: [值列表]}}"""
        return self._data

    def get_times(self):
        """返回历元列表"""
        return self.times

    def get_satellites(self):
        """返回卫星PRN列表"""
        return self._satellites

    def get_obs_for_sat(self, sat):
        """返回指定卫星的观测数据"""
        return self._data.get(sat, {})

    def has_obs_type(self, code):
        """检查是否存在某观测类型"""
        return code in self.obs_types

    def summary(self):
        """打印文件摘要"""
        info = [
            f"文件: {self.filepath}",
            f"版本: RINEX {self.header.get('version', '?')}",
            f"测站: {self.header.get('marker_name', '?')}",
            f"观测类型 ({len(self.obs_types)}): {', '.join(self.obs_types)}",
            f"卫星数: {len(self._satellites)}",
            f"历元数: {len(self.times)}",
            f"卫星列表: {', '.join(self._satellites[:10])}{'...' if len(self._satellites) > 10 else ''}",
        ]
        return '\n'.join(info)
