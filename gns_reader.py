"""
HITARGET GNS 二进制文件读取器

支持海星通/HITARGET RTK 接收机产出的 .GNS 原始观测文件，
解析文件头元数据、统计历元与卫星信息。

注意：GNS 是专有二进制格式，完整观测值解析需要借助 RTKLIB convbin 工具。
本模块提供尽力解析：提取测站ID、观测日期、历元数、卫星列表等基本信息。
"""

from __future__ import annotations

import re
import struct
from datetime import datetime
from typing import Any, Dict, List, Optional


class GnsFile:
    """HITARGET GNS 二进制文件读取器（尽力解析）。"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header: Dict[str, Any] = {}
        self.station_name: str = ""
        self.n_epochs: int = 0
        self.satellites: List[str] = []
        self.has_obs_data: bool = False  # 是否成功解析到观测数据
        self._raw: bytes = b""
        self._parse()

    def _parse(self) -> None:
        with open(self.filepath, "rb") as f:
            self._raw = f.read()

        self._parse_header()
        self._count_epochs()
        self._extract_satellites()

    def _parse_header(self) -> None:
        """从二进制头部提取文本元数据。"""
        # 搜索 epoch 标记，此前的内容为头部
        marker = b"\x44\x12\x1c\x30"
        header_end = self._raw.find(marker)
        if header_end < 0:
            header_end = min(2048, len(self._raw))

        header_bytes = self._raw[:header_end]
        text = header_bytes.decode("latin-1", errors="replace")

        # 版本
        ver_match = re.search(r"ver\s+([\d.]+)", text)
        if ver_match:
            self.header["version"] = ver_match.group(1)

        # 接收机型号（精确定位 HITRTK 后跟数字）
        rx_match = re.search(r"HITRTK(\d+)", text)
        if rx_match:
            self.header["receiver"] = f"HITRTK{rx_match.group(1)}"

        # 固件版本
        fw_match = re.search(r"\[(V\d+)\]", text)
        if fw_match:
            self.header["firmware"] = fw_match.group(1)

        self.header["manufacturer"] = "HITARGET" if "HITARGET" in text else ""

        # 日期
        date_match = re.search(r"Date:\s*(\d{4})/(\d{1,2})/(\d{1,2})", text)
        if date_match:
            self.header["date"] = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"

        # 设备ID → 作为测站名
        id_match = re.search(r"ID:\s*(\S+)", text)
        if id_match:
            raw_id = id_match.group(1)
            digits = re.findall(r"\d+", raw_id)
            self.station_name = digits[0] if digits else raw_id[:12]
        else:
            self.station_name = "GNS-Unknown"

        self.header["station_name"] = self.station_name

    def _count_epochs(self) -> None:
        """统计历元标记数量。"""
        marker = b"\x44\x12\x1c\x30"
        self.n_epochs = self._raw.count(marker)
        self.has_obs_data = self.n_epochs > 0

    def _extract_satellites(self) -> None:
        """
        从观测数据段提取卫星 PRN 列表。

        GNS 观测记录中 PRN 以 16-bit 整数出现在记录头部。
        遍历所有历元，提取出现过的 GPS(1-32) / BDS(1-63) PRN。
        """
        marker = b"\x44\x12\x1c\x30"
        seen = set()

        search_pos = 0
        while True:
            p = self._raw.find(marker, search_pos)
            if p < 0:
                break
            search_pos = p + 1

            # epoch 头部 64 字节后是卫星数据
            data_start = p + 64
            # 下一个 epoch 标记 或 文件末尾
            next_p = self._raw.find(marker, search_pos)
            data_end = next_p if next_p >= 0 else len(self._raw)
            data = self._raw[data_start:data_end]

            # 扫描 2 字节小整数
            for i in range(0, len(data) - 1, 2):
                v = struct.unpack("<H", data[i : i + 2])[0]
                if 1 <= v <= 63:
                    sys_char = "C" if v > 32 else "G"
                    seen.add(f"{sys_char}{v:02d}")

            if len(seen) >= 50:  # 够多了，不再扫描所有历元
                break

        self.satellites = sorted(seen)

    def get_satellites(self) -> List[str]:
        return self.satellites

    def get_system_composition(self) -> Dict[str, int]:
        comp: Dict[str, int] = {}
        for s in self.satellites:
            sys = s[0]
            comp[sys] = comp.get(sys, 0) + 1
        return comp

    def get_sampling_interval(self) -> Optional[float]:
        """GNS 文件通常 1s 采样，无法从头部精确获取。"""
        return None

    def summary(self) -> str:
        lines = [
            f"文件: {self.filepath}",
            f"格式: HITARGET GNS 二进制",
            f"测站ID: {self.station_name}",
            f"接收机: {self.header.get('receiver', '?')}",
            f"固件: {self.header.get('firmware', '?')}",
            f"日期: {self.header.get('date', '?')}",
            f"历元数: {self.n_epochs}",
            f"卫星数: {len(self.satellites)}",
            f"卫星列表: {', '.join(self.satellites[:20])}"
            + ("..." if len(self.satellites) > 20 else ""),
        ]
        return "\n".join(lines)
