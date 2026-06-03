# 卫星导航定位程序 — GNSS控制网观测数据处理与定位解算

## 课程设计题目

**卫星导航定位程序** — 面向本科课程设计，涵盖 RINEX 数据读取、观测质量分析、卫星轨道计算、伪距单点定位 (SPP) 与坐标转换的完整处理链。

## 主程序入口

```bash
app.py
```

## 功能模块

### 数据读取
1. **RINEX OBS 文件读取** — 支持 RINEX 2.xx 观测文件，解析头信息、观测类型、历元与各卫星观测数据
2. **RINEX NAV 文件读取** — 支持 RINEX 2.xx 导航电文，解析 GPS/BDS 广播星历参数、电离层 Klobuchar 参数

### 观测数据质量分析
3. **文件摘要** — 测站名、历元数、观测类型、卫星数量、卫星列表
4. **SNR 信噪比分析** — 平均值、最大值、最小值、时间序列图、统计表
5. **MW + GF 周跳探测** — TurboEdit 简化版，双阈值联合判别，标注可疑跳变
6. **MP1 / MP2 多路径分析** — 双频伪距-载波组合，去均值展示，RMS 统计
7. **卫星可见性统计** — GPS/BDS/GAL/GLO 分类统计，时间序列图

### 卫星轨道与定位解算 ★ 核心
8. **广播星历轨道计算** — 基于 ICD-GPS-200 标准算法，由星历参数计算卫星 ECEF 坐标
9. **卫星钟差改正** — 广播星历钟差多项式 + 相对论修正
10. **对流层延迟改正** — Saastamoinen 模型
11. **电离层延迟改正** — Klobuchar 模型（需 NAV 文件含电离层参数）
12. **伪距单点定位 (SPP)** — 加权最小二乘迭代解算，输出接收机三维坐标与钟差
13. **精度评定** — GDOP / PDOP / HDOP / VDOP / TDOP、定位残差 RMS、坐标标准差
14. **卫星天空图 (Skyplot)** — 极坐标显示卫星方位角/高度角分布

### 坐标转换
15. **BLH ↔ XYZ** — 大地坐标与空间直角坐标互转（WGS84 / CGCS2000）
16. **高斯-克吕格投影** — 3° 带和 6° 带正反算

### 综合报告
17. **综合质量评价** — 100 分制评分 + A/B/C/D 等级 + 扣分明细
18. **文本报告导出** — 完整分析报告 + 简洁评价报告可下载

## 架构

```
app.py                  # Streamlit 网页前端主界面
rinex_reader.py         # RINEX OBS 观测文件读取器
rinex_nav_reader.py     # RINEX NAV 导航电文读取器 ★ 新增
satellite_orbit.py      # 卫星轨道计算 (GPS/BDS ICD) ★ 新增
spp_positioning.py      # 伪距单点定位解算 ★ 新增
gnss_quality.py         # 观测数据质量分析 (SNR/周跳/多路径/评分)
coord_convert.py        # 坐标转换 (BLH↔XYZ, 高斯投影)
```

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 使用流程

1. 在左侧上传 **RINEX OBS 文件**（必须）
2. 在左侧上传 **RINEX NAV 文件**（用于定位解算，可选但强烈推荐）
3. 调整分析参数（MW/GF阈值、截止高度角等）
4. 查看各标签页的分析结果

## 数据文件获取

IGS 公开观测数据可从以下站点下载：
- CDDIS: https://cddis.nasa.gov/
- IGS: https://www.igs.org/
- SOPAC: http://sopac-csrc.ucsd.edu/
- 武汉大学 IGS 数据中心: http://www.igs.gnsswhu.cn/

## 技术参考

- ICD-GPS-200: GPS 空间段/用户段接口规范
- BDS-SIS-ICD: 北斗卫星导航系统空间信号接口控制文件
- RINEX 2.11 格式规范
- TurboEdit 周跳探测算法 (Blewitt, 1990)
- Saastamoinen 对流层延迟模型
- Klobuchar 电离层延迟模型
