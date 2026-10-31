# change_detection_scripts — AEF 无监督变化检测

基于 **AlphaEarth Foundations (AEF)** 论文补充材料 S4.1 无监督变化检测流程的本地可视化工具。对同一瓦片的 64 维像素嵌入做逐像素比较，输出可交互的变化程度热力图，并与数据包自带的原始光学图像并排显示。

> ⚠️ 数据不含变化参考标注，所有结果只作**探索性对照**，不提供准确率或置信度。

---

## 📚 文档导航

- **[《WebUI 使用说明书》](./docs/webui-user-guide.md)** —— 启动方式、每个控件的作用、数据目录结构要求、常见问题
- **[完整目标规格](./docs/comet/specs/change-detection-webui/spec.md)** —— 数值口径、界面契约与验收条件
- **[归档变更记录](./docs/comet/archive/)** —— 三个已完成需求的需求简报、规格与验证报告

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **三种组合类型** | 同通道跨日（同模态对照）、同日跨通道（跨模态基线）、跨日跨通道（跨模态残差） |
| **数据根切换** | 界面内填写路径即可切换数据包，无需重启；支持数据包根或任意命名的瓦片容器 |
| **分档色阶** | 固定分段 / 动态分段两种模式，档数 2–64 可调，档间等值线可开关 |
| **自适应阈值** | 量程 `(0, x]` 随当前数据自动计算，滑块与数值框双向同步 |
| **三种显示模式** | 半透明红色叠加、二值掩膜、灰底彩色高亮 |
| **瓦片坐标** | 从 `tile_manifest.json` 的 `parent` 解析中心经纬度 |
| **有效像素掩膜** | 按传感器选取 `common_mask`，可开关 |

---

## 计算口径

| 量 | 公式 |
|----|------|
| 像素距离 | `d = (1 − cos) / 2`，取值 `[0, 1]` |
| 同日基线 | `base = d(A_ref, B_ref)` |
| 正向残差 | `fwd = d(A_ref, B_other) − base` |
| 反向残差 | `rev = d(B_ref, A_other) − base` |
| 对称化残差 | `sym = (fwd + rev) / 2` |

残差取值截断在 `[−1, 1]`。距离计算前对每个像素向量做 L2 归一化。

---

## 环境要求

- Python ≥ 3.12
- 依赖：`nicegui`、`numpy`、`pillow`

```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
.venv/Scripts/python.exe -m pip install nicegui numpy pillow
```

---

## 运行

```bash
.venv/Scripts/python.exe app/change_detection_webui.py
```

浏览器打开 <http://127.0.0.1:8080/>。

### 命令行参数

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8080` | 监听端口 |
| `--data-root` | `data/spectral` | 初始数据包根目录 |
| `--show` | 关闭 | 启动后自动打开浏览器 |

---

## 数据

`data/` 目录下的数据包**不纳入版本控制**（单份约 2.2 GB）。使用前请准备符合要求的数据包，或用 `--data-root` / 界面路径控件指向本地数据包。

目录结构要求与逐项字段说明见 [《WebUI 使用说明书》第 2.4 节](./docs/webui-user-guide.md#24-数据目录结构要求)。

最小必需结构：

```
<数据根>/
├── tiles/<瓦片 id>/
│   ├── tile_manifest.json
│   ├── embeddings/<视图>.npy + <视图>_index.json
│   └── distance_maps/<传感器>_*.npz        # 含 common_mask
└── gallery/
    ├── index.json
    └── <瓦片>_<传感器>_<帧序号>.png
```

---

## 目录结构

```
change_detection_scripts/
├── app/
│   └── change_detection_webui.py    # 单文件 NiceGUI 应用
├── docs/
│   ├── webui-user-guide.md          # 使用说明书
│   └── comet/                       # 需求规格与归档记录
├── data/                            # 数据包（git 忽略）
├── AGENTS.md                        # Agent 协作规则
└── .gitignore
```

---

## 验证

项目内保留了两套独立复算脚本，可在改动后回归：

```bash
# 跨模态数值复算（C0 严格 2 倍关系、残差与对称化公式）
.venv/Scripts/python.exe docs/comet/archive/2026-10-27-webui-cross-channel-change/scratch/verify_impl.py

# 同通道跨日回归与旋转一致性
.venv/Scripts/python.exe docs/comet/archive/2026-10-28-webui-gallery-layout-and-mask-default/scratch/check_regression.py
```

两套脚本均不需要启动服务，直接读取 `data/spectral` 即可。
