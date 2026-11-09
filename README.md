# change_detection_scripts — AEF 变化检测平台

本地多页面可视化平台，包含两个页面：

- **AEF 无监督变化检测**（`/`、`/aef`）：基于 **AlphaEarth Foundations (AEF)** 论文补充材料 S4.1 无监督变化检测流程，对同一瓦片的 64 维像素嵌入做逐像素比较，输出可交互的变化程度热力图，并与数据包自带的原始光学图像并排显示。
- **AEF 变化检测与 DW 标签对比**（`/dw`）：对带 Dynamic World 标签的数据包，展示两期标签与官方配色、变化过渡叠加，并给出整瓦片的嵌入距离统计。

> ⚠️ 数据不含人工核实的变化参考标注，DW 标签本身也是模型分类。所有结果只作**探索性对照**，不提供准确率、精度或置信度。

---

## 📚 文档导航

- **[《平台使用说明书》](./docs/webui-user-guide.md)** —— 启动方式、两个页面、每个控件的作用、数据目录结构要求、常见问题
- **[完整目标规格](./docs/comet/specs/change-detection-webui/spec.md)** —— 数值口径、界面契约与验收条件（归档后为当前生效版本）
- **[归档变更记录](./docs/comet/archive/)** —— 已完成需求的需求简报、规格与验证报告

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **三种组合类型** | 同通道跨日（同模态对照）、同日跨通道（跨模态基线）、跨日跨通道（跨模态残差） |
| **数据根切换** | 界面内填写路径即可切换数据包，无需重启；支持数据包根或任意命名的瓦片容器 |
| **分档色阶** | 固定分段 / 动态分段两种模式，档数 2–64 可调，档间等值线可开关 |
| **自适应阈值** | 量程 `(0, x]` 随当前数据自动计算，滑块与数值框双向同步 |
| **三种显示模式** | 半透明红色叠加、二值掩膜、灰底彩色高亮 |
| **瓦片坐标** | 从 `tile_manifest.json` 的 `parent` 解析中心经纬度，缺失时回退 `region_id` |
| **有效像素掩膜** | 按传感器选取 `common_mask`，可开关；无 `distance_maps` 时按文档顺序回退 |
| **平台导航** | `/`、`/aef` 为 AEF 页，`/dw` 为 DW 标签对比页，顶部常驻入口与选中态 |
| **DW 标签对比** | 两期 DW 标签（0–8 类别、255 无效）与官方配色、`candidate_mask`/bbox 叠加、整瓦片嵌入距离统计、`preview.png`/`review.png` |
| **两页共用的显示自定义** | 色阶模式（固定/动态分段）、档数 2–64、档间边界线开关、三种显示模式；同一套渲染函数与常量 |

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

DW 页使用同一个距离函数，只统计整瓦片的有效像素；`candidate_mask` 与 `bbox_yxyx` 仅作叠加显示，不参与统计筛选。DW 页不做 S2 ↔ S1 跨传感器比较。

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

浏览器打开 <http://127.0.0.1:8080/>（AEF 页）；DW 页在 <http://127.0.0.1:8080/dw>。

### 命令行参数

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8080` | 监听端口 |
| `--data-root` | `data/spectral` | AEF 页初始数据包根目录 |
| `--dw-data-root` | 仓库同级的 `AEF_DW_change_data_20261031` | DW 页初始数据包根目录 |
| `--show` | 关闭 | 启动后自动打开浏览器 |

---

## 数据

`data/` 目录下的数据包**不纳入版本控制**（单份约 2.2 GB）。使用前请准备符合要求的数据包，或用 `--data-root` / 界面路径控件指向本地数据包。

DW 页默认使用仓库同级的 `AEF_DW_change_data_20261031`（同样不纳入版本控制），可用 `--dw-data-root` 或 DW 页自己的路径控件切换。**两个页面都只读取数据包，不写入、不修改任何数据文件**；旋转与着色全部在内存中完成。

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
│   ├── change_detection_webui.py    # 入口：命令行参数、路由、服务启动
│   └── webui/                       # 平台模块
│       ├── dataset.py               # 路径解析、Dataset、掩膜与图像回退
│       ├── render.py                # 距离、色阶、阈值量程、统计
│       ├── layout.py                # 平台标题栏与导航
│       ├── aef_page.py              # AEF 无监督变化检测页
│       ├── dw_data.py               # DW 标签、图例、变化过渡
│       └── dw_page.py               # DW 标签对比页
├── tests/
│   └── test_platform.py             # unittest：数值复算与数据包读取
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
# 平台自动化检查（距离/残差/阈值复算、DW 读取、兼容回退、整瓦片统计、
# 显示控件夹取与"只影响显示"约束）
.venv/Scripts/python.exe -m unittest discover -s tests -v

# 三页路由可访问性（启动临时服务后 GET / /aef /dw）
.venv/Scripts/python.exe tests/check_routes.py

# 同通道跨日回归与旋转一致性（读取 data/spectral 与归档基线）
.venv/Scripts/python.exe docs/comet/archive/2026-10-28-webui-gallery-layout-and-mask-default/scratch/check_regression.py
```

这些检查都不需要浏览器；`unittest` 会在数据包不存在时自动跳过相关用例。

> 归档目录中的 `2026-10-27-.../scratch/verify_impl.py` 是按**单文件**版本写的，它直接扫描
> `app/change_detection_webui.py` 的源码并假定同一目录下有 `data/`；拆分模块后该脚本不再适用，
> 其数值检查已由 `tests/test_platform.py` 覆盖，因此不再列为回归命令。
