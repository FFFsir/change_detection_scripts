# 目标

把现有单文件 NiceGUI 应用 `app/change_detection_webui.py` 改造为「平台外壳 + 多页面」，并新增一个 DW 标签对比页：

1. **平台外壳**：所有页面共享标题栏与页面导航，可在「AEF 无监督变化检测」与「AEF 变化检测与 DW 标签对比」之间切换。
2. **AEF 无监督变化检测页**：把 `app/change_detection_webui.py` 现有的全部能力原样迁移到该页；根路径 `/` 仍直接呈现该页。三种组合类型、数据根切换与失败回滚、固定/动态分档色阶、自适应阈值、三种显示模式、common_mask 开关、瓦片坐标、三栏独立滚动、gallery 与热力图顺时针 90° 等行为不得回归。
3. **DW 标签对比页**：对带 Dynamic World 标签的数据包（参考 `D:\Project\光机所项目\AEF_DW_change_data_20261031`）做可视化分析：两期 DW 标签与官方配色、`candidate_mask` 与 bbox 叠加、manifest 元信息、整瓦片嵌入距离统计，并支持在同一页切换数据根。本页统计口径为整个瓦片，不划分「核心区 / 核心区外」。本页的热力图显示可由用户自定义，能力与「AEF 无监督变化检测」页一致：色阶模式（固定/动态分段）、档数（2–64）、档间边界线开关、三种显示模式。
4. **兼容差异**：同一套读取层同时接受旧包与 DW 包在视图命名、掩膜、gallery、manifest 字段上的差异。

# 范围

- 改造 `app/change_detection_webui.py` 为入口脚本（命令行参数、页面注册、服务启动），把数据访问、计算、渲染、页面构建拆分到 `app/webui/` 包内的模块；不新增第三方依赖。
- 新增 DW 页的数据读取与页面构建模块（标签、官方配色图例、`label_transition.npz`、manifest、`preview.png` / `review.png`）。
- 更新完整目标规格 `docs/comet/changes/webui-multipage-dw-labels/specs/change-detection-webui/spec.md`（归档时替换 `docs/comet/specs/change-detection-webui/spec.md`）。
- 更新 `docs/webui-user-guide.md` 与 `README.md`，补充平台导航与 DW 页说明。
- 新增基于标准库 `unittest` 的自动化检查（数值复算、DW 读取、兼容回退、页面可访问性）；不新增 pytest 等依赖。
- 不修改 `data/` 与 DW 数据包内任何文件；运行与验证使用 `.venv/Scripts/python.exe`。

## Source coverage

来源：用户在 2026-11-08 会话中通过 `/comet` 提出的需求（对话内直接给出，无附件）；调查对象为仓库内 `app/change_detection_webui.py`、`docs/comet/specs/change-detection-webui/spec.md` 与本地数据包 `D:\Project\光机所项目\AEF_DW_change_data_20261031`。读取状态：complete。

| 单元 | 内容 | 状态 | Spec 位置 | 验收 ID | 覆盖 |
| --- | --- | --- | --- | --- | --- |
| U1 | 改造为平台外壳 + 多页面，提供页面导航 | complete | 平台结构·外壳与导航 | A1 | covered |
| U2 | 现有能力原样迁移为「AEF 无监督变化检测」页，旧根路径 `/` 仍可直接访问 | complete | AEF 页·全部既有章节、平台结构·路由 | A1、A2 | covered |
| U3 | 两期 DW 标签（0–8 类别，255 无效）及类别名与官方配色，取自 `land_cover/dynamic_world.npz` 与 `dynamic_world_legend.json` | complete | DW 页·标签数据与配色 | A3 | covered |
| U4 | 叠加 `masks/label_transition.npz` 的 `candidate_mask`、`bbox_yxyx`、`before_label`/`after_label` | complete | DW 页·candidate_mask 叠加 | A4 | covered |
| U5 | 展示 `tile_manifest.json` 的前后类别、`core_pixels`、两期平均置信度（pmax） | complete | DW 页·元信息与 pmax 口径 | A5 | covered |
| U6 | 复用嵌入距离计算，给出整瓦片的均值、中位数、p90、p95、最大值与超阈值比例 | complete | DW 页·整瓦片距离统计、计算·距离 | A6 | covered |
| U7 | 在同一页把数据根切到 DW 数据包 | complete | DW 页·数据根切换 | A7 | covered |
| U8 | 兼容差异：视图名无波段组、无 `distance_maps`（回退 `valid_mask[0]&valid_mask[1]` 或全 True）、无 `gallery/index.json`（回退 `preview.png`/`review.png`）、manifest 用 `tile_id`/`region_id` | complete | 输入数据契约·兼容回退、DW 页·图像 | A7、A8 | covered |
| U9 | 只读数据包；使用 `.venv/Scripts/python.exe` | complete | 服务与运行 | A9 | covered |
| U10 | 不改距离 `d=(1-cos)/2`、残差与对称化公式、阈值量程与统计口径 | complete | 计算·全部小节 | A6、A9 | covered |
| U11 | 不破坏三份已归档验收结论 | complete | 非目标、AEF 页·既有行为 | A2 | covered |
| U12 | DW 标签为模型分类而非人工核实真值；可展示 pmax 但须注明为模型概率，不得出现准确率或精度类结论 | complete | DW 页·元信息与 pmax 口径、结果语义提示 | A5 | covered |
| U13 | S2↔S1 跨传感器对比是否放开，在 Shape 阶段确认 | complete | 决策 D1、DW 页·传感器范围 | A8 | covered |
| U14 | DW 页补齐与 AEF 页一致的显示自定义能力：档间边界线开关、档数调节栏等 | complete | DW 页·显示自定义 | A6、A11 | covered |

# 非目标

- 不修改距离公式 `d=(1-cos)/2`、残差/对称化公式、阈值量程规则与统计口径。
- 不在 DW 页放开 Sentinel-2 ↔ Sentinel-1 跨传感器对比（见 D1）。
- 不修改 `data/` 与 DW 数据包内任何文件，不写配置文件，不跨页面共享数据根选择。
- 不新增第三方依赖（含 pytest、playwright）；不访问网络。
- 不修改三份已归档 change 的任何文件与结论。
- 不提供 DW 标签的人工校正、精度评估、准确率或混淆矩阵。
- 不改 AEF 页的既有布局尺寸、颜色表色相与旋转约定。
- 不把 DW 页的热力图旋转：DW 页标签图、热力图与 `preview.png` 全部保持数据原生朝向（见 D4）。

# 验收示例

- A1：启动应用后 `/` 与 `/aef` 都呈现「AEF 无监督变化检测」页，`/dw` 呈现 DW 标签对比页；每页顶部常驻平台标题与两个页面入口，当前页入口有可见的选中样式；点击入口完成整页切换，切换后无需重启进程，三页均无服务端异常。
- A2：AEF 页保留三种组合类型及其控件/语义提示、数据根切换与失败回滚（非法路径提示且不破坏当前数据与用户阈值）、固定/动态分档色阶（档数 2–64、边界线开关、二值掩膜不画边界）、自适应阈值 `(0, x]` 滑块与数值框、三种显示模式、common_mask 开关（默认关闭，分母 16384）、瓦片中心坐标与“无坐标信息”提示、三栏各自独立滚动且整页不滚动、gallery 两帧并排且与热力图同向顺时针 90°、缺图提示保留多行；同日跨通道与 C0 的严格 2 倍关系、跨日跨通道 `fwd`/`rev`/`sym` 公式、单期 Landsat 过滤、掩膜按传感器选取均与迁移前一致。
- A3：DW 页从 `land_cover/dynamic_world.npz` 读取 `label`（2×128×128，0–8 为类别、255 为无效）、`probabilities`、`valid_mask`、`pmax`，从数据包根 `dynamic_world_legend.json` 读取 9 个类别的代码、名称与官方色值；前后两期标签图逐像素按官方配色渲染，255/无效像素用固定中性色并注明，页面图例列出全部 9 类与无效值。
- A4：DW 页两期标签图上叠加 `masks/label_transition.npz` 的 `candidate_mask` 轮廓与 `bbox_yxyx` 矩形（`[y0,x0,y1,x1]`，末端不含），并在界面显示 `before_label`/`after_label` 的代码与类别名；显示的 `core_pixels` 等于 `candidate_mask.sum()`，且与 `tile_manifest.json` 的 `core_pixels`、`before_label`、`after_label` 一致。
- A5：DW 页显示 `tile_manifest.json` 的 `tile_id`、`region_id`、`split`、`before_class → after_class`、`core_pixels` 与两期 `before_mean_confidence`/`after_mean_confidence`；置信度一律标注为“平均 pmax（模型概率）”，页面同时声明 DW 标签是模型分类、不是人工核实真值；全页不出现准确率、精度、召回率或真值一致性之类结论。
- A6：DW 页对当前瓦片与所选传感器，用 `d=(1-cos)/2` 计算两期嵌入距离，并在整瓦片全部有效像素上给出像素数、均值、中位数、p90、p95、最大值与超阈值比例；阈值量程按该像素集合计算；切换瓦片、传感器或阈值后热力图与统计立即更新，且可由独立 numpy 复算逐项复现；页面不做「核心区 / 核心区外」的分组统计。
- A7：DW 页的数据根输入框可在同一页把数据根切到 DW 数据包并完成整体刷新（瓦片、标签图、热力图、统计、预览图全部更新），非法路径显示可读错误且不破坏当前已加载数据；切入不含 DW 标签的旧包时显示“该数据包不含 DW 标签”类提示而不是伪造标签；包内无 `distance_maps` 时掩膜回退为 `valid_mask[0] & valid_mask[1]`，该掩膜也不存在时回退为全 True，界面注明当前掩膜来源与有效像素数；包内无 `gallery/index.json` 时图像区显示该瓦片的 `preview.png` 与 `review.png`，文件缺失时给出可读提示。
- A8：同一读取层在 DW 包上把视图名 `sentinel2`/`sentinel1` 当作无波段组的单视图处理：AEF 页不把它标成“分布内/分布外”，而显示“单视图（无波段组）”；瓦片标签回退使用 `tile_id`/`region_id`/`split`，坐标从 `region_id` 解析；DW 页不提供任何 S2↔S1 跨传感器对比入口或结果。
- A9：`d=(1-cos)/2`、残差与对称化公式、阈值量程规则与统计口径的自动化复算结果与迁移前逐像素一致；运行前后 `data/` 与 DW 数据包的路径、大小聚合哈希不变（只读）；全部模块可用 `.venv/Scripts/python.exe -m py_compile` 编译通过。
- A10：`docs/webui-user-guide.md` 覆盖平台导航与 DW 页：标签配色、`candidate_mask` 与 bbox、pmax 口径、整瓦片统计、数据根切换、掩膜与图像回退、原生朝向说明；`README.md` 的文档导航与功能表同步更新，文档描述与当前实现一致。
- A11：DW 页提供与 AEF 页一致的图像显示自定义控件：色阶模式可切换「固定分段」与「动态分段」，档数可用整数输入框调整（默认 12，范围 2–64，越界输入被夹取并回写），档间边界线开关可开可关（默认开启），显示模式可选半透明红色叠加/二值掩膜/灰底彩色高亮（默认叠加）；色阶只影响显示，不改变距离数值、阈值、超阈值统计与量程；二值掩膜模式不绘制档间边界线；切换任一控件后热力图与色条立即更新且无需刷新页面。

# 约束与不变量

- 只读数据包：运行期不得写入 `data/`、DW 数据包或任何数据文件；旋转与渲染只在内存中完成。
- 使用 `.venv/Scripts/python.exe`；不新增第三方依赖；不访问网络。
- 距离、残差、对称化、阈值量程、统计口径逐字沿用现有实现：
  - `d = (1 - cos) / 2`，截断到 `[0, 1]`；
  - `base = d(A_ref, B_ref)`、`fwd = d(A_ref, B_other) - base`、`rev = d(B_ref, A_other) - base`、`sym = (fwd + rev) / 2`，截断到 `[-1, 1]`；
  - 阈值量程 `(0, x]`，`x` 为严格大于当前最大值的步长倍数，左端不可取 0；
  - 超阈值统一按 `q > threshold`，带符号视图按 `|r| > threshold`。
- AEF 页迁移必须保持模块级行为：`/` 直接构建同一页面；命令行参数 `--host`、`--port`、`--data-root`、`--show` 保留原有语义。
- DW 页的距离统计只做同一传感器内的前后两期比较；每期取该视图的第一帧与最后一帧（DW 包中均为 2 帧）。
- 数据根切换按页独立：不写配置文件，不跨页面或跨客户端共享当前选择。
- 掩膜回退顺序固定：`distance_maps/<sensor>_*.npz` 的 `common_mask` → `land_cover/dynamic_world.npz` 的 `valid_mask[0] & valid_mask[1]` → `observations/<sensor>.npz` 的 `valid_mask` 两帧与 → 全 True。若 `distance_maps/` 目录存在但缺少该传感器掩膜，仍按既有契约为错误（不得静默回退），以保持归档结论。
- DW 包页面不得出现准确率或精度类结论；`before_mean_confidence`/`after_mean_confidence` 必须标注为模型概率（平均 pmax）。
- 三份已归档 change（`2026-10-20-aef-unsupervised-change-webui`、`2026-10-27-webui-cross-channel-change`、`2026-10-28-webui-gallery-layout-and-mask-default`）的文件与结论不得修改。

# 决策

- D1（用户确认）：DW 页不放开 S2↔S1 跨传感器对比。只做同一传感器内前后两期比较；跨模态差异不显示、不构造残差。
- D2（用户确认，二次修正）：DW 页统计口径为整个瓦片，不做「核心区 / 核心区外」的任何分组统计。嵌入距离的有效像素集合 = 嵌入可用且掩膜为真的全部像素；`candidate_mask` 只作为标签图与热力图上的空间叠加层和元信息展示，不参与任何统计筛选。
- D3：DW 页默认数据根为仓库同级目录 `AEF_DW_change_data_20261031`（即用户给出的参考包），可用新增命令行参数 `--dw-data-root` 覆盖；该目录不存在时回退到 `--data-root` 并在页面显示提示。
- D4：DW 页的标签图、热力图与 `preview.png`/`review.png` 全部保持数据原生朝向，不做 90° 旋转；AEF 页保留既有顺时针 90° 约定。理由是 DW 页需要与 `preview.png` 的面板朝向逐像素对齐，旋转多面板图会让图内文字侧转。
- D5：平台采用「入口脚本 + 包」结构：`app/change_detection_webui.py` 保留为唯一入口（命令行与服务启动），数据访问、计算渲染、页面外壳、两个页面分别位于 `app/webui/` 下的模块，便于独立测试。用户可见行为不变。这是结构决定，不是用户可见需求；DW 页与 AEF 页仍共用同一份距离/阈值/统计实现，避免两套口径。
- D6：数据根按页独立，不跨页面共享；每页各自提供输入框与应用按钮。
- D7：AEF 页在无波段组视图（`sentinel2`/`sentinel1`）上显示“单视图（无波段组）”，不再套用“分布内/分布外”标签；旧包的 `full` 与波段消融视图标注保持不变。
- D8（用户修正）：DW 页改为暴露与 AEF 页一致的完整显示自定义控件：色阶模式（固定/动态分段）、档数（2–64，默认 12）、档间边界线开关（默认开启）、三种显示模式（默认半透明红色叠加）。两页共用同一套 `render_bands` / `render_band_strip` 与同一组常量（`BAND_MODES`、`LEVELS_MIN/MAX/DEFAULT`、`MODES`）和同一套夹取/回写规则，避免两页口径分叉。
- D9：DW 页的显示自定义控件只作用于距离热力图与色条；标签图、预览图、`candidate_mask` 轮廓与 bbox 矩形保持始终显示（与已验证的 A4 一致），不提供显隐开关。DW 页固定分段的量程为 `[0, 1]`（非负量），动态分段沿用与阈值量程一致的上界 `x`；切换模式不改变统计与阈值。
- D10：新增 `unittest` 自动化检查放在 `tests/`，仅在数据包存在时运行对应用例；不引入 pytest。

# 待解决问题

（无）

# 验证预期

- 自动化（`.venv/Scripts/python.exe`，标准库 `unittest`）：
  - `d=(1-cos)/2` 与 `fwd`/`rev`/`sym` 逐像素复算；
  - DW 包读取：瓦片枚举、标签形状/取值、图例 9 类、`candidate_mask.sum() == core_pixels`、manifest 字段；
  - 掩膜回退链（DW `valid_mask`、`observations`、全 True）与旧包 `distance_maps` 缺失错误；
  - 整瓦片距离统计逐项复算（count、mean、median、p90、p95、max、超阈值比例）；
  - 坐标解析：旧包 `parent`、DW `region_id`、`V01` 无坐标；
  - 页面可访问性：启动服务后 `GET /`、`/aef`、`/dw` 均返回 200。
- `py_compile` 覆盖入口与 `app/webui/` 全部模块。
- 浏览器实测（本机 Chrome）：
  - 三页导航与选中态；
  - AEF 页回归自查：三种组合类型、瓦片/通道/帧切换、阈值与显示模式、common_mask、坐标、三栏滚动、旋转、缺图提示、非法数据根回滚；
  - DW 页：切换 6 个瓦片与两个传感器、阈值拖动、色阶模式/档数/边界线/显示模式四项显示控件、`candidate_mask`/bbox 叠加、预览图、数据根切到旧包时的提示；
  - 截图核对官方配色是否与 `dynamic_world_legend.json` 一致、页面无准确率/精度措辞。
- 只读校验：运行前后对 `data/` 与 DW 数据包做路径+大小聚合哈希比对。
