# ABot 720p 数据管线：实现现状（审阅稿）

截至 2026-10-10 17:30（UTC+8）。描述的是**已经实现并在跑的东西**，数字来自 pod 上的实际产出；
还没做的步骤单独列出，不写成已完成。操作命令见 `FASTVIDEO_TRAINING_DATA.md` 第 11 节，
这里只讲每一步怎么做、为什么这样做、现在到了哪里、有什么问题。

---

## 0. 总览

```text
ABot-World-Explorer-subset2000（2000 个 episode，1920×1080，30 fps，60 s）
  │  proxy-extract 切片（GPU，4 节点 × 8 H200）
  ▼
ABot-sub-2000-clips-moge3-720p/clip_XXXXXX_K/      每 episode 5 片，每片 124 帧 @ 24 fps，1280×720
  ├─ target/rgb.mp4, anchor.png                   RGB
  ├─ duv/NNNNNN.depth.f32 + .semantic_id.png      逐帧米制深度 + CWM12 语义（训练用这个）
  ├─ proxy/duv.mp4                                兼容/预览用的合成 DUV 视频
  ├─ annotations/                                 episode 原始 caption、动作、相机；质量结论
  └─ clip_report.json                             来源帧、所有设置、统计；最后写，标志完成
  │  clip-prompts quality-judge（CPU + Vertex Gemini）
  ▼
annotations/quality.json → quality_audit.json      VLM 打分，接收 / 拒绝
  │  人工复核（工作台）→ quality_override.json
  │  caption 复用 + 复核导出 → prompt.txt               【未做】
  │  clips-audit / proxy-duv-manifest / proxy-duv-audit 【未做】
  │  按 episode 切分 train / val                        【未做】
  ▼
FastVideo H3 omni cache（1280×704 中心裁剪）          【未做】
  ▼
SolarWM Omni H3 LoRA 训练                            【未做】
```

| 阶段 | 状态 |
|---|---|
| 切片 | 主跑完成（9,217 片）；补切 768 片进行中（rank 3，预计 19:00 左右完成） |
| VLM 质量门禁 | 9,425 片已评，随切随评 |
| 工作台 | 运行中；内网访问待开发机凭据 |
| caption 复用与导出 | 未做 |
| DUV 审计、manifest、切分 | 未做 |
| H3 omni 编码、训练 | 未做；编码配置待定（见第 9 节） |

---

## 1. 输入

`/data/binghe/datasets/ABot-World-Explorer-subset2000/data/<xx>/<hash>/`，每个 episode 一个
`video.mp4` 和一个 `annotations.tar`。抽查的 episode 是 1920×1080、30 fps、60 s（1800 帧）。
`annotations.tar` 里有 episode 级 caption（约 60 秒的描述）、动作和相机轨迹。

---

## 2. 切片（`proxy-extract`，`clips.cut_episode`）

每个 episode 独立处理，一个 GPU 进程一次处理一片（一个窗口），中间文件写 pod 本地盘
（`WORK_ROOT=/workspace/clip-work`），只把成品写到 `/data`。

### 2.1 取哪些帧

- 每个 episode 均分成 5 段，每段正中取一个窗口，所以 5 片互不重叠、彼此间隔约 12 秒；
  位置只由 episode 长度决定，重跑得到完全相同的片。
- 每片 124 帧 @ 24 fps（5.17 s）。源是 30 fps，**靠丢帧换算而不是改标签**：输出第 k 帧取源帧
  `round(k × 30/24)`，每 5 帧丢 1 帧，一片覆盖 155 个源帧。直接拿 124 个连续源帧标成 24 fps
  会让每片都是 1.25 倍慢动作，单帧看不出来，对学动态是致命的。
- 窗口两端各多预测 2 帧（halo），时序稳定化时首尾帧也有完整上下文，之后丢掉。
- 实际用到的每个源帧序号记录在 `clip_report.json` 的 `source_ordinals` 里。

### 2.2 RGB

源 1920×1080 一次缩放到 **1280×720**（精确 2/3，不裁剪），H.264 yuv420p；`anchor.png` 是第 0 帧
的无损 PNG。

### 2.3 深度：MoGe-3

- 模型 `Ruicheng/moge-3-vitl`，fp32，`resolution_level=9`，`refine_steps=3`，输出米制深度。
- **视场角每片只估一次**：从 8 个探测帧解出 FOV 后固定给整片所有帧（例：84.6°，探测帧间
  差 2.6°）。逐帧重估 FOV 会让整张深度图每帧一起缩放几个百分点，这是单目视频深度闪烁的
  主因之一。
- **米制尺度锁定**：`temporal.lock_depth_scale` 用光流对齐的 ±12 帧邻域，只去掉尺度的高频抖动，
  保留整片的绝对水平（例：去掉 3.75% 抖动，残余 RMS 1.19%）。刻意不做逐片归一化：那样每帧都
  合法，但片与片之间的米制标度不可比。
- 天空（语义 = sky）处的深度强制为 0。
- 产出 `duv/NNNNNN.depth.f32`：1280×720 little-endian float32，米，正 view-z，0 = 无效/天空，
  每帧 3.69 MB。

### 2.4 语义：Mask2Former + SAM 2

1. **主干**：`facebook/mask2former-swin-large-ade-semantic`（ADE20K 150 类，bf16）。用闭集模型
   是因为 CWM12 的一半类别（sky、water、terrain、road、vegetation、building）是没有实例的
   "stuff"，ADE20K 覆盖得最密。ADE 类按**类名**（不是下标）映射到 standard11，换 checkpoint
   时下标错位会直接报错而不是悄悄污染数据；映射不到的 ADE 类记录在报告里。
2. **主角 / 路人拆分**：分割器分不出主角和路人。第三人称相机绑在主角身上，主角始终在画面固定
   锚点附近，所以按人体轨迹离锚点的距离挑出 player，其余为 ped。挑不出时（例如主角只在 22%
   的帧里出现）记 `hero_split.resolved=false`，不硬猜。
3. **SAM 2.1（Hiera-L）做时序一致**：SAM 2 不分类，只负责"哪些像素和上一帧是同一块表面"。
   在第 0 帧把主干标签图里每个足够大的连通块作为提示，SAM 2 沿视频传播；每 24 帧把主干认为有、
   但没被任何 masklet 覆盖的区域补种（新入画的车由此被跟上）；最后**每个 masklet 在整片上
   投一次票**决定类别，而不是逐帧决定。这是闪烁真正被消掉的原因，不是事后平滑。
   报告里 `flicker_before / flicker_after` 记录了前后的闪烁率。
4. SAM 2 必须带编译好的 `sam2._C`（mask 补洞）；缺失时 SAM 2 只打 warning 就跳过补洞，所以
   refiner 和启动器 preflight 都会直接拒绝。之前没有 `_C` 时切出的 768 片已删掉重切。
5. 最后投影到 **CWM12**：void_unknown, sky, water, terrain, road_paved, vegetation,
   building_structure, infrastructure, human, animal, vehicle, prop。产出
   `duv/NNNNNN.semantic_id.png`（1280×720，L 模式，值即类别 id）。

### 2.5 DUV 两种形态

- **`duv/`（训练用）**：上面的逐帧 depth.f32 + semantic_id.png，与 RGB 同网格、逐帧对齐。
  原分辨率保存，这样只做裁剪的消费者（H3 裁到 1280×704）和不裁剪的（Wan 等直接用 1280×720）
  都能用同一份。根目录 `semantic.json` 记录这个语料唯一的类别 → U/V 映射和网格。
- **`proxy/duv.mp4`（兼容 / 预览）**：1280×720 合成视频，R = log 深度码（0.1–256 m，8 bit），
  G/B = 类别色。训练不读它；VLM 审片视频的深度格取自它（见 3.4）。
- 合成时深度按中值、标签按多数票降采样（原分辨率时不降），每个像素都是真实出现过的值，
  不会插值出不存在的深度码或类别。

### 2.6 其余产物

- `annotations/caption.json`：episode 级原始 caption 整份复制（描述 60 秒，**不能**当这 5 秒的
  训练文本）；`action.json`：动作按窗口切出；`cameras.npz`：124 帧全部配准的相机。
- `clip_report.json`：来源、窗口、`source_ordinals`、几何、所有后端设置和统计。**最后原子写入**，
  所以它存在就代表这片完整；resume 和质量评审都以它为准。resume 还会核对后端、refiner、
  DUV 网格是否与本次一致，不一致的片会重切而不是被跳过。

### 2.7 运行方式与吞吐

- 4 节点 × 8 H200，每卡 10 个进程（共 320 个分片），每进程 2 个 CPU 线程。
- 环境是 SolarWM 镜像上单独建的 `/opt/fv-venv`（Python 3.12，torch 2.12 cu130），用 `env.sh`
  把 venv 自带的 cuDNN 排到系统 cuDNN 9.1 前面；`sam2._C` 用 nvcc 13.0 现编。
- 中间文件从 GCS 挂载移到 pod 本地盘、每卡进程数 6 → 10 后，GPU 利用率从 12–82% 升到 75–100%，
  吞吐从约 26 片/分钟升到约 38 片/分钟。一万片约 4.6 TB。

---

## 3. VLM 质量门禁（`clip-prompts quality-judge`）

### 3.1 VLM 看到什么

每片渲染 `annotations/quality_panel.mp4`：2×2 拼图，每格 640×360、24 fps——RGB｜depth（turbo
色，近暖远蓝，天空黑）｜semantic（12 类固定色）｜semantic 半透明叠在 RGB 上。以 8 fps 采样发给
Gemini（`gemini-3.8-flash`，Vertex，temperature 0，JSON 输出）。prompt 里写明每种颜色的含义、
什么算闪烁、什么算错误、HUD 不计分。

### 3.2 打分与判定

- 四项 1–5 分：`depth_temporal`、`depth_accuracy`、`semantic_temporal`、`semantic_accuracy`；
  外加带起止时间的问题列表（track / kind / minor|major / 说明）、VLM 自己的 verdict 和一句结论。
- 回复格式不对时把错误告诉它重答一次。
- 判定：VLM 判 reject，或任一项 < 3，或有任一 major 问题 → **拒绝**；score = 四项均值 / 5。
  阈值和 prompt 版本（`clip_prompts.quality.v1`）写进每片的 `quality.json`。

### 3.3 不经 VLM 的本地指标

用于排序和核对 VLM，**目前不参与判定**：

- `depth_jitter`：log 深度码三帧二阶差分 |c[t+1] − 2c[t] + c[t−1]| 的中位数（1 码 ≈ 4.4% 距离），
  匀速运动相消，只剩抖动。
- `semantic_flicker`：t 帧标签与 t±3 不同、而 t±3 彼此一致的像素比例，即持续 ≤ 5 帧的闪烁；
  单调运动的边缘不计入。最初用 ±1，实测几乎全为 0（SAM 2 已消掉单帧闪烁），VLM 报的闪烁都
  持续数帧，所以改成 ±3。VLM 判 major 闪烁的片在这个指标上是 1.8%，正常片 0.2–0.6%。

### 3.4 运行

rank 0 的 `quality` 会话每 5 分钟一轮，只评已有 `clip_report.json` 的片，评过的复用（指标改版
只重算指标，不重调 VLM）。每片约 18 s，48 并发约 70 片/分钟，只用 CPU 和网络；至今无失败、
无限流。

---

## 4. 人工复核与工作台（`clip-prompts workbench`）

- 页面：按接收 / 拒绝 / 待评 / 人工筛选，score 直方图，按任一分数或指标排序；右侧播放审片视频
  （0.25×/0.5×、逐帧），列出分数、拒绝原因、问题（点击跳到对应时间）、指标和来源。
- 改判写 `annotations/quality_override.json`：当前结论、审核人、时间、备注、此前的改判历史；
  清除也记一条。人工结论优先于 VLM，切分按最终结论。
- 权限：默认只读；审核人用 `/data/binghe/secrets/workbench_reviewers.txt` 里的个人口令登录后才能
  改判（`clip-prompts workbench-token` 发放 / 轮换）。
- 访问：pod 上只监听 127.0.0.1:8765；本机经 `kubectl port-forward`，同事经云开发机
  `21.130.243.218:8765`（开发机上的 systemd 服务转发，**还差你放上 kubeconfig**）。

---

## 5. 当前数字

截至 17:30：

- 磁盘上 9,527 个 clip 目录，其中 9,519 片完整；补切本次已完成 243 片，80 个分片全部在跑，
  无报错。最终应为 9,985 片：10,000 减去约 15 片太短切不满的 episode，与旧 768p 语料的 9,985
  一致。
- 质量门禁：已评 9,425 片，**接收 6,601（70.0%），拒绝 2,824**，待评 94，人工改判 2，
  平均 score 0.633。
- 拒绝原因（一片可有多条）：VLM 判拒绝 2,797；`semantic_accuracy < 3` 2,476；
  major semantic wrong_label 2,328；`semantic_temporal < 3` 746；major semantic flicker 367；
  `depth_temporal < 3` 286；major depth flicker 195；major semantic missing_object 155。

---

## 6. 还没做的步骤

按顺序：

1. 补切完成（约 6 片/分钟，预计 19:00 左右）后，质量门禁追平。
2. **caption 复用**：从旧 768p 语料复制 `annotations/prompt.json`（只在两边 `source_ordinals`
   相同时），用新 DUV 本地复核（不调 VLM）后导出 `prompt.txt`。旧语料是 9,721 片有 caption、
   9,660 片导出。720p 语料目前 0 片有 caption。
3. `clips-audit`、`proxy-duv-manifest`、`proxy-duv-audit`（含 `.work` 残留检查）。
4. `quality-audit` → 按 episode 切分 train/val：DUV 审计通过 ∩ 文本门禁 ∩ score ≥ 0.90 ∩
   质量门禁接收。
5. FastVideo omni 编码先 1 片冒烟，再 4 节点全量；然后 SolarWM 训练。

---

## 7. 已知问题与风险

1. **拒绝几乎都来自语义错标，根因在主干映射，不在时序。** 我抽查的 3 片拒绝样本都是真问题：
   停车场标成 infrastructure、狗标成 vehicle、主角没分出来。这些是 Mask2Former ADE 类到 CWM
   类的映射和主角拆分的问题，SAM 2 只会把错标稳定地传播下去。丢掉 30% 的数据是最省事的处理；
   修主干映射后重算语义（深度不用重算）可能收回其中相当一部分。
2. **VLM 分数挤在 3–4 分。** 几乎没有 5 分，平均 score 0.633，阈值"任一项 < 3"正好切在分布中间，
   结论对阈值很敏感。我只核对过少数几片的结论，没有逐条核对 VLM 问题描述的细节。建议先在工作台人工复核一批（例如接收、拒绝各 100 片），算出人与 VLM 的
   一致率，再定是否调阈值或把本地指标加进判定。
3. **VLM 只看 8 fps。** 持续 1–2 帧的闪烁可能漏看；本地 `semantic_flicker` 能看到，但目前不参与
   判定。
4. **审片视频里的深度来自 `proxy/duv.mp4`**（8 bit log 码），而训练读 `duv/` 的 float32。两者来自
   同一份深度，只是量化不同，看闪烁和明显错误足够，但不是逐字节相同的输入。
5. **`clip_report.json` 的 `source_fps` 记错为 24**（实际 30）。抽帧步长一直用的是探测到的 30，
   clip 本身正确；下游只有 caption 把这个字段抄进元数据，没有计算用到它。代码已修
   （`1f2fc91`），但已经写出的约一万份报告仍是 24，改写它们要动 `/data` 上的文件，见第 9 节。
6. **pod 本地状态会随 pod 重建丢失**：`workbench`、`quality` 会话，`/workspace/quality-runs/` 日志。
   `/data` 上的结论文件不受影响。
7. **工作台走内网明文 HTTP**：口令防误操作、记录是谁改的，挡不住内网抓包。
8. **rank 3 补切与 eval 共用 GPU**：rank 3 同时挂着等 checkpoint 的 eval worker（`r1-eval`），第一个
   checkpoint 预计 22:30 左右；补切预计 19:00 左右完成，有余量，但需要盯着。

---

## 8. 代码位置

datapipe 仓库（`git.woa.com` origin 与 GitHub `hebing-sjtu/proxy-extract` 同步）：

- 切片：`proxy-extract/src/proxy_extract/clips.py`（窗口、组装），`depth/moge3.py`、
  `temporal.py`（FOV 与尺度锁定），`semantic/panoptic.py`、`semantic/player.py`、`semantic/sam2.py`，
  `taxonomy.py`（CWM12 与映射），`proxy_duv.py`（`duv/` 交付）；启动器 `scripts/run_clip_episodes.sh`。
- 质量门禁与工作台：`clip-prompts/src/clip_prompts/quality.py`、`workbench.py`、`workbench.html`，
  命令在 `cli.py`（`quality-judge`、`quality-audit`、`workbench`、`workbench-token`）。
- 测试：`proxy-extract/tests/`（`test_clips.py` 41 个），`clip-prompts/tests/`（186 个，含质量
  门禁和工作台 21 个）。
- 编码在 FastVideo `scripts/h3_proxy/prepare_data/encode_proxy_samples.py`；训练配置在 SolarWM。

---

## 9. 需要你决定的事项

1. **语义错标怎么处理**：直接丢弃被拒的约 30%；还是先修 Mask2Former → CWM 的映射和主角拆分，
   重算语义后重新评审。
2. **判定阈值**：保持现有规则，还是先做一轮人工抽检算一致率再调；本地闪烁指标是否加进判定。
3. **H3 omni 编码配置**：第 11.2 节写的是两路参考（depth + semantic，1280×704，
   `--proxy-references depth semantic --cwm-system w0_depth_semantic`）；今天 GTA 训练用的是之后
   加进 FastVideo 的混合参考配置（`--proxy-variants duv depth semantic style --cwm-system w0_omni`，
   参考 640×352）。ABot 要和哪一组对齐，决定了 cache 和 SolarWM 训练配置。
4. **旧报告里的 `source_fps`**：是否批量改正已写出的约一万份 `clip_report.json`（只改这一个字段，
   原子写入）。
