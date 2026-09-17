# Proxy 交付规格：从视频里抽出 depth + semantic，喂给 MiniMax-H3 proxy 训练

写给要产出这批数据的人。消费方是 `FastVideo/scripts/h3_proxy/prepare_data/encode_proxy_samples.py`，
它把交付物过一遍 VAE 和 Qwen3-VL，写成训练直接读的 `.pt` cache。**这里的每个数字都是从消费端
代码里核出来的**，不是约定俗成——凡是消费端只做断言、不做换算的地方都标了出来，因为那些是唯一
会静默出错的地方。

本仓库现有的 DUV（`DATA_F.md` 第 "DUV（后处理）" 节）**不能直接用**。那一版 R 是 near 0.1 /
far 8000 正向、255 留给天空，G/B 是语义颜色且 sky 和 road 撞码。这里要的是 **CWM 约定**：R 是
near 0.3 / far 256 反向、0 表示无效，G/B 是一张 4×3 的单射码表。两者差别全是静默的——形状、帧数、
loss 全都正常，只是训练不出来。

权威定义在 `FastVideo/fastvideo/pipelines/basic/minimax_h3/proxy.py`，常量名
`PROXY_DEPTH_NEAR_METRES`、`PROXY_SEMANTIC_U`、`PROXY_SEMANTIC_CLASSES`。本文和它冲突时以它为准。

---

## 0. 首选交付形态：逐帧，不要自己合成 DUV

消费端支持两种 proxy 形态，**请交第一种**：

| 形态 | manifest key | 谁来合成 DUV |
|---|---|---|
| **逐帧 depth + semantic**（首选） | `proxy_duv` | FastVideo，用它自己的常量 |
| 已合成的 DUV 视频 | `proxy_duv_video` | 你们 |

首选逐帧有三个理由，都不是风格问题：

1. **调色板不可能写错。** 交 `duv.mp4` 就等于把 R 的 log 曲线、G/B 的 12 个码、天空的处理方式
   全部复制一遍到你们这边。这些在消费端一律不校验、也无法校验——一张写错了范围的深度图仍然
   长得像深度图，只是它承载的视差被单调重标了一遍。
2. **VAE 能拿到完整精度。** 逐帧路径下 `read_duv_clip` 把 float 深度直接 log-normalize 成 VAE
   输入，8-bit 量化只发生在给 Qwen 看的那份预览上。视频路径下深度永远只有 8 bit，每码是固定
   2.68% 的距离比值。
3. **没有编解码风险。** 见第 5 节。

逐帧的空间代价是可算的：192×336 的 float32 深度每帧 252 KiB，124 帧一段 30.5 MiB。语义 PNG
可以忽略。

---

## 1. 逐帧格式（`proxy_duv`）

一段一个目录，里面恰好 `num_frames` 对文件，序号从 `000000` 起六位连续：

```
<seg>/duv/
├── 000000.depth.f32        无头 little-endian float32，C order，H*W*4 字节，单位米
├── 000000.semantic_id.png  8-bit 灰度（PIL mode "L"），像素值就是类 id
├── 000001.depth.f32
├── 000001.semantic_id.png
└── ... 到 000123
```

消费端 `read_raw_depth` / `read_semantic_png` 会**断言**字节数和 PNG 的 mode/尺寸，不符就报错。
所以：

- `.depth.f32` 必须正好 `192 * 336 * 4 = 258048` 字节。没有 header、没有 shape、没有压缩。
- `.semantic_id.png` 必须是 `L` 模式、`336x192`。**不要**写成 RGB 再指望读 B 通道——那是
  `DATA_F.md` 里 `semantic.mp4` 的约定，这里不是。
- 文件数必须 ≥ `num_frames`（124）。消费端按 `range(124)` 逐个开，缺一个就 ENOENT。

### depth 的语义

相机空间**正 view-z**（视线方向距离，米），不是视锥 NDC，不是 disparity。

- `0`（准确说 ≤ `1e-3`）= **没有表面**：天空、未命中、模型不确定。
- 必须有限且非负。`nan` / `inf` / 负数会让 `encode_depth` 直接 raise。
- 超出 `[0.3, 256]` 的有效值不用你们裁，消费端 clip。但落在范围外的那部分信息就没了，所以
  如果你们的场景普遍在 256 m 外，先说，这个范围是可以改的（改了要重编全部 cache）。

天空写 0 而不是写一个很大的数，是因为 0 在这套编码里读作"无穷远"，而这是安全的方向：一个虚假的
远表面会被忽略，一个虚假的近表面会遮挡整帧。

### semantic 的语义

每像素一个整数，取值必须在 `[0, 12)`。越界会 raise。12 类是 CWM 的体系：

| id | 类名 | 说明 |
|---:|---|---|
| 0 | `void_unknown` | 无法归类。**不是**天空 |
| 1 | `sky` | 天空 |
| 2 | `water` | 海、湖、河、池 |
| 3 | `terrain` | 山地、岩石、野外地面 |
| 4 | `road_paved` | 铺装路面 |
| 5 | `vegetation` | 树、草、灌木 |
| 6 | `building_structure` | 楼、墙、地标 |
| 7 | `infrastructure` | 人行道、路缘、桥、杆、护栏 |
| 8 | `human` | 人 |
| 9 | `animal` | 动物 |
| 10 | `vehicle` | 载具 |
| 11 | `prop` | 灯、椅、牌、HVAC 等；兜底类 |

从本仓库现有 11 类体系（`DATA_F.md` 第 "Semantic（序号）" 节）过来，**请逐字照抄这张表**，它就是
`compose_gta_duv.py` 里的 `GTA_TO_CWM`：

| 现有 id | 现有类 | → CWM id | CWM 类 |
|---:|---|---:|---|
| 0 | sky | 1 | sky |
| 1 | player | 8 | human |
| 2 | ped | 9 | animal |
| 3 | vehicle | 10 | vehicle |
| 4 | building | 6 | building_structure |
| 5 | road | 4 | road_paved |
| 6 | ground | 7 | infrastructure |
| 7 | vegetation | 5 | vegetation |
| 8 | terrain | 3 | terrain |
| 9 | water | 2 | water |
| 10 | prop | 11 | prop |

`ped → animal` 看着不对，是有意的：适配器是在从没见过 DUV 帧的 base Ref2VA 上从零训的，这些码和
它们指代什么之间的关联住在 CWM 的 LoRA 里，而那个 LoRA 不加载。对模型来说 `(96, 128)` 就是两个
字节，所以类别到槽位的分配是一次重标，任何单射的分配训出来都一样。真正有约束的只有三条：单射、
过 VAE 之后仍然可分、**在整个语料上稳定**。照抄这张表是为了让新语料能和 `gta_v2_*` 那批 cache
混用或对比——换一张表就换了一个实验。

如果你们的抽取模型给出的是别的类别体系（Cityscapes、ADE20K、COCO-Stuff），**把映射表写进交付
物里**，一段一份 `duv/semantic.json`，形如 `{"classes": {"0": "void_unknown", ...}}`，并且保证
全语料同一张。映射本身可以商量，映射漂移不行。

---

## 2. 深度的定标：这是模型抽取路线最容易静默毁掉的一条

R 通道是**在米制上定标**的：`0.3 m` 亮、`256 m` 暗，对数等分。这意味着一个码对应固定的距离
**比值**（2.68%），而不是画面里的相对远近。

单目深度模型（DepthAnything 这类）默认给的是**相对/仿射深度**，不是米。直接拿去归一化会出现
这种情况：同一条路在近景段被编码成 12，在远景段被编码成 200，因为每段各自归一化了。形状对、
帧数对、图像看着完全正常，但深度通道在语料上不再是同一个量——这是最贵的那类错误，因为它训得
动、也评得出分，只是学不到东西。

所以：

- **绝对不要按帧或按段做 min/max 归一化。** 一次都不行。
- 首选真米制：用带 metric 头的模型（Metric3D、UniDepth、Depth Pro 这类），或者用已知基线
  （车高、人高、标定过的双目）把相对深度标到米。
- 退而求其次：定一个**全语料固定**的线性/仿射系数把模型输出映到米，写进交付物的报告里，并且
  保证它在整个语料上是同一个常数。这时请顺手交一份诊断：全语料有效深度的分位数（p1/p50/p99）。
  如果 p99 在 256 m 外或者 p1 在 0.3 m 内，范围就该调，而不是让 clip 去吃掉。
- `annotations.tar` 里的 COLMAP **不能用来定标**。它只定义到一个相似变换，不是米制——
  `DATA_F.md` 第 "对齐" 节已经写了这条。

---

## 3. 几何：proxy 必须正好 192 × 336，target 正好 16:9

| 什么 | 值 | 谁读 |
|---|---|---|
| proxy 网格（H × W） | **192 × 336** | 逐帧文件本身的尺寸 |
| target 画布（H × W） | **768 × 1344** | 消费端会 LANCZOS 缩放，源片只要是 16:9 |

两条都不是偏好：

- **proxy 不允许重采样。** 消费端 `read_duv_video_clip` 没有 resize 分支，尺寸不符直接报错；
  逐帧路径则由 `read_raw_depth` 的字节数断言挡住。理由是 DUV 是三张整数码表穿了 RGB 的外衣，
  任何插值都会把不相邻的深度平均成第三个值、把类别边界画成分割器从没输出过的码，同时产出一张
  看起来完全正常的图。**所以深度和语义要么原生就在 192×336 抽，要么用最近邻降到 192×336。**
  最近邻是唯一能保证每个输出像素都是真被观测过的值的滤波器。
- **target latent 网格必须是 proxy 的整数倍**，因为 ControlNet 把 proxy latent 复制到 target
  的 latent 网格上。每条像素轴除以 16（VAE 的 8× 空间压缩乘 transformer 的 2×2 patch）：
  `768/16 = 48`、`1344/16 = 84`、`192/16 = 12`、`336/16 = 21`，`48/12 = 84/21 = 4`，整数。
  `704×1280` 是 44×80，对 192×336 是 3.67× 和 3.81×——**驱动不了 trunk**，这就是现有
  `gta_v2_cwm` cache 报废的原因。

两个网格的宽高比都是 1.75，所以 16:9 的源片降到 192×336 不引入形变。1280×720 的源正合适。

---

## 4. 时间轴：24 fps，每段至少 124 帧

- 消费端的时间轴固定 **24 fps**。视频路径会自动重采样（只做整帧的选取和重复，对码表安全）；
  **逐帧路径不会重采样**，所以逐帧交付的 `000000..` 必须本来就是 24 fps 的序列。
- `--num-frames` 必须满足 `n % 17 == 5`（H3 causal VAE 的约束），默认 **124**，即 5.17 秒。
  段长不足 124 帧的会被拒。
- depth、semantic、target **必须逐帧对齐**，同一次解码的同一批像素。`DATA_F.md` 已经解释过为
  什么不能把源片另外交给一次 ffmpeg 转出 target：两个 resampler 对不到像素级，而 RGB 和自己的
  depth 差半个像素的交付集，对任何学对应关系的东西都是负价值。这条对本规格同样成立。

---

## 5. 如果一定要交 `duv.mp4`

只有一种写法是 bit-exact：**`libx264rgb` + `pix_fmt rgb24` + `crf 0`**。

`crf 0` 单独不够。它只保证无损地编码**交给编码器的东西**，所以把 rgb24 帧喂给一个 `yuv444p`
的流，仍然要付一次 8 bit 无法逆的色彩矩阵。这个坑我们自己踩过：`compose_gta_duv.py` 一直用
`libx264` + `yuv444p` 而 docstring 写的是 "Lossless RGB mp4"，现在已经改成 `libx264rgb` 并且
每次写完都解码回来逐字节比对。**请照做——写完读回来比对，不一致就当失败**，而不是相信 codec 的
名字。

三个通道的字节定义（和 `proxy.py` 逐位一致）：

```
R = round( (ln(256) - ln(clip(d, 0.3, 256))) / (ln(256) - ln(0.3)) * 65535 ) / 257   四舍五入到 uint8
  = 0                                                                                d 无效 或 天空

G = (32, 96, 160, 224)[label % 4]
B = (43, 128, 213)[label // 4]
```

注意 **u 变化最快**（`label % 4` 走 U）。把网格按另一个顺序构造出来，会得到 12 个互不相同但
全都不对的码。R 先量化到 uint16 再除 257，是为了和 `load_duv_frame` 的字节完全一致，而不是
"在一个舍入步内吻合"。

天空和无效深度共用码 0 没有代价，**因为语义通道是单射的**——天空有自己的 (G, B)。

---

## 6. 其余三样交付物

| key | 是什么 | 要求 |
|---|---|---|
| `target` | 要生成的那段真实视频 | 16:9，≥124 帧，普通 mp4（有损可接受，消费端会 LANCZOS 到 768×1344） |
| `anchor` | 整段的**外观字典**，可省 | 省掉就用 target 第 0 帧。给了就必须是这段的外观来源 |
| `prompt` | 文本 | 见下 |

`anchor` 不是"第一帧的图片"，是整条 take 的外观参考——消费端按短边 2048 缩放它，占的视觉 token
是 768 画布的约 7 倍。所以它可以是一张 target 从没展示过的外观（另一种画风、一张参考照片）。
**不确定就别给这个 key**，让它回落到 target 第 0 帧；给一张和第 0 帧平均绝对差 16–49 的"风格图"
会把外观钉在错的地方。

`prompt` 如果按窗口切分（`wn` regime），需要和 `--cwm-system` 对应。这一条不影响你们的产出格式，
但会决定 cache 怎么编，所以交付时说明 prompt 是整段的还是按窗口的。

---

## 7. manifest

一段一行 JSON，路径全部**相对于 `--root`**（`--root` 是放 `seg_*/` 的那个数据集目录）：

```json
{"name": "seg_000000", "id": "seg_000000", "target": "seg_000000/video.mp4", "proxy_duv": "seg_000000/duv", "prompt": "..."}
```

- `proxy`、`proxy_duv`、`proxy_duv_video` **三者恰选其一**。
- 交视频就把 `proxy_duv` 换成 `proxy_duv_video": "seg_000000/proxy/duv.mp4"`。
- `anchor` 可省。
- 这份 manifest 也可以不用你们写——`seg_dir_to_encode_manifest.py` 能从 `seg_*/` 目录树扫出来，
  前提是目录布局固定。布局定下来之后告诉我们即可。

---

## 8. 验收自检

交付前跑一遍。这段不依赖 FastVideo，可以直接在你们的环境里跑：

```python
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image

SEG, H, W, N = Path(sys.argv[1]), 192, 336, 124
U, V = (32, 96, 160, 224), (43, 128, 213)

depths = []
for i in range(N):
    raw = (SEG / "duv" / f"{i:06d}.depth.f32").read_bytes()
    assert len(raw) == H * W * 4, f"frame {i}: {len(raw)} bytes, want {H*W*4}"
    d = np.frombuffer(raw, "<f4").reshape(H, W)
    assert np.all(np.isfinite(d)) and np.all(d >= 0), f"frame {i}: non-finite or negative depth"
    depths.append(d)

    with Image.open(SEG / "duv" / f"{i:06d}.semantic_id.png") as im:
        assert im.mode == "L" and im.size == (W, H), f"frame {i}: mode {im.mode} at {im.size}"
        ids = np.asarray(im)
    assert ids.max() < 12, f"frame {i}: class id {ids.max()} >= 12"

d = np.concatenate([x.ravel() for x in depths])
valid = d[d > 1e-3]
print(f"valid {len(valid)/len(d):.1%} of pixels")
print("metres p1/p50/p99: %.2f / %.2f / %.2f" % tuple(np.percentile(valid, [1, 50, 99])))
print("out of [0.3, 256]: %.2f%% near, %.2f%% far"
      % (100 * np.mean(valid < 0.3), 100 * np.mean(valid > 256)))
```

要看的不只是"没报错"：

- **`valid` 占比**。全 100% 说明天空没被标成无效，回去查；接近 0% 说明深度整体没写进去。
- **p1/p50/p99**。这是定标对不对的唯一证据。p99 显著超过 256 或 p1 低于 0.3，说明范围该调；
  **更要紧的是它在不同段之间该是可比的**——抽十段跑一遍，如果 p50 在段之间差一个数量级，那就是
  第 2 节说的按段归一化，整批数据作废。
- 交视频的话额外验一条：解码回来和写进去的帧逐字节相等。

---

## 9. 一句话版本

原生或最近邻降到 **192×336**、**24 fps**、每段 **≥124 帧**，每帧交一个 `258048` 字节的
little-endian float32 **米制** view-z（无效写 0）和一张 `L` 模式 336×192 的 **`[0,12)` 类 id**
PNG，类别照抄第 1 节那张映射表，深度用**全语料同一个**定标、绝不按段归一化。DUV 我们自己合成。
