# ABot-sub-2000-clips 数据格式

写给要照着这批数据设计读取接口的人。**这里写的每一条都是从产出代码里核出来的**，
包含反解公式的数值验证；凡是有损、不可逆、可能缺失的地方都单独标了出来，因为那些是
接口设计里唯一会出错的地方。

上游长段的格式见 `DATA_F.md`，管线怎么跑见 `RUNBOOK.md` 第 8 节。本文只讲短片。

```
根目录  /data/binghe/datasets/ABot-sub-2000-clips
规模    9985 片（不是 10000，原因见文末「已知缺口」）
来源    2000 条 ABot-World-Explorer episode，每条切 5 片
每片    124 帧 @ 24 fps，恰好一个 CWM 窗口
```

**124 不是挑出来的，是 H3 那条推理路径的固定几何**（`cwm_h3_inference/constants.py`
开头就写着 frozen geometry）：一个窗口 124 帧，续跑重叠 34 帧，每续一次多 90 帧新的，
所以 n 个窗口是 `124 + 90 * (n - 1)` 帧。切成正好一个窗口，是为了让**一片就是一个
训练样本**，下游不用再开窗——一旦一片是 1.5 个窗口，切窗的规则就得在每个 loader 里
各写一遍，而它们不会写得一样。第 0 帧是给定的初始条件，见第 2 节 `anchor.png`。

---

## 1. 目录结构

```
ABot-sub-2000-clips/
    clip_000000_0/                    # clip_<六位 episode 号>_<片号 0..4>
        target/
            rgb.mp4                   # 1344x768, 124 帧, 24 fps, H.264 yuv420p
            anchor.png                # 1344x768, 无损 PNG, 就是 rgb.mp4 第 0 帧
        proxy/
            duv.mp4                   # 336x192, 124 帧, 24 fps, H.264 RGB 无损
        annotations/                  # 可能不存在，见第 5 节
            action.json
            caption.json
            cameras.npz
            prompt.json               # 逐秒 caption，另一条管线写的，见第 5 节
        clip_report.json              # 这一片的全部元数据
    clip_000000_1/
    ...
    clips_manifest.<i>-of-112.json    # 112 份分片清单，没有总清单
    logs/
    audit.json
```

命名可以直接解析：`clip_000414_2` 是第 414 条 episode 的第 3 片。同一 episode 的 5 片
互不重叠，按时间顺序分布在整条 episode 上。

**没有单一的总清单文件。** 跑的时候分了 112 个 shard，各写各的
`clips_manifest.<i>-of-112.json`。要建索引就直接遍历 `clip_*/clip_report.json`——工具链
本身也是这么做的（`clips-audit` 扫盘，从不读清单）。

---

## 2. target/rgb.mp4

| | |
| --- | --- |
| 分辨率 | 1344 x 768 |
| 帧数 | 124 |
| 帧率 | 24 fps |
| 编码 | H.264 (`libx264`)，`yuv420p`，CRF 16 |
| 色彩 | 解码出来是 BGR（OpenCV）或 RGB（按你的解码器） |

**有损。** 这是全流程里唯一允许有损的一条流，因为它装的是照片，8-bit YUV 加色度
下采样是下游都预期的格式。另外三种流装的是"伪装成像素的数字"，都是位精确的。

**画面来源是交付长段的 1280x720 彩色帧，用 Lanczos4 上采到 1344x768。** 也就是说
相对 1920x1080 原片，它经过了 `1920x1080 →(INTER_AREA)→ 1280x720 →(Lanczos4)→ 1344x768`
两次重采样，第二次是上采，补不回第一次丢掉的细节。这是刻意的取舍：换来的是第 6 节那条
逐像素对齐保证。（新增的 `clip-episodes` 一趟管线只缩一次，但**这批 subset 不是那么产的**。）

**1344x768 的宽高比是 1.75，不是源片的 16:9。** 横向压了 1.6%，target 和 DUV 压得
一模一样，所以两者仍然描述同一批像素。要还原真实几何比例的话，横向拉伸 1.016 倍。

### target/anchor.png

1344x768 无损 PNG，内容就是 `rgb.mp4` 的第 0 帧。单独存一份是因为它要走无损路径进
VAE，而 rgb.mp4 是有损的——**两者不是逐像素相同的**，anchor 是权威版本。

它单独存在还有个更具体的理由：**它就是模型的初始条件，而且是一帧，不是几帧。**
`cwm_h3_inference` 把这张图编成一个时间长度为 1 的 latent，放在视频 latent 的时间
索引 0 上，denoise mask 记的是 `0_fixed_1_regenerate` —— 索引 0 冻住，其余全部重新
生成。所以一片 124 帧里，**第 0 帧是给定的，第 1..123 帧是要生成的**。

这只对窗口 0 成立。多窗口续跑（`Retake34`）时，下一个窗口的初始条件是**上一个窗口
最后 34 帧**重新编码出来的前 10 个 latent，被按住不去噪。切片语料每片只有一个窗口，
所以走的是前一种。

`duv.mp4` 不是初始条件，它是**整个 124 帧的控制信号**，别把两者混成一个概念。

---

## 3. proxy/duv.mp4 —— 唯一需要解码逻辑的文件

| | |
| --- | --- |
| 分辨率 | 336 x 192 |
| 帧数 | 124（与 target 逐帧对应） |
| 帧率 | 24 fps |
| 编码 | H.264 (`libx264rgb`)，`rgb24`，**CRF 0 无损** |

无损是硬要求：R 通道是深度码，GB 是类别码，任何色度下采样都会把相邻的码平均成一个
从没预测过的值。读的时候务必确认你的解码器给出的是 RGB 而不是 BGR。

### R 通道：对数深度

```
near = 0.1 m      far = 8000.0 m      max_code = 254      sky_code = 255
```

编码是 `code = round(254 * ln(z/0.1) / ln(8000/0.1))`，近处是 0，远处是 254。

**反解：**

```python
import numpy as np

NEAR, FAR, TOP = 0.1, 8000.0, 254.0

def duv_depth_metres(red: np.ndarray) -> np.ndarray:
    """R 通道 -> 米。天空/无效处返回 nan。"""
    code = red.astype(np.float64)
    z = NEAR * (FAR / NEAR) ** (code / TOP)
    return np.where(code >= 255, np.nan, z)
```

数值验证过（编码再反解）：

| 真实米数 | R 码 | 反解 | 相对误差 |
| --- | --- | --- | --- |
| 0.1 | 0 | 0.1000 | 0 |
| 1.0 | 52 | 1.0087 | 0.9% |
| 5.0 | 88 | 4.9970 | 0.06% |
| 50.0 | 140 | 50.407 | 0.8% |
| 500.0 | 192 | 508.48 | 1.7% |
| 8000.0 | 254 | 8000.0 | 0 |

误差上界是量化步长决定的：254 个码位覆盖 80000 倍的动程，每档 `ln(80000)/254 = 4.4%`，
所以最坏相对误差约 ±2.2%。**这是对数量化，不是均匀量化**——按线性反解会错得离谱。

`code == 255` 是天空哨兵，同时也覆盖深度无效（≤ 1e-3 m）的像素。它不是"8000 米"，
要当缺失值处理。`clip_report.json` 里的 `duv_depth_inverted` 这批全是 `false`；真遇到
`true` 的话近远方向相反。

**深度是绝对米数**（backend 原生 metric），但**不能和 `cameras.npz` 的位姿混用**，
理由见第 5 节。

### G/B 通道：类别 —— 注意这里不可逆

11 类语义按下表编成 (G, B)：

| id | 类别 | G | B |
| --- | --- | --- | --- |
| 0 | sky | 255 | 255 |
| 1 | player（主角） | 0 | 255 |
| 2 | ped（其他行人） | 0 | 128 |
| 3 | vehicle | 64 | 0 |
| — | vehicle（主角在驾驶时） | 128 | 0 |
| 4 | building | 0 | 0 |
| 5 | road | 255 | 255 |
| 6 | ground | 0 | 0 |
| 7 | vegetation | 255 | 0 |
| 8 | terrain | 0 | 0 |
| 9 | water | 0 | 0 |
| 10 | prop | 0 | 0 |

**这张表不是单射的，11 类没法从 DUV 还原。** 设计接口时必须知道：

- `building` / `ground` / `terrain` / `water` / `prop` 五类共用 `(0, 0)`，读回来只能
  合成一个"静态其他"。
- `sky` 和 `road` 共用 `(255, 255)`，靠 R 通道区分：R == 255 是天空，R < 255 是路面。
  **但这个区分不是完全可靠的**：R == 255 的真实含义是"天空**或**深度无效"，所以一个
  深度没解出来的路面像素编码出来是 `(255, 255, 255)`，和天空一字不差。实测确认过。
  路面深度失败一般出现在远处或反光处，占比很小，但如果你的下游对天空掩码敏感，这是
  已知的污染源。其他类别不受影响——`vehicle` 深度无效时是 `(255, 64, 0)`，R 表示深度
  未知而 GB 仍然认得出类别。

所以能可靠区分的是 **8 组**：

```python
def duv_classes(frame: np.ndarray) -> np.ndarray:
    """DUV RGB 帧 -> 8 组的组号。frame 是 (H, W, 3) uint8 RGB。"""
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    out = np.zeros(r.shape, np.uint8)          # 0 = static other
    gb = (g.astype(np.uint16) << 8) | b
    out[gb == (255 << 8 | 255)] = 1            # road
    out[(gb == (255 << 8 | 255)) & (r == 255)] = 2   # sky
    out[gb == (0 << 8 | 255)] = 3              # player
    out[gb == (0 << 8 | 128)] = 4              # ped
    out[gb == (64 << 8 | 0)] = 5               # vehicle
    out[gb == (128 << 8 | 0)] = 6              # ego vehicle
    out[gb == (255 << 8 | 0)] = 7              # vegetation
    return out
```

要真正的 11 类逐帧 id，只能回到长段的 `frames/semantic/*.npy`（uint8 数组，见
`DATA_F.md`）——但**这批短片没有带那个**。

`player` 和 `ped` 的区分来自主角追踪器，它有可能判不出来。每片的
`clip_report.json → semantic.hero_split.resolved` 说了这一片判没判出来；判不出来时
所有人都会落在 `ped`。

---

## 4. 抽帧与 source_ordinals

源片是 30 fps，短片是 24 fps，**用抽帧实现，不是改标签**。输出第 k 帧取的是源片第
`round(k * 30/24)` 帧——五取四。所以 124 帧横跨源片 155 帧，画面速度和录制时一致。

`clip_report.json → source_ordinals` 是长度 124 的整数列表，逐帧记了它在原 episode
里的帧号。相邻差值在 1 和 2 之间跳，平均 1.252（不是 1.25——帧号只能取整）。

要把短片和 episode 级别的任何东西对齐，都用这个列表，别去重算。

---

## 5. annotations/

**整个目录可能不存在**，取决于源 episode 有没有 `annotations.tar`。三个文件各自也
可能单独缺席。`clip_report.json → annotations` 记录了实际写了什么。

那三个是**语料自己的说法**，逐字节或逐帧地从 `annotations.tar` 裁下来的。
`prompt.json` 不是——它是本仓库另一条管线事后写进来的，第 5.4 节单独讲。混在一个
目录里是因为「关于这一片已知什么」的东西读的人只会去一个地方找；要分清哪半边是谁
说的，看 `clip_report.json → annotations`（只记前三个）和 `prompt.json → provenance`。

### action.json

按帧切好的，只含这一片的 124 帧，顺序与 `source_ordinals` 一致。原文件如果是个 list
就切成 124 项的 list；如果是 `{"actions": [...]}` 这种包了一层的，保留外层结构只切
里面的 list。长度对不上帧数的字段会原样保留（说明它不是逐帧的）。

### caption.json

**整段级，原样复制，没有切。** 一条 episode 的描述对它其中 5 秒仍然成立，但要清楚
它描述的是整条 60 秒而不是这 124 帧。`clip_report.json` 里标了
`"caption": "episode-level, copied whole"`。

### cameras.npz

从 episode 的 COLMAP 稀疏重建里抽出这一片的位姿。字段：

| 键 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `cam2world` | (N, 4, 4) float64 | 相机到世界 |
| `intrinsics` | (3, 3) 或 (N, 3, 3) float64 | 内参 |
| `metric` | 标量 bool，**恒为 False** | 见下 |
| `source_ordinals` | (N,) int64 | 这 N 个位姿各自对应的源帧号 |

两个坑：

**N 不一定等于 124。** 稀疏重建会丢掉解不出来的帧，所以位姿是稀疏的。**必须**用
`source_ordinals` 去和视频帧对齐，不能假设第 i 个位姿就是第 i 帧。整片一个位姿都没有
时这个文件不会写。

**`metric` 恒为 False，位姿不是米制的。** COLMAP 只把场景定到一个相似变换，尺度未知。
所以 **DUV 的米制深度和这里的位姿不在同一个尺度上，直接混用是错的**，除非你自己解出
那个尺度因子——本管线没有解。原始 COLMAP 模型的路径记在 `clip_report.json →
annotations.source`，需要完整重建就去那里取。

### 5.4 prompt.json —— 逐秒 caption

**不是语料带的，是 `clip-prompts/` 用 VLM 写的。** 格式是 `contract: "timeline"`
v4，完整说明在 `clip-prompts/README.md`，这里只讲读它要知道的。

跟另外三个的区别，写接口时会踩到的有三条：

**它的有无和 `annotations.tar` 无关。** 那三个文件缺席是因为源 episode 没带标注；
`prompt.json` 缺席是因为这一片还没 caption 过，或者 caption 失败了。所以
`annotations/` 目录可能只有 `prompt.json` 一个文件——`clip-prompts` 在目录不存在时
会建它。反过来，`clip_report.json → annotations` 里**不会**出现 `prompt.json`，那个
字段记的是从 tar 里裁了什么。

**它描述的是这 124 帧，不是整条 episode。** 这一点上它和 `caption.json` 正好相反：
`caption.json` 是整段级原样复制的（见 5.2），拿它当这 5 秒的描述会系统性地跑偏。

**时间轴按 1 秒切，不是按帧。** 124 帧 @ 24fps 是 5.1667 秒，所以是 **5 个 bin 而不
是 6 个**：最后 0.1667 秒只有 4 帧，不够描述任何东西，并进前一个 bin，于是末尾那个
bin 是 1.1667 秒。尾巴既不四舍五入也不丢掉，`timeline.bins[]` 里同时写了它真实的秒
区间和帧区间：

```python
{"index": 4, "t": [4.0, 5.167], "frames": [96, 124]}
```

**要逐帧对齐就读 `frames`，不要拿 `t` 乘 fps 再取整**——那是第 4 节
`source_ordinals` 那条规矩的同一件事。

顶层字段：

| 字段 | 说明 |
| --- | --- |
| `contract` / `version` | `"timeline"` / `4`。不匹配就直接拒，别猜 |
| `compiler` | 渲染文本的规则版本。文本变了它变，可以据此挑片重编 |
| `window` | 这一片是谁的哪一段；切窗后还有 `t0` / `frame_offset` |
| `timeline` | bin 网格，秒和帧两套边界 |
| `scene` / `entities` / `events` | 结构化的观察，`events` 分 `subject` / `camera` 两个 channel |
| `evidence` | **从这一片自己的 `duv.mp4` 实测的**逐秒事实，没有模型参与 |
| `checks` | `evidence` 和 caption 对不上的地方，分 `fail` / `warn` |
| `compiled` | 拿去训练的文本：`lean` / `rich` / `timed` / `conditioning` |
| `provenance` | 模型、后端、重试次数、token、`score`、时间 |

**要过滤语料就看 `checks.fail` 和 `provenance.score`。** `fail` 非空表示 caption 说
了 DUV 直接打脸的东西（凭空的车、不在场的主角），那是幻觉；`warn` 是有无辜解释的
分歧，不该为它丢片。`score` 是「平均置信度扣掉警告」，不是标定过的概率，只用来排序。

**`compiled.conditioning.card` 是个指针，不是文本。** 描述 DUV 编码含义的那段话是整
个语料共用的一句，存在 `clip_prompts/conditioning.py` 里，每片只留 id；逐片不同的那
半句在 `contents` 里。要拼回完整段落用 `conditioning.full_text()`。

**默认的时间戳写法和 H3 自己的一样。** `code-world-model/examples/` 里那份多窗口配置
的窗口 prompt 就是 `"[0.00s-5.17s] ..."` 和 `"[3.75s-8.92s] ..."`——两位小数，时间是
**整段输出的绝对时间**，不是窗口内相对时间。所以：

```
[0.00s-1.00s] ...
[1.00s-2.00s] ...
[4.00s-5.17s] ...       ← 五行的并集正好是上游那个 [0.00s-5.17s]
```

逐秒 caption 是在**把 H3 已经见过的约定切细**，不是教它一套新记号。`script` 会把
`window.t0` 加进去（`timed.offset` 记着加了多少），而 `timed.bins[].t` 保持片内相对，
结构化的那份才是真值。换别的标记是 `render.script(marker=...)` 一个参数，不用重跑
VLM。

`compiled.lean.global` / `rich.global` 是**不带时间戳**的整段描述，建议按比例混进
训练集，免得模型变成没有时间戳就不会写。

**真正喂给模型的那一句见 `clip-prompts/CWM_TEXT_EXPORT.md`。** `prompt.json` 是结构，
训练用的用户句是它的一个确定性投影——`compiled.cwm.user`，以及导出到
`<clip>/prompt.txt` 的那份。规则（整窗时间戳、CRLF、w0/wn 怎么选）都在那份合同里，
这里不复述，免得两处说法漂移。

还有一条来自 `INFERENCE.md`：**prompt 在进 Qwen 之前行尾会被规范成 CRLF**。导出训练
样本时按同一条规矩来，否则编辑器把 CRLF 存成 LF 就会改掉实际消费的 token。

### 5.5 从 caption 到 VAE 打包

`prompt.json` 不直接进训练。链路是三段，**每一段的产物都是下一段的输入，中间不需要
改任何代码**：

```
annotations/prompt.json     结构（VLM 产出，贵，只做一次）
        │  clip-prompts captions-export --write-txt
        ▼
<clip>/prompt.txt           用户句（确定性投影，随时可重来）
        │  FastVideo clip_dir_to_encode_manifest.py
        ▼
h3_abot_train.jsonl         encode manifest → VAE / text 打包
```

**`prompt.txt` 放在片根目录，不在 `annotations/` 下。** 这是整个 clip-prompts 里唯一
一处不放 `annotations/` 的文件，因为路径是消费方定的：`clip_dir_to_encode_manifest.py`
的 `read_prompt()` 只看 `<clip>/prompt.txt`。

manifest 每行长这样，`prompt` 字段就是 `prompt.txt` 原文：

```json
{"name": "clip_000414_2",
 "target": "clip_000414_2/target/rgb.mp4",
 "proxy_duv_video": "clip_000414_2/proxy/duv.mp4",
 "anchor": "clip_000414_2/target/anchor.png",
 "prompt": "[0.00s-5.17s] Third-person open-world video game. City street. ...",
 "id": "clip_000414_2"}
```

三件容易出事的：

1. **一定要加 `--no-episode-caption`。** 没有 `prompt.txt` 时 `read_prompt()` 会回落到
   `annotations/caption.json`——那是**整条 60 秒 episode** 的文案，喂给 5.17 秒的窗口
   等于教模型在 5 秒里演完一整集。跑完 manifest 要看输出里那行
   `prompt sources:`，应该是 `prompt.txt=N` 而没有 `caption.json=`。
2. **切分按 episode，不按 clip。** 同一条 episode 的 5 片共享天气、光照、地形，按片切
   会拿训练过的画面做验证。`--val-episodes` 已经是按 episode 切的，别自己改成按片。
3. **文本变了要重编 text embedding，VAE latent 不用动。** 这两个 cache 是分开的。

单行的默认变体在盘上根本没有换行，所以 `read_text()` 的 universal newlines 影响不到
它。**只有 `--style timed` 那种多行变体**，CRLF 会在读取时被还原成 LF，需要训练侧在
送进 Qwen 前再规范一次（`CWM_TEXT_EXPORT.md` 第 10 节第 3 步）。

---

## 6. 几何对齐保证

**一个 DUV 像素恰好等于 target 的一个 4x4 块。** 不是近似，是严格的：

```
DUV 像素 (x, y)  <->  target 像素 (4x .. 4x+3, 4y .. 4y+3)
1344 / 336 = 4        768 / 192 = 4
```

这条保证来自 DUV 的产生方式：**它是从深度和语义数组重新合成的，不是把大图缩小的。**
深度按中位数归约，标签按多数投票，然后再编码成 DUV。如果直接对合成好的 DUV 图做插值，
就会把互不相干的深度码平均掉、把类别涂成从没预测过的颜色。

所以 DUV 里出现的每个值都是真实出现过的值。第 3 节那个 8 组解码函数不需要做任何容错
匹配——调色板之外的颜色不会出现。

---

## 7. clip_report.json

每片一份，是这一片的权威元数据。字段：

| 字段 | 例 | 说明 |
| --- | --- | --- |
| `clip` | `"clip_000414_2"` | 目录名 |
| `scene` | `"seg_000414"` | 交付长段名 |
| `sample_id` | | 原始 ABot 样本 id |
| `source_video` | | 原 episode 视频路径 |
| `window` | `2` | 片号 0..4 |
| `source_ordinals` | `[368, 369, 371, ...]` | 124 个源帧号 |
| `source_fps` | `30.0` | 源帧率 |
| `frames` | `124` | |
| `fps` | `24.0` | |
| `target_size` | `[1344, 768]` | |
| `target_from` | `"delivered frames"` | 这批全是这个值 |
| `duv_size` | `[336, 192]` | |
| `duv_depth_inverted` | `false` | 这批全是 false |
| `taxonomy` | `"standard11"` | |
| `deliverable` | `true` | false 表示是占位后端产的假数据，不可训练 |
| `annotations` | | 实际写了哪些标注 |

`clip-episodes` 那条一趟管线产出的报告多两个字段（`route: "source"`、`halo_frames`），
并且带每片自己的 `depth` / `semantic` 诊断块。**这批 subset 是两步产的，没有那些字段**，
读的时候用 `.get()`。

---

## 8. 最小读取示例

```python
import json
from pathlib import Path
import cv2, numpy as np

def load_clip(clip_dir: Path):
    report = json.loads((clip_dir / "clip_report.json").read_text())

    def frames_of(path, rgb=True):
        cap, out = cv2.VideoCapture(str(path)), []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(bgr[:, :, ::-1] if rgb else bgr)
        cap.release()
        return np.stack(out)

    target = frames_of(clip_dir / "target" / "rgb.mp4")        # (124, 768, 1344, 3)
    duv = frames_of(clip_dir / "proxy" / "duv.mp4")            # (124, 192, 336, 3)
    anchor = cv2.imread(str(clip_dir / "target" / "anchor.png"))[:, :, ::-1]

    assert len(target) == len(duv) == report["frames"] == 124
    return {
        "target": target,
        "anchor": anchor,
        "depth_metres": duv_depth_metres(duv[..., 0]),         # (124, 192, 336)
        "classes": np.stack([duv_classes(f) for f in duv]),    # (124, 192, 336)
        "source_ordinals": report["source_ordinals"],
        "report": report,
    }
```

---

## 9. 已知缺口与边界情况

按写接口时踩到的可能性排序：

1. **是 9985 片，不是 10000。** 三条 episode 太短装不下 5 片（`seg_000414` 709 帧、
   `seg_000528` 740 帧、`seg_001829` 758 帧；门槛是 775 帧）。这三条**一片都没有**，
   不是残缺，是整条跳过。别假设 `clip_<scene>_0..4` 一定齐全，遍历目录来发现。

2. **`annotations/` 及其中任一文件都可能缺席。** 见第 5 节。

3. **`cameras.npz` 里的位姿是稀疏且非米制的。** 见第 5 节，这是最容易被误用的一处。

4. **DUV 的 11 类不可逆，只能可靠还原 8 组**，而且 `sky` / `road` 的区分会被深度
   无效的像素污染。见第 3 节。

5. **`anchor.png` 与 `rgb.mp4` 第 0 帧不是逐像素相同**（一个无损一个有损）。要无损就
   用 anchor。

6. **1344x768 是 1.75 宽高比**，相对源片横向压了 1.6%。

7. **`caption.json` 描述的是整条 60 秒 episode**，不是这 124 帧。要这 124 帧自己的
   描述用 `prompt.json`（第 5.4 节），但它不是语料带的，而且不保证每片都有。

8. **主角（`player`）可能没判出来**，那一片的所有人会落进 `ped`。查
   `semantic.hero_split.resolved`——但那个字段只有一趟管线产出的报告里才有，两步产的
   这批要回长段的 `extraction_report.json` 去看。

9. **没有总清单文件**，只有 112 份分片清单。见第 1 节。
