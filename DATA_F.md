# ABot-seg-long-2000 数据格式

交付集的格式说明。怎么把它跑出来见 `RUNBOOK.md`。

源是 **ABot-World-Explorer-subset2000**（2000 段，Apache-2.0），只有 RGB 和一个
COLMAP 稀疏模型 —— **没有深度，也没有语义**。这两路都是本仓库预测出来的，不是
引擎通道。DUV 不是第四次预测，是同帧 depth + semantic 的合成。

| | 源 | 交付 |
|---|---|---|
| 路径 | `/data/binghe/datasets/ABot-World-Explorer-subset2000/` | `/data/binghe/datasets/ABot-seg-long-2000/` |
| 布局 | `data/<prefix>/<sample_id>/{video.mp4, annotations.tar}` | `seg_NNNNNN/` |
| 分辨率 | 1920×1080 | 1280×720（正好 2/3，不引入形变） |
| 长度 | 整段，约 1800 帧 | 同上，**不切窗、不截断** |
| 帧率 | 源帧率 | 源帧率 |

## 目录

```
<dataset_root>/
├── scenes_manifest.json          # 哪个 sample_id 变成了哪个 seg
├── seg_000000/
│   ├── frames/                   # 逐帧中间结果，也是交付物的一部分
│   │   ├── color/000000.png      #   RGB，无损
│   │   ├── depth/000000.npy      #   float16 米，0 = 无深度
│   │   ├── semantic/000000.npy   #   uint8 类别 id
│   │   └── duv/000000.png        #   RGB，无损
│   ├── proxy/                    # 由 frames/ 编出来的四路视频
│   │   ├── color.mp4             #   RGB
│   │   ├── depth.mp4             #   深度灰度
│   │   ├── semantic.mp4          #   语义 ID
│   │   └── duv.mp4               #   depth + semantic 的合成
│   ├── annotations.tar           # 源 episode 的标注，逐字节原样拷贝
│   └── extraction_report.json    # 这一段是怎么跑出来的
├── seg_000001/
└── ...
```

派生的东西全在 `frames/` 和 `proxy/` 里，源带来的东西留在外面。这样 `ls` 一眼就能
分清哪半边是数据集自己的说法、哪半边是我们的预测。

### frames/ 和 proxy/ 的关系

`proxy/*.mp4` **完全由 `frames/` 编出来**，先落逐帧、再转视频。所以两者逐帧一致，
不是两条独立的路径。这么排有两个原因：一是管线可以按帧断点续跑，worker 死在第
1700 帧不用从头再来；二是 `frames/` 保留了视频编码丢掉的精度。

四路里只有 **depth 是视频复现不出来的**：

| 流 | 逐帧 | 视频 | 逐帧是否更有信息 |
|---|---|---|---|
| depth | float16 米 | 8-bit log 灰度 | **是** —— 每码 3.1%，float16 约 0.05% |
| color | 无损 PNG | libx264 CRF 16 | 略微 |
| semantic | uint8 id | 无损 RGB 的同一批 id | 否，等价 |
| duv | 无损 PNG | 无损 RGB | 否，且可由 depth + semantic 完全推出 |

空间代价按 1280×720 算，两路数组是定长的：depth 每帧 1.758 MiB、
semantic 每帧 0.879 MiB，一段 1800 帧就是 4.6 GiB，还没算两路图像。
所以哪几路值得留是要选的，
`--keep-frames` 按流选（见 `RUNBOOK.md` 第 4 节）。跑完只剩 `proxy/` 的话用
`--keep-frames none`。

`frames/depth/*.npy` 用 `np.load` 直接读，单位是米，`0` 表示天空或未命中 —— 和
`depth.mp4` 的 `gray == 0` 是同一个约定。

`annotations.tar` 里是 `action.json`（逐帧键鼠输入）、`caption.json`，以及完整的
`sparse/0/{cameras,images,points3D}.txt` COLMAP 模型（相机内外参在这里）。**不做
任何转换**：它的内容是数据集自己的声明，重打包会让本管线变成一份它没有生产过的
数据的第二个真值来源。要读它用 `proxy_extract.datasets.abot`，它直接从 tar 里取
成员，不解包。

编号按 `sample_id` 字典序，从 `seg_000000` 起，六位。排序而非发现顺序，是为了让
编号只是输入集合的函数：分片的 worker 不用协调就能算出同样的号，补跑新增 episode
时也只会插入、不会把已交付的段重新编号。`scenes_manifest.json` 记着每段的来源，
否则重编号就把数据集自己的标识丢了。

## 对齐

四路视频**逐帧对齐**，帧数相同，尺寸相同。它们由同一次解码的同一批像素产出 ——
color 不是把源文件另外交给 ffmpeg 转出来的，因为两个 resampler 不会对到像素级，
而 RGB 和自己的 depth 差半个像素的交付集，对任何学对应关系的东西都是负价值。

`annotations.tar` 里的 COLMAP 是对 **1920×1080 源片**解的，内参要自己按 2/3 缩放
到 1280×720。它只定义到一个相似变换，**不是米制**，不能拿来给深度定标。

## Depth（米制）

相机空间 **正 view-z**（视线方向距离，米），不是视锥 NDC。

1. 距离 `d` clip 到 `[near, far] = [0.1 m, 256 m]`
2. 量化到 16-bit 再压成 8-bit 灰度（**越近越亮**）：

```
q16  = round( (ln(far) − ln(d)) / (ln(far) − ln(near)) × 65535 )
gray = 0                          若 q16 == 0（未命中 / 无效）
     = max(1, round(q16 × 255 / 65535))
```

反解米制（`gray > 0`）：

```
d_m = exp( ln(256) − (gray / 255) × (ln(256) − ln(0.1)) )
```

`gray == 0`：天空或未命中，没有有效距离。

编码名：`h264-logz-gray8`。codec 是 `libx264`，像素格式 `gray`，`crf 0`。
**解码请读灰度，不要当普通 YUV 色度用。**

一个码是 log 尺度上的固定比值，所以误差是相对的：`(256/0.1)^(1/255) ≈ 1.031`，
即每码约 3.1%。

## Semantic（序号）

每像素一个类别整数。视频是**无损 RGB**：`(R, G, B) = (0, 0, id)`，codec
`libx264rgb` + `rgb24`（`crf 0`）。**不要用 YUV H.264**，色度下采样会把相邻 id
混成一个从没被预测过的类，而且下游发现不了。

读 **B 通道**（或确认 R=G=0 后读任一通道）：

| id | 类 | 说明 |
|---:|---|---|
| 0 | sky | 未命中 / 天空 |
| 1 | player | 本段可控角色 |
| 2 | ped | NPC |
| 3 | vehicle | 载具网格（车、部分船/机） |
| 4 | building | 楼、地标 |
| 5 | road | 沥青路面 |
| 6 | ground | 地块、人行道、标线、路砖非沥青部分 |
| 7 | vegetation | 树、草、灌木、树池 |
| 8 | terrain | 山地、岩石、野外地面 |
| 9 | water | 海、湖、河、池 |
| 10 | prop | 灯、椅、牌、栏、HVAC 等；对不上上面规则时的默认类 |

只有这 11 个类，**没有实例 ID**。开车片段里自己的车和交通车在 semantic 里都是
`3`，不区分 ego —— 那个区分只出现在 DUV 的 G/B 里。

两点要知道：

- **player / ped 不是分割器分出来的**。任何类别体系下主角和路人都是同样的像素，
  分开靠的是相机怎么架 —— 第三人称相机绑在玩家身上，玩家投影在一个固定的屏幕锚点
  附近不动，别人会飘过画面。实现见 `semantic/player.py`。
- **road / ground 是功能区分，不是外观区分**。原始定义里 road 只认沥青材质，路砖上
  的水泥和标线算 ground —— 这需要场景图才判得准，从图像上只能逼近。见
  `taxonomy.py` 的 `ROAD_GROUND_IS_FUNCTIONAL`。

## DUV（后处理）

由**同帧** depth + semantic 合成，不是第三次前向。depth 和 semantic 各只预测一次，
DUV 是它们的确定性变换。

| 通道 | 含义 |
|---|---|
| **R** | 全图像素的 log-z。有效深度从 depth 灰度反解到米，再按 **near=0.1 / far=8000** 重新 log 压到 `[0, 254]`（**近 0，远 254**，方向与 depth.mp4 相反）；天空/无效 = **255** |
| **G, B** | 语义色，不占 R |

```
R = round( (ln(d) − ln(0.1)) / (ln(8000) − ln(0.1)) × 254 )     d 有效
  = 255                                                         天空 / 无效
```

| 类 | G | B | 备注 |
|---|---:|---:|---|
| player | 0 | 255 | semantic id 1 |
| ped | 0 | 128 | id 2 |
| ego | 128 | 0 | id 3 且该段是驾驶段 |
| vehicle | 64 | 0 | id 3，非驾驶段 |
| vegetation | 255 | 0 | id 7 |
| road | 255 | 255 | id 5 |
| other | 0 | 0 | building / ground / terrain / water / prop |
| sky | 255 | 255 | id 0 或 depth 无效；靠 **R=255** 和 road 区分 |

sky 和 road 的 G/B 撞在一起是原格式如此，**只能靠 R 分**：sky 的 R 恒为 255，
而有效深度最大只到 254。这也是 R 走正向、把 255 留作哨兵的原因。

编码同 semantic：`libx264rgb` + `rgb24` + `crf 0`。不无损的话 R 会被压出
254/255 之间的中间值，天空掩码就碎了。

**这跟 code-world-model 的 DUV 不是一回事**，混用会静默污染训练。CWM 的 DUV 在
R 里放的是 near 0.3 / far 256 的深度码，G/B 是调色板的 (u, v)；这里 R 是
near 0.1 / far 8000 且 255 保留给天空，G/B 是语义**颜色**。

## extraction_report.json

每段自己的运行记录：帧数、尺寸、fps、两个后端的名字、深度的编码参数和统计、
各类别像素占比、稳定化前后的 flicker、耗时。

要看的两个字段：

- `deliverable` —— `false` 表示这段是用**合成后端**跑的。合成后端伪造输出，产物
  在结构上和真的完全一样（同 codec、同类 id、同报告形状），只有看像素才认得出，
  所以事实写在报告里而不是留给人从后端名字去推。
- `frames` —— `--resume` 拿它跟四路视频实际的帧数比对。被杀在编码中途的段会留下
  长度不足的 mp4，这个比对会认出来并重跑，而不是让截断的段悄悄进数据集。
- `frames_kept` —— 这一段实际留下了哪几路逐帧目录。跑到一半改过 `--keep-frames`
  的数据集里，各段可以不一样，所以写在每段自己的报告里而不是全局记一次。

## 读取建议

留了 `frames/` 的话，depth 和 semantic 直接读数组最省事，也不掉精度：

```python
import numpy as np

metres = np.load("seg_000000/frames/depth/000000.npy").astype(np.float32)  # 0 = 无深度
ids = np.load("seg_000000/frames/semantic/000000.npy")                     # uint8
```

只有视频的话：

```python
import cv2, numpy as np

# semantic / duv：必须无损 RGB 读
cap = cv2.VideoCapture("seg_000000/proxy/semantic.mp4")
ok, bgr = cap.read()
ids = bgr[:, :, 0]          # OpenCV 是 BGR，所以 B 在 0 号通道

# depth：读灰度再反解
cap = cv2.VideoCapture("seg_000000/proxy/depth.mp4")
ok, bgr = cap.read()
gray = bgr[:, :, 0].astype(np.float64)
metres = np.exp(np.log(256) - gray / 255 * (np.log(256) - np.log(0.1)))
metres[gray == 0] = 0        # 天空 / 无效
```

仓库里的等价实现是 `proxy_extract.proxy` 的 `encode_depth_frame` /
`decode_depth_frame` / `encode_semantic_frame` / `compose_proxy_frame`。

交付格式设计上就是不能直视的：semantic 播出来近乎全黑，depth 是一片平灰。要人眼
过一遍用

```
python -m proxy_extract scenes-preview --scene <dataset_root>/seg_000000 --out /tmp/sheet.png
```
