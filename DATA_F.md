# gta-web clip 数据集

第三人称固定步长录制（契约 v3）：24 fps，1280×720，每段 **124 帧 / 5.166… 秒**。同一帧的 color / depth / semantic / proxy / track 对齐：`t = frame / 24`。

## 目录

| 目录 | 内容 |
|---|---|
| `clips-1000-20260828-234354/` | 主集：1000 段。walk / run / drive / look 各 250。stamp `smtd4g8ri`，2 worker。 |
| `clips-1000-fix-run-101/` | 从主集筛出的 101 条 run 重录（去掉「先环绕再跑」）。stamp `fixrun101`。轨迹与主集对应段相同，文件名不同。 |

清单都在各目录的 `clips.json`。

## 文件命名

每个 worker 一条长会话，再按段切成 `_pXXX` 分片：

```
color_<stamp>_pXXX.mp4      # RGB 画面  libx264
depth_<stamp>_pXXX.mp4      # 深度灰度  h264-logz-gray8
semantic_<stamp>_pXXX.mp4   # 语义 ID   libx264rgb / rgb24
proxy_<stamp>_pXXX.mp4      # 后处理合成，R=深度 G/B=语义
```

长会话（整 worker、未切段）以及对齐元数据：

```
color_<stamp>.mp4
depth_<stamp>.mp4
semantic_<stamp>.mp4
capture_<stamp>.json
depth_<stamp>.json
track_<stamp>.json          # 每帧相机 / 玩家 / 输入
```

主集 stamp：`w00_smtd4g8ri`、`w01_smtd4g8ri`（各 500 段）。  
101 集 stamp：`w00_fixrun101`、`w01_fixrun101`。

`clips.json` 里每段有 `tag`、出生点 `(x,z,heading)`、目标 `(goalX,goalZ)`、`turn`、以及上述分片文件名。`frameStart` / `frameEnd` 是该段在长会话里的帧区间（半开）。

## 动作标签

| tag | 含义 |
|---|---|
| `walk` | 步行到目标 |
| `run` | 冲刺到目标 |
| `drive` / `driveFast` | 开车（本集只有 `drive`） |
| `look` | 原地，相机环绕。`subject` 为 `person` 或 `car` |

`turn=true` 表示规划路径上有一次拐弯（旧数据里弯常常贴在开场；新规划会先直走再转）。

## Depth（米制）

相机空间 **正 view-z**（视线方向距离，米），不是视锥 NDC。

1. 距离 `d` clip 到 `[near, far] = [0.1 m, 256 m]`
2. 量化到 16-bit 再压成 8-bit 灰度（越近越亮）：

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

编码名：`h264-logz-gray8`。解码请读灰度，不要当普通 YUV 色度用。

## Semantic（序号）

每像素一个类别整数。视频是 **无损 RGB**：`(R, G, B) = (0, 0, id)`，codec `libx264rgb` + `rgb24`（`crf 0`）。**不要用 YUV H.264**，相邻 id 会被量化并掉。

读 **B 通道**（或确认 R=G=0 后读任一通道）：

| id | 类 | 怎么定 |
|---:|---|---|
| 0 | sky | 未命中 / 天空 |
| 1 | player | 本段可控角色（`userData.semantic`） |
| 2 | ped | NPC |
| 3 | vehicle | 载具网格（车、部分船/机） |
| 4 | building | 楼、地标，名字/资产匹配 building、住宅、mall 等 |
| 5 | road | **仅沥青材质**（`asphalt`）。路砖上的水泥、人行道、标线不算 road |
| 6 | ground | 地块、人行道、标线、路砖非沥青部分 |
| 7 | vegetation | 树、草、灌木、树池（含材质名 leaf/foliage） |
| 8 | terrain | 山地、岩石、野外地面 |
| 9 | water | 海、湖、河、池 |
| 10 | prop | 灯、椅、牌、栏、HVAC 等；对不上上面规则时的默认类 |

判定顺序（实现见 `src/semantic.ts`）：

1. 沿父节点找显式 `userData.semantic`（角色、车、行人录制时打上）
2. 名称 / `sourceFile` / `category` 正则（建筑、路、树、水、车、地形、道具）
3. 材质名覆盖：沥青 → road；树叶 → vegetation；road 网格上的 concrete/sidewalk/paint → ground
4. 其余 → prop

没有实例 ID，只有这 11 个类。开车片段里自己的车和交通车都是 `3`，不在 semantic 里区分 ego。

## Proxy（后处理）

由同帧 depth + semantic 合成，不是游戏内实时通道。文件名把 `color_` 换成 `proxy_`。

| 通道 | 含义 |
|---|---|
| **R** | 全图像素的 log-z `d`。有效深度从 depth 灰度反解到米，再按 **near=0.1 / far=8000** 重新 log 压到 `[0, 254]`；天空/无效 = **255** |
| **G, B** | 语义色，不占 R |

| 类 | G | B | 备注 |
|---|---:|---:|---|
| player | 0 | 255 | semantic id 1 |
| ped | 0 | 128 | id 2 |
| ego | 128 | 0 | id 3 且该段 `tag` 为 drive |
| vehicle | 64 | 0 | id 3，非 drive 段 |
| vegetation | 255 | 0 | id 7 |
| road | 255 | 255 | id 5 |
| other | 0 | 0 | building / ground / terrain / water / prop |
| sky | 255 | 255 | id 0 或 depth=0；靠 **R=255** 和 road 区分 |

合成脚本：`node scripts/compose-proxy.mjs <dir>`。

## 对齐与读取建议

- 分片帧数必须是 124。`ffprobe` 数帧即可校验。
- color 是普通 YUV H.264；depth 当灰度；semantic / proxy 必须按 RGB 无损解（`rgb24`）。
- 相机外参、玩家位姿在 `track_<stamp>.json` 的 `camera.frames` / `player.frames`，下标与长会话帧号一致；分片对应 `frameStart .. frameEnd-1`。
- 主集里 101 条有问题的 run 已用新视频覆盖同名 `*_smtd4g8ri_pXXX.mp4`；`track_w01_*.json` 仍是旧运动，和这 101 条画面不再逐帧对应。要以新轨迹为准请用 `clips-1000-fix-run-101/track_*.json`。
