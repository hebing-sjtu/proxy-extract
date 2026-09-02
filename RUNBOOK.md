# Runbook

把 **ABot-World-Explorer-subset2000**（只有 RGB）跑成 **ABot-seg-long-2000**
（RGB + 深度 + 语义 + DUV）的操作手册。不需要先读源码。

交付格式本身在 `DATA_F.md`，这里只讲怎么跑、看什么、怎样算对、出错了怎么办。
代码注释和 `proxy-extract/README.md` 是英文的。

上手顺序：

1. 第 1 节装 venv
2. 第 2 节拉权重
3. `make doctor` —— 一次查完所有前置条件，不中途退出
4. 第 3 节起跑，第 4 节验收

---

## 0. 整条管线长什么样

一条 episode：

一条 episode 走三个阶段，**每个阶段的产物落盘之后下一个才开始**：

```
data/<prefix>/<sample_id>/video.mp4   1920x1080, ~1800 帧
        │
        │  video.prefetch(iter_frames)  解码 + resize 到 1280x720，按 64 帧一批，
        │                               在后台线程上跑，不让 GPU 等磁盘
        ▼
┌─ 阶段 1  infer ─────────────────────────────── 断点粒度：帧 ──┐
│  depth 后端    → 米制深度       (GPU)                          │
│  semantic 后端 → 11 类 ID 图    (GPU)                          │
│  streaming.WindowStabiliser  滑窗光流补偿：中值 + 多数投票     │
│                              (CPU，最慢的一段；只持有 ±radius) │
│  streaming.RangeGuard        逐帧截断到 0.1 / 256 m            │
│        ↓ 写 frames/color/*.png、frames/depth/*.npy             │
└────────────────────────────────────────────────────────────────┘
        ▼
┌─ 阶段 2  derive ───────────────────────── 断点粒度：帧 ────────┐
│  temporal.suppress_short_runs   抹掉过短的标签游程                │
│  semantic.people.split_people   从人物掩码里挑出主角，其余为 NPC  │
│      两步都必须看完整段，所以这里持有一段的 labels（约 1.6 GiB）  │
│        ↓ 写 frames/semantic/*.npy、frames/duv/*.png            │
└────────────────────────────────────────────────────────────────┘
        ▼
┌─ 阶段 3  encode ──────────────────── 不可续跑，但最便宜 ───────┐
│  把四个 frames/ 目录读回来编成四路 mp4                          │
└────────────────────────────────────────────────────────────────┘
        ▼
seg_000123/
    frames/{color,depth,semantic,duv}/NNNNNN.{png,npy}
    proxy/{color,depth,semantic,duv}.mp4
    annotations.tar
    extraction_report.json
```

三个含义值得先知道：

- **先落逐帧、再转 mp4**。所以 worker 被杀在第 1700 帧，重跑从 1700 帧继续，不是
  从头。想看进度就看 `frames/depth/` 里的文件数。
- **跑到一半没有任何 mp4 是正常的**，不是卡住 —— 四路视频都在阶段 3 一次编出来。
- 深度和语义**各只前向一次**。`duv.mp4` 是这两者的确定性合成，不是第三次推理。

续跑时阶段 1 会把停下来那一点之前的 `temporal-radius` 帧重新前向一遍。那几帧的
结果直接丢掉，它们的作用是让**第一帧没写出来的**拿到和不中断时一样的左邻居 ——
否则接缝两侧会用截断的时间窗做稳定，而这种缺陷没有任何帧数或格式检查看得出来。
`test_streaming.py` 直接对比了续跑和一次跑完的逐帧结果。

外层编排是 `scripts/run_scenes.sh`：按 `--shard i/N` 把 episode 列表无重叠切开，
一个 worker 一个进程绑一张卡，`--resume` 让重跑只补缺的，`--keep-going` 让一条坏
episode 只损失一条，结束时跑 `scenes-audit` 统计完整度。

代码在 `proxy-extract/src/proxy_extract/delivery.py`。

---

## 1. 装 venv

```bash
cd /path/to/fastvideo_datapipe
DA3=1 scripts/setup_venv.sh          # 或 make venv
```

按顺序做四件事：查前置（Python ≥ 3.10）→ 建 `.venv` → 按 `requirements.txt` 装死
pin 的依赖并以 editable 装 `proxy-extract` → 跑自检并打印 ffmpeg 位置和 torch/CUDA
可见性。torch 那步是大头，约 3 GB。

`DA3=1` 额外装默认深度后端 `depth_anything_v3`。它没法写进 `requirements.txt`
（不在 PyPI，且自称要 numpy<2 / python<=3.13，跟我们的 pin 冲突），所以单独一步、
用 `--no-deps` 装。不加这个开关就得用 `DEPTH=depth_anything` 跑，理由见第 5 节。

> **为什么是 venv。** 这些节点上已经有一套在用的主环境。venv 只往自己那个目录里
> 装东西，删掉目录就等于卸干净，是最小的干预。

三个可调的环境变量：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `VENV` | `./.venv` | 装到别处，例如 `/data/binghe/venvs/proxy` |
| `PYTHON` | `python3` | 指定解释器。系统 `python3` 常常是 3.9，会被守卫拦下 |
| `EXTRAS` | `full` | 改成 `core` 则不装 torch，只能跑合约/分类/编码那部分 |

装完不必 activate，直接拿 venv 里的解释器调：

```bash
.venv/bin/python -m proxy_extract --help
```

**用 `$VENV/bin/python -m pip` 而不是 `$VENV/bin/pip`。** venv 建好之后被移动或
拷贝过，`pip` 脚本里的 shebang 就是过期的绝对路径，会报 `bad interpreter`；模块
形式不看 shebang。

### ffmpeg 从哪来

交付视频全部经 ffmpeg 写出（OpenCV 的 VideoWriter 要不到无损 RGB）。解析顺序：

```
$FFMPEG  →  PATH 上的 ffmpeg  →  imageio-ffmpeg 自带的静态构建
```

第三项在 `requirements.txt` 里，所以**装完 venv 就已经有一个可用的 ffmpeg**，节点
上没有 root 也不影响交付。系统装的优先，因为发行版构建的编码器更全。

确认当前用的是哪一个：

```bash
.venv/bin/python -c "from proxy_extract.proxy import ffmpeg_binary; print(ffmpeg_binary())"
```

没有 root 又想要完整发行版构建，用静态包：

```bash
mkdir -p ~/.local/bin
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz
tar -xf /tmp/ffmpeg.tar.xz -C /tmp
cp /tmp/ffmpeg-*-static/ffmpeg /tmp/ffmpeg-*-static/ffprobe ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
```

---

## 2. 拉权重

```bash
.venv/bin/python scripts/fetch_models.py --set default    # 或 make venv-fetch
.venv/bin/python scripts/fetch_models.py --set da3        # 默认深度后端，6.8 GB
```

`default` 拉 Mask2Former（语义）和 Depth Anything V2 Metric Outdoor（深度兜底）。
权重位置由 `HF_HOME` 决定，建议显式指到大盘上：

```bash
export HF_HOME=/data/binghe/cache/huggingface
```

国内网络慢先 `export HF_ENDPOINT=https://hf-mirror.com`。拉全之后批量跑可以
`HF_HUB_OFFLINE=1` —— 不只是整洁：每个 worker 都去撞 hub 会把自己限流，跑到一半
赶上 hub 抖动会留下一个处理了一半的数据集。

---

## 3. 跑交付

```bash
make scenes
```

等价于（两个路径都是默认值，改用别处就显式传）：

```bash
DATA_DIR=/data/binghe/datasets/ABot-World-Explorer-subset2000/data \
OUT_DIR=/data/binghe/datasets/ABot-seg-long-2000 \
scripts/run_scenes.sh
```

启动前它会真的把两个后端各跑一次单帧推理，所以后端装错或权重没拉会在几秒内失败，
而不是在 2000 条上各失败一次。它同时核对磁盘空间和主机内存 —— 两个数都随
`KEEP_FRAMES` 变，见下面的「体积」。

### 开几个 worker

默认 `WORKERS_PER_GPU=6`。一条 episode 里只有模型前向用 GPU —— 解码、光流稳定、
PNG 写盘、ffmpeg 编码全是 CPU，深度模型又是一次一帧 —— 所以单 worker 会让卡大段
空转。填满卡的办法是在一张卡上叠 worker，让别人的前向填进这些空档：

```bash
WORKERS_PER_GPU=8 make scenes
```

**能开几个仍然由主机内存决定，不是显存**，但比以前宽得多。实测一个 worker 峰值
约 **11 GiB**（阶段 1 恒定 6.6 GiB，阶段 2 每帧再加约 2.4 MiB），过去是约 40 GiB
——这正是默认值从 1 变成 6 的原因。8 卡 × 6 worker 约 530 GiB，脚本会按总数核对
`MemTotal` 并在不够时警告。

显存这边没有预检，因为每 worker 占多少取决于后端和 `process_res`，没实测过的数
写进脚本只会变成一个假的保证。跑起来之后自己看一眼：

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5
```

利用率没到顶而显存还有富余就加 `WORKERS_PER_GPU`。加过头的后果是有界的：一个
worker 一个进程，CUDA OOM 只杀一个 shard，重跑 `make scenes` 会从它写到的那一帧
接着走。

> **DA3 的 `window` 不是 batch size。** 它是多视图设置：`window=4` 会把这 4 帧当成
> 同一场景的 4 个视角，把它们的深度尺度绑在一起，而不是把前向做宽。所以填满 GPU
> 只能靠叠 worker。语义模型那边可以真的加宽：
> `SEMANTIC_OPTIONS="batch_size=8"`。

### 线程：叠 worker 的另一半

**叠 worker 而不同时限制线程，会得到一台比单 worker 还慢的机器。**这里每个 CPU
库都按「整台机器归我」来定线程池大小：OpenCV 的光流按核数，torch 的 CPU 算子按
核数，x264 更是按核数的约 1.5 倍。128 核上开 64 个 worker，于是有上万个线程抢 128
个核，机器把时间全花在切换上下文上。

症状极具迷惑性：**每张卡都占着显存、利用率却是 0，日志不动，而且哪里都不报错**
—— 因为确实没有出错，只是每个 worker 都慢了两个数量级。

`run_scenes.sh` 会按 `nproc / worker 数` 均分，下限 1，上限 4（再多这些池子也不
再变快，只是从别的 worker 手里抢核），并把结果打在启动摘要里：

```
shards     64 (8 GPU(s) x 8 worker(s), ~704 GiB RAM)
threads    2 per worker, of 128 core(s)
```

要自己定就 `THREADS_PER_WORKER=4 make scenes`。手跑单条时不必管：环境变量没设
时上限就不存在，每个库照旧用满整台机器。

### 精度

两个后端都在 CUDA 上跑 **bfloat16**，但来路不同，**只有语义那边是我们控制的**。

语义后端默认 `dtype=auto`：CUDA（Ampere 及以后）上用 `torch.autocast` 跑 bfloat16，
其余 float32。权重仍然是 float32，只有 autocast 认的那些算子降精度 —— 归一化层留在
float32，它们数值上脆弱而且不占时间。bfloat16 而不是 float16，是因为它保留 float32
的指数范围：深度是米，跨 0.1 到几千，一个会在这个量程中途上溢的格式恰好会在没人
检查的地方出错。想对比就两边各跑一条：

```bash
SEMANTIC_OPTIONS="dtype=float32" ...
```

**DA3 不接受 `dtype`，它自己管。** 它的 `inference()` 内部已经套了一层
`torch.autocast(bfloat16)`，同时把输入图像 `.float()` 成 float32。所以把它的权重转成
半精度不会多拿到任何 tensor core，只会让半精度权重在第一个卷积上撞见 float32 输入：

```
RuntimeError: expected scalar type Float but found BFloat16
```

后端因此把 `dtype=auto` 解析成 float32，显式要 `bfloat16` 会被当场拒绝并说明原因，而
不是让它跑到 shard 里再炸。报告里 `depth.meta` 同时记 `dtype: float32` 和
`autocast: internal`，因为单看前者会误以为这一遍是全精度跑的。

### 单条手跑

```bash
.venv/bin/python -m proxy_extract scenes \
  --video /data/binghe/datasets/ABot-World-Explorer-subset2000/data \
  --recursive \
  --out /data/binghe/datasets/ABot-seg-long-2000 \
  --semantic-backend standard11 \
  --depth-backend depth_anything_v3 \
  --resume --keep-going
```

`--recursive` 是必须的：ABot 把每条 episode 放在 `data/<prefix>/<sample_id>/`
下面，单层列目录什么也找不到。

### 先用 LIMIT 试一小轮

正式开 2000 条之前，**在目标节点上跑几条**。`LIMIT=8` 只交付前 8 个 scene，编号跟
全量跑一模一样（先编号、后截断），所以这一轮的产物就是正式交付的前缀，不用删掉重
来；空间预检也只按这 8 条算，不会拿 10 TiB 的需求把试跑挡在门外。

两卡节点上先这样，跑通再往上加：

```bash
LIMIT=4 N_GPUS=2 WORKERS_PER_GPU=2 HEARTBEAT_SECONDS=30 \
  OUT_DIR=/data/binghe/datasets/ABot-seg-trial make scenes
```

要看的三件事，按顺序：

1. 启动摘要里 `threads` 那行在不在（在，说明跑的是新脚本）；
2. 前台心跳的 `shards alive` 是不是等于 shard 总数 —— **少了就是有 worker 一起步就
   死了**，`head -20 logs/shard-*.log` 会直接给出 traceback；
3. `load` 是不是在核数附近。远超核数说明线程还在抢核。

一条 episode 走完（心跳的 `done` 加一）就说明整条链路是通的。然后 `WORKERS_PER_GPU`
往上加，每加一档看一眼 `nvidia-smi` 的利用率和 `load`，直到利用率不再涨。

### 跟一眼进度

**`make scenes` 的终端本身几乎不打印东西，这是设计如此**：每个 worker 的 stdout 各
自进 `<out>/logs/shard-i.log`，否则 64 路日志交织在一起没法看。前台只留一行心跳，
它数的是磁盘上落了什么，不依赖任何 worker 还活着：

```
[19:22:04] 12/2000 done (+10 this run), 64/64 alive, 64 writing, load 121.7
```

四个数各回答一个别的数回答不了的问题：

- **done** 是 `find <out> -maxdepth 2 -name extraction_report.json` 数出来的，**累计
  值**。往一个已经有东西的目录里续跑时它不从 0 开始，所以本轮的增量单独标出来 ——
  刚起步就显示 `2/2000 done (+0 this run)` 是对的，那 2 条是上一轮留下的。
- **alive** 是 shard 进程数。少于总数就是有 worker 死了，去 `head -20 logs/shard-*.log`。
- **writing** 是这个心跳周期内写过日志的 shard 数，也就是活性。**头十分钟看这个**：
  那时候 `done` 不可能动（一条 episode 还没跑完），但 `writing` 应该等于 shard 总数。
- **load** 是走哪条岔路。远高于核数是在抢 CPU（见第 3 节「线程」）；**远低于核数、
  `writing` 也低，是在等什么东西** —— 权重、存储，或者卡住了。

启动那几分钟 load 偏低是正常的：每个 worker 都要把 6.8 GB 的 DA3 权重读进来，几十
个进程同时读同一个文件系统，这一段既不吃 CPU 也不碰 GPU。load 应该随着 worker 陆续
读完而往上爬。十分钟后还趴着不动才是有问题。

`HEARTBEAT_SECONDS=0` 关掉心跳。要看细节就跟一个 shard 的日志：

```bash
tail -f /data/binghe/datasets/ABot-seg-long-2000/logs/shard-0.log
make scenes-audit
```

日志长这样，每条 episode 有起止，中间每 30 秒一行心跳（`--progress-interval` 可调，
`--quiet` 关掉）：

```
[19:22:04] shard 0/64: 32 of 2000 episodes
[19:22:04] seg_000000 [1/32] start
[19:22:05] seg_000000 infer: /data/.../ep/video.mp4
[19:22:35] seg_000000 infer 412 frames written, 7 batches
[19:30:10] seg_000000 derive: 1800 frames
[19:30:40] seg_000000 derive 900/1800
[19:31:12] seg_000000 encode: four videos
[19:32:01] seg_000000 done in 597s, 1800 frames
```

**心跳停了才是卡住，没输出不算。** worker 用 `python -u` 起，所以这些行是即时落盘的，
不会攒在缓冲区里 —— 早先没加 `-u` 的时候，一个正常干活的 shard 日志能空好几分钟，
跟真卡住完全分不出来。

`scenes-audit` 会把每个 scene 重新打开、按 `extraction_report.json` 的帧数核对四路
视频的实际帧数，而不是看目录在不在 —— 被杀在编码中途的 worker 留下的正是四个长度
不足的文件，而这恰恰是标记文件会掩盖的失败。它在还有缺口时返回非零，所以可以放进
shell 循环里等。它需要 `scenes_manifest.json`，那个文件是 worker 在**加载模型之前**
就写下的；所以「audit 说没有 manifest」意味着这个 `--out` 下从来没有 worker 跑起来
过，通常是 `--out` 跟 `OUT_DIR` 不是同一个路径。

### 体积

四路视频约 **465 MiB / 1800 帧**（`duv` 最大，因为 R 通道扛着 log-z 的全部细节），
2000 条约 0.89 TiB。逐帧目录比它大一个数量级：

| 流 | 逐帧 / 帧 | 逐帧 / 1800 帧 | 视频 / 1800 帧 |
| --- | --- | --- | --- |
| `depth` | 1.758 MiB（定长） | 3.09 GiB | 133 MiB |
| `semantic` | 0.879 MiB（定长） | 1.55 GiB | 37 MiB |
| `color` | 取决于画面 | — | 88 MiB |
| `duv` | 取决于画面 | — | 205 MiB |
| 合计 | | **≥ 4.6 GiB** | ≈ 465 MiB |

两路数组是**定长**的，因为它们就是原始数组；两路 PNG 取决于内容，合成素材上很小，
真实游戏画面上会大得多。**全留的话 2000 条要按 10 TiB 以上准备**，文件数约 1400 万
—— ceph 上后面这个数和前面一样要紧。

所以 `KEEP_FRAMES` 是要选的，按流选：

```bash
KEEP_FRAMES=depth make scenes    # 只留视频复现不出来的那一路，约 3.2 GiB/段
KEEP_FRAMES=none  make scenes    # 只要四路视频，回到 465 MiB/段
```

哪一路值得留见 `DATA_F.md` 的对照表。一句话：**只有 depth 是视频复现不出来的**
（8-bit log 每码 3.1%，float16 约 0.05%）；`duv` 完全可以由 depth + semantic 推
出来，`semantic.mp4` 本来就是同一批 id 的无损编码。

注意逐帧目录不影响断点续跑的能力 —— 它在跑的过程中总是存在的，`KEEP_FRAMES` 只
决定编完 mp4 之后删不删。

---

## 4. 验收

### 交付格式是设计上不可直视的

`semantic.mp4` 把类别 ID 0–10 放在蓝通道，直接播放几乎全黑；`depth.mp4` 是 log-z
灰度斜坡，看起来是一片平灰。**不要用播放器判断质量**，用：

```bash
.venv/bin/python -m proxy_extract scenes-preview \
  --scene /data/binghe/datasets/ABot-seg-long-2000/seg_000000 \
  --out /tmp/sheet.png --frames 6
```

`.png` 出等距采样的 contact sheet（`color | semantic | depth` 三联 + 类别图例），
适合 scp 回来看；后缀换成 `.mp4` 出同样面板的视频。

### 合成后端会伪造输出，而且看起来完全正常

`synthetic` 深度/语义后端存在的意义是无 GPU 时验证管线，它们产出的视频在结构上跟
真实产物**不可区分** —— 同样的编码、同样的类别 ID、同样的报告字段。这已经害过一次：
一张用 `synthetic` 出的对照图被当成了「真实分割器失效」的证据。

现在有三道防线：

- `run_scenes.sh` 直接**拒绝** `SEMANTIC=synthetic` 或 `DEPTH=synthetic`，除非显式
  设 `ALLOW_SYNTHETIC=1`；
- `extract_scene` 在写盘时抛 `PlaceholderOutput` 警告；
- `extraction_report.json` 里带 `"deliverable": false` 和 `"placeholder_backends"`。

验收任何一批数据，先：

```bash
grep -L '"deliverable": true' /data/binghe/datasets/ABot-seg-long-2000/*/extraction_report.json
```

### 会拒绝什么

深度后端返回的若是 up-to-scale（非米制）深度，`scenes` 会**直接报错**而不是交付。
交付视频编的是绝对米制，而 ABot 的 COLMAP 模型自己只定义到一个相似变换，补不上这个
尺度。所以必须用能预测米制深度的后端。

### 真后端在 ABot 上长什么样

真实 episode 实测（`standard11` = Mask2Former swin-large ADE20K，本机 CPU
2.9 s/帧），荒野徒步素材的构成大致是：

| | 占比 |
| --- | --- |
| vegetation | 72–88% |
| terrain | 6–25% |
| sky | 2.5–17% |
| person | 1.7–2.2% |

草地归 vegetation 而不是 terrain 是 `DATA_F.md` 明确规定的，不是映射错误。

主角判定在同一素材上跑 90 帧连续窗口：`resolved: True`，轨迹全程 90/90 帧，中心距
锚点 0.08 帧宽，中位面积 1.53%，merged 0%。掩码是贴合的人形轮廓。

**坐骑归 `vehicle`。** 11 类里没有动物类。被骑乘的马是载具，归 `prop` 会告诉世界
模型「一个大体积移动主体是静态杂物」，这是两种可选错误里更有害的一种。代价是
ADE20K 只有**一个** `animal` 标签，分不出马鹿狗鸟，所以野生动物也一并算作 vehicle；
要分开需要实例掩码，映射表做不到。实测两条 episode 共 24 帧里 `animal` 命中 0 像素，
所以实际暴露面很小。

---

## 5. 后端怎么选

```
--semantic-backend  standard11  11 类交付体系（交付用这个）
                    coarse6     6 类粗分类
                    ade20k      12 类 CWM 原生，Mask2Former
                    cityscapes  12 类，SegFormer，街景更准但没有室内类
                    synthetic   假数据，只用来验证接线

--depth-backend     depth_anything_v3  DA3 嵌套模型，米制（默认，见下）
                    depth_anything     DA V2 单帧米制，装完即用的兜底
                    mapanything        多帧 + 可吃 GT 相机，但取权重会卡（见下）
                    synthetic          假数据
```

实际约束不是精度，而是**权重能不能在这台机器上拿到**，以及**它肯不肯声明自己是
米制**。

**`depth_anything_v3` 是默认值。** 三个里唯一同时满足两个条件的：DINOv2 主干烘焙在
它自己的 `model.safetensors` 里，所以只要连得上 HF 就能拿全所有权重，不像 mapanything
那样另外走 `torch.hub`；而且它的嵌套 checkpoint 会用自己预测的焦距完成换算并置
`is_metric = 1`。三个代价：

1. **不在 PyPI，声明的 pin 跟本环境冲突**（`numpy<2` vs 我们的 2.5.1；
   `requires-python <=3.13` vs 3.14）。直接装会把 numpy 降级、把 torch 拽走，所以
   必须绕开它的依赖解析（`setup_venv.sh` 的 `DA3=1` 做的就是这个）：

   ```bash
   .venv/bin/python -m pip install --no-deps --ignore-requires-python \
       git+https://github.com/ByteDance-Seed/depth-anything-3
   .venv/bin/python -m pip install einops omegaconf addict imageio
   .venv/bin/python scripts/fetch_models.py --set da3      # 6.8 GB
   ```

   `--no-deps` 会漏掉一大堆，`pip check` 会把它们全列出来：`gsplat` / `open3d` /
   `pycolmap` / `moviepy` / `evo` / `trimesh` / `plyfile` 服务高斯导出、位姿对齐和
   benchmark，`fastapi` / `uvicorn` / `gradio` / `pillow-heif` 服务它自带的 demo
   服务，`pre-commit` 是开发用的。**这些缺失是正常状态，不要去补**。后端在导入前给
   `utils.export` 和 `utils.pose_align` 两个子模块装了占位实现，占位被真的调用才会
   报错 —— 单目路径永远走不到（源码里 `if extrinsics is None: return`）。

   两个看着像例外的其实也不用装：`xformers` 只在 DINOv2 的 SwiGLU 里 try/except 导入，
   失败就退回纯 torch 实现；`e3nn` 同理，只用于旋转高斯的球谐系数。装 `xformers` 反而
   要跟 torch 版本严格对齐，是净负债。

   它声明的 `numpy<2` 也不要照做：那条 pin 是 `open3d`/`pycolmap` 的要求，DA3 自己的
   代码里没有任何 numpy 1 独有的符号（`np.float_`、`np.bool8` 之类一个都没有）。降级
   numpy 会连带拆掉 opencv 和 torch。

   真正需要的只有下面这些，`einops`/`omegaconf`/`addict`/`imageio` 之外的都已经在
   `requirements.txt` 里：torch、torchvision、numpy、pillow、opencv、huggingface-hub、
   safetensors、tqdm。`python scripts/doctor.py` 会逐个核对，并把 `pip check` 的噪声
   跟真正的缺失分开。

2. **权重是 CC BY-NC 4.0**，只能研究用。想要 Apache-2.0 的只有 `DA3METRIC-LARGE`
   （`--set da3-apache`），但它是 DinoV2+DPT、**没有相机头**：输出 canonical 深度、
   从不设置 `is_metric`、也给不出换算所需的焦距，所以 `scenes` 会直接拒收。要用它就得
   外部补焦距（`annotations.tar` 的 COLMAP `cameras.txt` 里有像素焦距，且像素焦距不受
   COLMAP 尺度不确定性影响）—— 这条路还没实现。

3. **它不输出天空掩码。** 所以天空只能由语义分支认定：`duv.mp4` 靠 `ids == sky` 仍然
   正确，但 `depth.mp4` 的 0 哨兵拿不到它。这跟另外两个后端的现状一样，不是新问题。

**`depth_anything`（V2）是不想折腾时的兜底。** `requirements.txt` 覆盖它的依赖，
`fetch_models.py --set default` 拉的正是它的权重。它逐帧预测，帧与帧之间的尺度没有
绑定，这个 caveat 会写进每份 report 的 `depth.meta.caveat`。

**`mapanything` 要单独装，而且它的第三份权重 `fetch_models.py` 拿不到。**

```bash
.venv/bin/python -m pip install 'git+https://github.com/facebookresearch/map-anything'
.venv/bin/python scripts/fetch_models.py --set mapanything     # 需要 hf auth login
```

它的 DINOv2 主干走 `torch.hub` 从 `dl.fbaipublicfiles.com` 拉 —— 跟 HF 无关，所以
`HF_ENDPOINT` 和 `HF_HUB_OFFLINE=1` 对它都不起作用。很多集群的出网白名单不含这个域名
（跟节点在哪个国家无关），症状是日志停在

```
Using cache found in ~/.cache/torch/hub/facebookresearch_dinov2_main
```

然后**不报错、GPU 零占用**。那行说的是仓库代码已缓存，卡住的是紧随其后的 1.2 GB 权重
下载。确认：

```bash
ls -la ~/.cache/torch/hub/checkpoints/      # 空的就是没下下来
curl -sI --max-time 10 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
```

出路是在能出网的机器上跑一次预检把 `torch.hub` 缓存填满，再整个 rsync 到节点上 ——
这样不用猜 MapAnything 用的是哪个 DINOv2 变体。

DA3 也支持多帧联合重建（它是 any-view 模型）：

```bash
--depth-backend-option window=4        # 窗口内的帧当作同一场景的多个视角
--depth-backend-option process_res=728 # 默认 504；调高更清晰也更慢
```

（没有 `dtype` 可调，理由见第 3 节的「精度」。）

`window > 1` 时该窗口内的尺度绑定在一起。默认 `window=1` 即逐帧 —— 动态场景下多视角
假设不成立，所以不默认打开。**也别把它当 batch size 用**：调大它是在改模型看到的
东西，不是在把前向做宽。

---

## 6. 时间花在哪，该调什么

问过一次「效率太低，是不是卡 IO，要不要先把视频拆成帧再打包喂模型」。答案是**不是
IO，也不要预拆帧**。下面是本机（macOS，CPU 单机）在真实 ABot episode 上实测、按
1800 帧折算的时间构成。绝对值和服务器不可比，比例可以参考。

| 阶段 | 秒 / 1800 帧 | 设备 |
| --- | --- | --- |
| 解码 + resize 到 1280x720 | 4 | CPU |
| `color.mp4` 编码（x264 CRF 16 medium） | 59 | CPU |
| 时序稳定（radius 2, flow_downscale 2） | 382 | CPU |
| `depth`/`semantic`/`duv` 三路编码 | ~30 | CPU |
| 深度 + 语义前向 | ~720 | GPU |

CPU 侧合计约 475 秒，占整条 episode 的四成，这段时间 GPU 是空的 —— 这就是显存只占
零头、算力断断续续的原因。

**关于预拆帧**：曾经的结论是「不要预拆帧」，理由是解码只占 4 秒，而把 1800 帧写出来
再读回来是几 GB 的额外读写 —— 拿 IO 换那 4 秒，方向是反的。**这个理由现在仍然成立，
但管线还是落了逐帧**，因为落盘买的不是解码时间，是另外两样东西：按帧续跑，以及
`frames/depth` 这个视频复现不出来的交付物。代价是真实的（每段 ≥ 4.6 GiB 的写），
所以它是可选的 —— `KEEP_FRAMES=none` 仍然会在跑的过程中落盘（续跑要它），只是编完
mp4 就删。

**按性价比该调什么**：

1. `WORKERS_PER_GPU`（默认 6）。不缩短单条 episode，但让别的 worker 的前向填进那
   475 秒空档，吞吐提升最直接，且不改变任何输出。上限是主机内存，不是显存。
2. `dtype=auto`（已默认）。CUDA 上两个模型都跑 bfloat16，是单个最大的杠杆。见第 3 节。
3. 光流共享（已默认生效）。`stabilize_depth` 和 `stabilize_labels` 原来各算一遍**完全
   相同**的 Farneback 光流，现在 `stabilize_pair` 只算一遍。实测 483 → 382 秒，省 21%，
   输出逐位一致（`TestSharedFlow` 钉住了这一点）。
4. 剩下的都是取舍，默认没开：
   - `--flow-downscale 4`：光流在真实 720p guide 上的实测成本是 ds1 366.6s /
     **ds2 101.4s（默认）** / ds4 29.1s（均按 1800 帧折算），所以从默认换到 ds4 省
     **72 秒**，稳定化 382 → 约 310 秒。质量代价没有在真实内容上量到：合成平移片段上
     ds2 和 ds4 与精确光流的标签分歧都是 0.00%，但那个片段只有平滑全局位移，任何尺度
     都能解出来，测不到风险所在的独立运动物体边界。要开就先跑一条 episode 用
     `scenes-preview` 看人物边缘。
   - `--temporal-radius 1`：稳定化再省一半左右（窗口内的投票和中值都减半，不只是光流），
     但时序平滑窗口从 ±2 缩到 ±1，去闪烁减弱。
   - color 换 `-preset veryfast`：59 → 19 秒，实测 PSNR 从 38.20 掉到 37.48 dB。
     不是白捡。

怎么传：`scenes` 直接加这些 flag；走 `run_scenes.sh` 用 `SCENES_ARGS` 透传，例如
`SCENES_ARGS="--flow-downscale 4" WORKERS_PER_GPU=8 make scenes`。启动时打印的
`extra` 一行会回显，确认没打错。

### 光流的取舍

600 帧实测（真实 ABot episode，本机 CPU，合成后端）：

| | 无光流 | `--flow-downscale 2`（默认） | `--flow-downscale 1` |
| --- | --- | --- | --- |
| 耗时 | 159 s | 222 s | 412 s |
| 光流净成本 | — | 63 s | 253 s |
| `flicker_after` | 0.0382 | 0.053909 | 0.053934 |
| 四路视频合计 | 138 MiB | 154 MiB | 154 MiB |

**降采样解光流几乎不花代价**：ds=2 和 ds=1 的 flicker 差 0.05%，而开销是 1/4，符合
Farneback 对像素数的平方关系。默认值就按这个定的。

> 这里**不能拿「无光流」的 flicker 更低当成它更好**。`flicker_rate` 数的是逐帧变类的
> 像素比例；不做流补偿时投票会在未对齐的窗口上把移动内容抹平，标签「粘」住了，于是
> 这个指标反而更低 —— 那是边缘涂抹，不是稳定。上面 ds1/ds2 的对比是在同样开启光流的
> 前提下比的，才是可比的。

### 内存怎么来的

本机实测峰值 RSS，按阶段分：

| 帧数 | 阶段 1 infer | 阶段 2 derive（累计峰值） |
| --- | --- | --- |
| 600 | 6.46 GiB | 7.63 GiB |
| 1200 | 6.64 GiB | 9.24 GiB |

**阶段 1 不随 episode 长度走**（600 → 1200 帧只动了 0.2 GiB）：滑窗只持有
`stabilise_block + 2 × radius` 帧，默认 132 帧。剩下的是每批的模型输出和预取队列，
都由 `--chunk-frames` 定，跟总长无关。

**阶段 2 随长度线性**，约 2.4 MiB/帧：`suppress_short_runs` 和 `split_people` 都
必须看完整段才能定一个结论 —— 一段游程有多长要到片尾才知道，哪个人是主角要比完
全片的所有人物轨迹才知道。所以这里持有一段的 labels，但只是 labels，一像素一字节，
不是过去那个 float32 的深度栈。

线性外推到 1800 帧是**约 11 GiB**，过去是约 40 GiB。上面这些是在一台笔记本上用
合成后端量的，第一次上 H200 时请拿真后端复核一遍 —— 模型自己的分配器也算在
worker 的 RSS 里，而合成后端什么都不分配：

```bash
python scripts/measure_footprint.py --frames 1800 \
    --depth-backend depth_anything_v3 --semantic-backend standard11
```

它按 `run_scenes.sh` 的变量名把 `GIB_PER_WORKER` 和 `MIB_PER_SCENE` 直接打出来。

要再省的话，剩下的都在阶段 2：`split_people` 里 `person_masks` 和 `labels.copy()`
各占一份整段（各约 0.9 MiB/帧）。没做，因为 11 GiB 下 8 卡 × 6 worker 只要 530 GiB，
瓶颈已经不在这里。

`--chunk-frames`（默认 64）管的是 GPU 上单次前向的激活量和每批输出的大小；
`--stabilise-block`（默认 128）管滑窗持有多少帧。两者都不影响输出，也都不在续跑的
指纹里 —— 换更大的卡重启一次跑，之前写好的帧照样接着用。

---

## 7. 出错了怎么办

**先跑体检。** `run_scenes.sh` 的前置检查遇到第一个问题就退出（对启动器是对的：用错
的 torch 起跑会白烧几个小时），代价是新节点上每修一个问题要来回一轮。体检脚本把所有
前置条件一次查完并各自给出修法，不中途退出：

```bash
make doctor
```

它检查 python 版本与是否在 venv 内、`proxy_extract` 能否导入、ffmpeg 解析到哪个二进制、
torch 版本与 `torch.cuda.is_available()`（对上 nvidia-smi 的驱动号）、torchaudio 是否
装了但加载不了、两个后端的包在不在、权重是否已在 HF 缓存里、`DATA_DIR` 有几条 episode、
`OUT_DIR` 空间够不够、主机内存能放几个 worker。它不改任何东西，也不下权重。

环境相关：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `bad interpreter` / `pip` 找不到 | venv 建好后被移动或拷贝过，`pip` 脚本的 shebang 是过期绝对路径 | 用 `.venv/bin/python -m pip`；venv 移动过就重建 |
| `ModuleNotFoundError: proxy_extract` | 装到了主环境而不是 venv，或没装 | `.venv/bin/python -m pip install --no-deps -e proxy-extract` |
| `error: need Python >= 3.10` | 系统 `python3` 常常是 3.9 | `PYTHON=/path/to/python3.12 scripts/setup_venv.sh` |
| `sys.prefix` 不是 venv 路径 | shell 里 activate 了 conda/主环境 | `deactivate`，或直接用 `.venv/bin/python` 全路径调用 |
| 报错说找不到 ffmpeg | 三条出路都写在报错里 | 见第 1 节「ffmpeg 从哪来」 |
| `No module named 'mapanything'` | 它不在 PyPI，`setup_venv.sh` 不会装它 | `DEPTH=depth_anything`，或按第 5 节从 git 装 |
| `pip` 报一串 `depth-anything-3 requires open3d / pycolmap / fastapi ...` | `--no-deps` 的预期结果，那些服务导出、benchmark 和 demo 服务 | 不用管。`scripts/doctor.py` 的 `da3 deps` 一行会告诉你真正缺没缺 |
| `pip` 报 `X 0.2.0 requires torch==2.12.0, but you have 2.13.0` | 这个 venv 里还装着别的项目，它的 pin 跟本管线的冲突；下次 `pip install` 就会把 torch 换掉 | 这个 venv 只跑本管线的话 `pip uninstall -y <X>`；否则给本管线单独建一个。`doctor.py` 会按包列出来 |
| `depth backend 'depth_anything_v3' failed: RuntimeError: expected scalar type Float but found BFloat16` | 有人把 DA3 的权重转成了半精度；它内部自己 autocast 并把输入 `.float()`，于是半精度权重撞上 float32 输入 | 别传 `dtype`。现在 `dtype=auto` 就是 float32，显式要半精度会被拒绝并说明原因，见第 3 节「精度」 |
| 日志里 `[WARN] Dependency 'e3nn' not found` | DA3 在导入高斯分支的球谐工具 | 正常，它只服务高斯导出，本管线走不到 |
| 占着显存、`utilization.gpu` 是 0、日志不动、也不报错 | 按可能性排：**① 线程抢核**（每个库都按整机开池子，见第 3 节「线程」，用旧版脚本启动的必然踩到）；② 上一轮的进程还占着显存和核；③ 正常地在跑 CPU 那一段 —— 稳定化、PNG、ffmpeg 都不用 GPU | 先 `uptime` 看 load average：远高于核数就是 ①，升级脚本后重启即可。`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 的 pid 数多于 worker 数就是 ②，见下一行。都不是就 `py-spy dump --pid <pid>` 看栈 |
| `nvidia-smi` 里的进程数少于自己开的 worker 数（心跳的 `shards alive` 也少） | 有 worker 在起步阶段就死了，日志里是 traceback，但它被淹在几十个日志文件中间 | `head -20 $OUT_DIR/logs/shard-*.log` 一次看全部开头。2026-09 之前的版本有一个必踩的：所有 shard 抢同一个 manifest 临时文件，64 路里能死掉一批，`git pull` 即可 |
| `nvidia-smi` 里的进程数多于自己开的 worker 数 | 上一轮 worker 没退干净，显存和核都还被它们占着，新一轮于是挤在剩下的卡上 —— 「每张卡 worker 数不一样」通常就是这么来的 | `comm -23 <(nvidia-smi --query-compute-apps=pid --format=csv,noheader \| sort -u) <(pgrep -f "proxy_extract scenes" \| sort -u)` 列出没人认领的 pid，确认后 `kill -9`。下一轮起来前 `nvidia-smi` 应该是干净的 |
| `scenes-audit` 报 `no scenes_manifest.json` | manifest 在模型加载之前就写了，所以这个 `--out` 下没有 worker 跑过 | 核对 `OUT_DIR` 和 audit 的 `--out` 是不是同一个路径（`ls -d /data/binghe/datasets/ABot-seg-*`） |
| `semantic backend 'standard11' failed: ImportError: Mask2FormerLoss requires the scipy library` | Mask2Former 的 `__init__` 无条件构造训练损失，那个损失在构造时就要 scipy（匈牙利匹配用）。本管线不训练也从不调它，但模型加载不过去 | `pip install scipy`。已经写进 `requirements.txt`，老 venv 补装即可 |

模型和数据相关：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 日志里 `CUDA initialization: The NVIDIA driver on your system is too old`，随后 `Non-CUDA device detected` | torch 的 wheel 是按比驱动更新的 CUDA 构建的，于是**整批 worker 静默退回 CPU** —— `nvidia-smi` 照样列出所有卡，`N_GPUS` 照样数对 | 对比 `python -c "import torch; print(torch.version.cuda)"` 和 `nvidia-smi --query-gpu=driver_version --format=csv`；驱动旧就按驱动的 CUDA 重装 torch，如 `--index-url https://download.pytorch.org/whl/cu126`。`run_scenes.sh` 会在启动前拦下这种情况 |
| 换过 torch 之后 `OSError: Could not load this library: .../torchaudio/lib/_torchaudio.abi3.so` | torchaudio 的 C++ 扩展是按换掉之前那个 torch 编译的。transformers 导入图像处理器时会路过 `audio_utils`，于是被牵连 —— 跟音频无关 | `pip uninstall -y torchaudio`。它不在 `requirements.txt` 里，transformers 那处 import 由 `is_torchaudio_available()` 守着，包不在就整段跳过。**不要**去装「匹配版本」：torchaudio 停在 2.11，没有配 torch 2.13 的构建 |
| 确实想用 CPU 跑 | — | `N_GPUS=1 ALLOW_CPU=1 scripts/run_scenes.sh` |
| 下权重卡住 / 超时 | hub 网络 | 先确认是不是集群出网白名单的问题（见第 5 节）；国内节点才需要 `export HF_ENDPOINT=https://hf-mirror.com` |
| `OSError: ... preprocessor_config.json` | 权重没拉全就跑了离线模式 | 重跑 fetch，确认 `--set` 覆盖了你用的后端 |
| `no video.mp4 under ...` | 忘了 `--recursive`；ABot 是嵌套目录 | 见第 3 节 |
| 跑到一半 scene 的 `proxy/` 里只有 `color.mp4` | 正常，不是卡住 | 见第 0 节 |
| GPU 显存只占零头、利用率断断续续 | 一条 episode 里只有模型前向是 GPU | 加 `WORKERS_PER_GPU`（默认 6），见第 3 节 |
| 觉得瓶颈是磁盘 / 想先把视频拆成帧 | 解码只占 4 秒 | 别这么做，见第 6 节 |
| preview 颜色不对 | 手动指定了错的调色板 | 别传，让它自己从 report 读 |
| 交付的语义看着完全不对 | 很可能用了 `synthetic` 占位后端 | 查 report 的 `deliverable` 字段，见第 4 节 |
| 某个 shard 挂了 | 看 `$OUT_DIR/logs/shard-N.log` | 修完重跑同一条命令，`--resume` 会补 |

---

## 8. 已知限制

写在前面，免得当成 bug 去查。

1. **深度是逐帧的**（用默认的 `depth_anything_v3`，`window=1`）。帧与帧之间的尺度
   没有绑定，时序稳定只能压掉抖动，压不掉整体漂移。
2. **player/ped 拆分的先验没有在 ABot 上重新拟合过。** 锚点和阈值是在一个自带
   player/ped 引擎标签的语料上拟合的；ABot 没有这种标签，所以这些参数在这里是外推。
   逻辑本身有单元测试覆盖，但「在人多的场景里靠谱吗」这个问题，目前没有能回答它的
   ground truth。
3. **road / ground 是功能区分，图像上判不准。** 见 `DATA_F.md`。
4. **`annotations.tar` 里的 COLMAP 不是米制**，只定义到一个相似变换，所以给不了深度
   尺度。它的**像素焦距**不受这个不确定性影响，是可用的。
5. **`animal` 一律映射到 `vehicle`**，坐骑和野生动物分不开。见第 4 节。

---

## 附录：`condition_root`（`extract`）

另一条产出路线，**共用同一套模型和后处理**，只有分辨率和落盘格式不同。它产出的是
[code-world-model](code-world-model/) 训练直接吃的 condition，不是本文档主线的交付物。
搞混这两条是最常见的困惑来源：

| | `scenes` → `seg_NNNNNN` | `extract` → `condition_root` |
| --- | --- | --- |
| 给谁用 | 按 `DATA_F.md` 交付 | CWM 训练直接吃 |
| 分辨率 | 1280×720 | 336×192 |
| 落盘 | `proxy/` 下四个 mp4 + `annotations.tar` | 每帧 `.depth.f32` + `.semantic_id.png` |
| 长度 | 整段 | 截到 `124 + 90k` 帧以对齐窗口 |
| 深度方向 | `depth.mp4` 反向，`duv.mp4` 正向 | 反向（近 = 高码值） |
| 代码 | `delivery.py` | `pipeline.py` |

CWM 的输入格式是写死的，只能适配不能改：336×192，窗口 124 帧步长 90 帧（所以一个
clip 至少要 124 帧），深度 0.3–256 m 对数且反向编码，语义是 12 类（CWM 原生）或
6 类（`coarse6`）。

```bash
.venv/bin/python -m proxy_extract extract \
  --video <dir> --recursive --out <out> \
  --semantic-backend coarse6 --depth-backend depth_anything \
  --chunk-frames 124 --resume --keep-going

.venv/bin/python -m proxy_extract validate --condition-root <out>/<clip> --expect-frames 124
.venv/bin/python -m proxy_extract preview  --condition-root <out>/<clip> --out /tmp/preview.mp4
```

`preview` 渲出来是三行：源画面、深度、语义色块。注意中间那行 **暖色是近处**，因为
编码是反的，第一次看的人十有八九会读反。

体积代价要先算：2000 条约 850 GiB、**约 700 万个文件**。ceph 上这个文件数是个真问题,
这也是交付走 `scenes` 的原因之一。
