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

```
data/<prefix>/<sample_id>/video.mp4   1920x1080, ~1800 帧
        │
        │  video.iter_frames   解码 + resize 到 1280x720，按 64 帧一批
        ▼
   ┌─ 每批 ────────────────────────────────────────┐
   │  color.mp4 边解码边写出（所以它最先出现）       │
   │  depth 后端    → 米制深度      (GPU)           │
   │  semantic 后端 → 11 类 ID 图   (GPU)           │
   └───────────────────────────────────────────────┘
        │   全片的 depth / labels / 灰度 guide 攒在内存（每 worker 约 40 GiB）
        ▼
   temporal.stabilize_pair        光流补偿的中值 + 多数投票   (CPU，最慢的一段)
        ▼
   semantic.people.split_people   从人物掩码里挑出主角，其余为 NPC
        ▼
   depth.scale.apply_range_guard  截断到 0.1 / 8000 m
        ▼
   delivery._encode               一次写出 depth / semantic / duv 三路
        ▼
seg_000123/
    proxy/{color,depth,semantic,duv}.mp4
    annotations.tar
    extraction_report.json
```

两个时序含义值得先知道：

- **跑到一半 `proxy/` 里只有 `color.mp4` 是正常的**，不是卡住。另外三路要等全片
  光流稳定做完才一次性写出。想看进度就看 `color.mp4` 有没有在变大。
- 深度和语义**各只前向一次**。`duv.mp4` 是这两者的确定性合成，不是第三次推理。

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
而不是在 2000 条上各失败一次。它同时核对磁盘空间（每条约 465 MiB）和主机内存。

### 开几个 worker

默认按 `nvidia-smi` 数出的卡数开进程，一卡一个。但一条 episode 里只有模型前向用
GPU —— 解码、全片光流稳定、ffmpeg 编码全是 CPU，深度模型又是一次一帧 —— 所以单
worker 会让卡大段空转（H200 上实测只占 140 GiB 里的 12 GiB）。想填满就多开：

```bash
WORKERS_PER_GPU=3 make scenes
```

**能开几个由主机内存决定，不是显存。** 每个 worker 要把整条 episode 的 720p
depth/label 栈放在内存里，约 40 GiB，脚本会按总数核对 `MemTotal` 并在不够时警告。
8 个 worker 就要约 320 GiB。

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

### 跟一眼进度

```bash
tail -f /data/binghe/datasets/ABot-seg-long-2000/logs/shard-0.log
make scenes-audit
```

`scenes-audit` 会把每个 scene 重新打开、按 `extraction_report.json` 的帧数核对四路
视频的实际帧数，而不是看目录在不在 —— 被杀在编码中途的 worker 留下的正是四个长度
不足的文件，而这恰恰是标记文件会掩盖的失败。它在还有缺口时返回非零，所以可以放进
shell 循环里等。

### 体积

单帧字节数（真实 episode 前 600 帧实测除以帧数；`duv` 最大是因为 R 通道扛着 log-z
的全部细节）：

| 流 | 600 帧 | 折算 1800 帧 |
| --- | --- | --- |
| `duv.mp4` | 68.3 MiB | 205 MiB |
| `depth.mp4` | 44.4 MiB | 133 MiB |
| `color.mp4` | 29.4 MiB | 88 MiB |
| `semantic.mp4` | 12.2 MiB | 37 MiB |
| 合计 + tar | 155 MiB | **≈ 465 MiB** |

**2000 条约 0.89 TiB**，是源 RGB（约 211 GiB）的 4.3 倍，每个 scene 6 个文件，
全集约 1.2 万个文件。深度进的是无损视频而不是逐帧 `.f32` —— 走 `condition_root`
那条路总量差不多（约 850 GiB）但文件数约 700 万，ceph 上这是决定性的差别。

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

   `--no-deps` 会漏掉 `gsplat` / `open3d` / `pycolmap` / `moviepy` / `evo`，那些只
   服务高斯导出和多视角位姿对齐。后端在导入前给这两个子模块装了占位实现，占位被真的
   调用才会报错 —— 单目路径永远走不到（源码里 `if extrinsics is None: return`）。

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

`window > 1` 时该窗口内的尺度绑定在一起。默认 `window=1` 即逐帧 —— 动态场景下多视角
假设不成立，所以不默认打开。

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

**为什么不预拆帧**：解码只占 4 秒。把 1800 帧 720p 写成文件再读回来是 5 GB 的额外写
加 5 GB 的额外读（原始帧 1280×720×3 = 2.76 MB），换的是那 4 秒里的一部分。方向是反的。

**按性价比该调什么**：

1. `WORKERS_PER_GPU=3`。不缩短单条 episode，但让别的 worker 的前向填进那 475 秒空档，
   吞吐提升最直接，且不改变任何输出。上限是主机内存，不是显存。
2. 光流共享（已默认生效）。`stabilize_depth` 和 `stabilize_labels` 原来各算一遍**完全
   相同**的 Farneback 光流，现在 `stabilize_pair` 只算一遍。实测 483 → 382 秒，省 21%，
   输出逐位一致（`TestSharedFlow` 钉住了这一点）。
3. 剩下的都是取舍，默认没开：
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
`SCENES_ARGS="--flow-downscale 4" WORKERS_PER_GPU=3 make scenes`。启动时打印的
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

600 帧实测峰值 RSS **13.3 GiB**。每一项都是「每帧数组 × 帧数」，没有别的量级项，
所以线性外推到 1800 帧是**约 40 GiB**（第一次在 H200 上跑满 1800 帧时请复核这个数）。

内存随 episode 长度走、不随窗口走，因为时序稳定和主角跟踪都跑在整段上 —— 这样它们
跟测试覆盖的批处理行为完全等价，不需要为流式再引入一套近似。峰值里除了三个主栈
（深度 6.6 GB、标签 1.7 GB、光流引导 1.7 GB）之外，还有稳定器在释放输入前先分配的
输出、`np.concatenate` 的双份、以及 range guard 的临时量。

`--chunk-frames`（默认 64）管的是 GPU 上单次前向的激活量，**不管**这些主机端的栈。

要省的话按性价比排：按 `probe` 的帧数预分配以消掉 `concatenate` 的双份（约省 6.6 GB）、
range guard 改成原地（再省 6.6 GB）、深度栈降到 float16（再省 3.3 GB，而 8-bit 对数
量化的步长是 3.1%，float16 的精度远远够）。这些都还没做，因为当前判断是主机内存不是
瓶颈。

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
| GPU 显存只占零头、利用率断断续续 | 一条 episode 里只有模型前向是 GPU | `WORKERS_PER_GPU=3`，见第 3 节 |
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
