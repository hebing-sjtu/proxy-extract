# Runbook：从零跑通 proxy-extract

面向第一次接触这个仓库的人。不需要先读论文，也不需要先读源码。

代码注释和 `proxy-extract/README.md`、`experiments/README.md` 是英文的。
**这份是操作手册**：怎么跑、看什么、怎样算对、出错了怎么办。

建议顺序：

1. **装 venv：`scripts/setup_venv.sh`**（第 2 节）。这是维护的主路径。
2. 交付数据：第 12 节 `scenes`，第 13 节验收。
3. 只想摸不需要模型的部分（合约/分类/QC/编码）：`EXTRAS=core scripts/setup_venv.sh`，纯 CPU。

> **为什么是 venv 而不是容器。** 这些节点上已经有一套在用的主环境，而镜像路线要
> docker daemon、container toolkit 和 root，动的是机器级的东西。venv 只往自己那个
> 目录里装东西，删掉目录就等于卸干净，是更小的干预。容器路线仍然保留在第 3 节，
> 两条路装的是**同一个** `requirements.txt`，所以版本不会分叉。

> **仓库里没有图和素材。** `handpick29_high_low/`（语料）和
> `experiments/figures/`（渲染出来的图和 preview）都不在版本控制里 ——
> 它们是商业游戏画面。量化结论以 JSON 形式保留在 `experiments/*.json`，
> 图本身在拿到语料后用 `experiments/fig_*.py` 重画，见第 10 节。

---

## 1. 这东西是干什么的

把一段游戏视频，变成 [code-world-model](code-world-model/)（下称 CWM）能吃的
**condition**：每帧一张米制深度图 + 一张语义 ID 图。

CWM 的输入格式是写死的，我们只能适配，不能改：

| 项 | 值 | 说明 |
| --- | --- | --- |
| 分辨率 | 336 × 192 | 深度是 raw float32，语义是 8-bit PNG |
| 窗口 | 124 帧 | 步长 90 帧，所以一个 clip 至少要 124 帧 |
| 深度范围 | 0.3 – 256 m | 对数编码且**反向**：0.3 m → 65535，256 m → 0 |
| 语义 | 小整数 ID | 12 类（CWM 原生）或 6 类（本项目的粗分类） |

跑完一条 clip，`preview` 渲出来是三行：源画面、深度、语义色块。

注意中间那行：**暖色是近处**。因为 DUV 编码是反的（近 = 高码值），
第一次看的人十有八九会读反。

6 类体系是：`background` / `road` / `vegetation` / `vehicle` / `npc` / `hero`。
`hero`（主角）没有任何分割模型能直接预测，是靠跟踪「谁一直被镜头跟着」事后拆出来的。

---

## 2. 装 venv（主路径）

```bash
cd /path/to/fastvideo_datapipe
DA3=1 scripts/setup_venv.sh
```

它按顺序做四件事：查前置（Python ≥ 3.10）→ 建 `.venv` → 按 `requirements.txt`
装死 pin 的依赖 + 以 editable 装 `proxy-extract` → 跑自检并打印 ffmpeg 位置和
torch/CUDA 可见性。torch 那步是大头，约 3 GB。

`DA3=1` 会额外装默认深度后端 `depth_anything_v3`。它没法写进 `requirements.txt`
（不在 PyPI，且自称要 numpy<2 / python<=3.13，跟上面的 pin 冲突），所以单独一步、
用 `--no-deps` 装。不加这个开关就得用 `DEPTH=depth_anything` 跑，理由见第 7 节。

### ffmpeg 从哪来

交付视频全部经 ffmpeg 写出（OpenCV 的 VideoWriter 要不到无损 RGB）。解析顺序：

```
$FFMPEG  →  PATH 上的 ffmpeg  →  imageio-ffmpeg 自带的静态构建
```

第三项在 `requirements.txt` 里，所以**装完 venv 就已经有一个可用的 ffmpeg**，
节点上没有 root 也不影响交付。系统装的优先，因为发行版构建的编码器更全。

它是核心依赖而不是可选，原因是 ffmpeg 装不进 pip 的其他任何途径：没有它，
一台管理员拿不到 root 的机器根本没有出路。而且编码测试本身要跑 ffmpeg，
`EXTRAS=core` 也需要。

要用系统的（编码器更全，需要 root）：

```bash
apt-get update && apt-get install -y ffmpeg
```

没有 root 又想要完整发行版构建，用静态包：

```bash
mkdir -p ~/.local/bin
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz
tar -xf /tmp/ffmpeg.tar.xz -C /tmp
cp /tmp/ffmpeg-*-static/ffmpeg /tmp/ffmpeg-*-static/ffprobe ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
```

确认当前用的是哪一个：

```bash
.venv/bin/python -c "from proxy_extract.proxy import ffmpeg_binary; print(ffmpeg_binary())"
```

三个可调的环境变量：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `VENV` | `./.venv` | 装到别处，例如 `/data/binghe/venvs/proxy` |
| `PYTHON` | `python3` | 指定解释器。系统 `python3` 常常是 3.9，会被守卫拦下 |
| `EXTRAS` | `full` | 改成 `core` 则不装 torch，只能跑合约/分类/QC/编码 |

装完不必 activate 也能用，直接拿 venv 里的解释器调：

```bash
.venv/bin/python -m proxy_extract --help
```

**用 `$VENV/bin/python -m pip` 而不是 `$VENV/bin/pip`。** venv 建好之后被移动或
拷贝过，`pip` 脚本里的 shebang 就是过期的绝对路径，会报 `bad interpreter`；
模块形式不看 shebang。脚本内部已经这么做了，手动补装依赖时也照这个来。

### 拉权重

```bash
.venv/bin/python scripts/fetch_models.py --set default
```

默认拉推荐配置的两个模型：Mask2Former（语义）和 Depth Anything V2 Metric
Outdoor（深度）。国内网络慢先 `export HF_ENDPOINT=https://hf-mirror.com`。
权重位置由 `HF_HOME` 决定，建议显式指到大盘上，例如
`export HF_HOME=/data/binghe/cache/huggingface`。

`fetch_models.py` 不依赖 docker，容器只是把它拷进镜像复用。

### 冒烟：不用权重跑通整条链路

```bash
.venv/bin/python -m proxy_extract extract \
  --video <任意视频>.mp4 --out /tmp/smoke \
  --semantic-backend synthetic --depth-backend synthetic
```

`synthetic` 后端只验证接线。**它的输出是伪造的**，看起来跟真产物一模一样，
别拿它判断质量 —— 详见第 13 节，这个坑已经踩过一次。

---

## 3. 容器路线（备选，不是主路径）

> **先看第 2 节。** 当前维护的是 venv：节点上已有在用的主环境，而这一节要动
> docker daemon、container toolkit 和 root 权限。这一节留着是因为多节点分发和
> 「环境完全隔离」这两种需求容器确实更合适，但它**没有被真正验证过**（见本节
> 第一段和第 9 节第 1 条）。两条路装的是同一个 `requirements.txt`。

### 先说结论：现在还没有可拉的镜像

这个仓库到目前为止只在一台 **arm64、没装 docker、而且还没有 git 仓库** 的 Mac
上开发过，所以从来没有人构建过镜像，registry 上也就没有东西可拉。
`make pull` 现在必然失败，这不是配置错，是还没到那一步。

要先有人构建一次。下面三条路选一条 —— 有 registry 的话直接看路线 B，
把构建交给 CI。

### 关键前提：必须是 linux/amd64

GPU 节点是 x86_64，而**捆绑 CUDA 的 torch wheel 只有 linux/x86_64 版本**。
在 Apple Silicon 上直接 `docker build` 出来的是 arm64 镜像，
到节点上要么根本起不来，要么在模拟层里跑一个没有 CUDA 的 torch。
所以任何跨机器的构建都要显式写平台，`make buildx` 已经把
`--platform linux/amd64` 写死了。

### 路线 A：直接在节点上构建（单节点首选）

最省事，没有 registry、没有跨架构问题，顺带证明 Dockerfile 本身是好的。

```bash
git clone <repo> && cd fastvideo_datapipe
make build          # 节点本身就是 amd64，普通 build 就够
make test           # 期望 373 passed
```

第一次构建大约 10–20 分钟，绝大部分时间花在下载 torch（约 2.5 GB）。

### 路线 B：推到 registry，节点上拉（多节点用这个）

这就是你说的「正常做法」。需要一台 amd64 的构建机（或 CI），以及一个
registry —— 公司自建 Harbor、云厂商的 ACR/SWR、GHCR、Docker Hub 都行。

**推荐让 CI 构建**，仓库里已经有 [`.github/workflows/image.yml`](.github/workflows/image.yml)：
push 到 main 就交叉构建 linux/amd64、在镜像里跑一遍测试、然后推到 registry。
在仓库设置里配四项即可，都留空的话默认发到本仓库的 GHCR：

| 类型 | 名字 | 例子 |
| --- | --- | --- |
| Variable | `REGISTRY` | `registry.example.com` |
| Variable | `IMAGE_NAME` | `your-team/proxy-extract` |
| Variable | `REGISTRY_USERNAME` | `ci-bot` |
| Secret | `REGISTRY_PASSWORD` | （token） |

每次成功推送会打三个 tag：`:0.1.0`、`:<12位 commit sha>`、main 上还有 `:latest`。
**批处理要 pin sha tag**，版本号 tag 会漂。跑完 Actions 页面会直接给出该拉哪个。

想手工构建（没有 CI，或者想先在本地验证），构建机上：

```bash
export REGISTRY=registry.example.com/your-team
docker login $REGISTRY

make buildx REGISTRY=$REGISTRY     # 交叉构建到 linux/amd64 并直接 push
```

`make buildx` 在设了 `REGISTRY` 时用 `--push`，因为 buildx 跨架构构建的产物
没法同时 load 进本地 daemon。想只在本地留一份就不要设 `REGISTRY`。

每个 GPU 节点上：

```bash
export REGISTRY=registry.example.com/your-team
docker login $REGISTRY
make pull REGISTRY=$REGISTRY
```

之后所有命令都带上同一个 `REGISTRY=`，compose 和 `run_shards.sh` 都会用
`IMAGE` 这个变量，不会再去构建：

```bash
make fetch  REGISTRY=$REGISTRY
make qc     REGISTRY=$REGISTRY
```

嫌每条都写太长就 `export IMAGE=registry.example.com/your-team/proxy-extract:0.1.0`，
`IMAGE` 优先级最高，直接覆盖 `REGISTRY`/`NAME`/`VERSION` 拼出来的默认值。

### 路线 C：save / load（机房不通外网）

没有 registry、节点也连不上外网时，用 tar 搬：

```bash
# 构建机
make buildx                                  # 不设 REGISTRY，产物 load 到本地
make save                                    # 得到 proxy-extract-0.1.0.tar.gz（约 6–8 GB）
scp proxy-extract-0.1.0.tar.gz node:/data/

# 节点
cd /data && make load
```

注意权重不在镜像里，所以离线节点还得把权重目录也搬过去：
在能联网的机器上 `make fetch`，然后把 `.hf-cache/` 整个 rsync 到节点，
再用 `HF_CACHE_DIR=/data/.hf-cache` 指过去。

### 路径怎么指

服务器上数据集一般不在仓库旁边，四个变量都可以覆盖，都要用**宿主机绝对路径**：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `IMAGE` | `proxy-extract:0.1.0` | 完整镜像引用 |
| `DATA_DIR` | `./handpick29_high_low` | 语料，容器内挂到 `/work/data`，只读 |
| `OUT_DIR` | `./out` | 产物，容器内挂到 `/work/out` |
| `HF_CACHE_DIR` | `./.hf-cache` | 权重，容器内挂到 `/cache/huggingface` |

`HF_CACHE_DIR` 是宿主机目录而不是 docker named volume，因为拉权重走 compose、
批处理走裸 `docker run`，两边必须落在同一个地方。抽取阶段强制
`HF_HUB_OFFLINE=1`，一旦不一致就不是变慢而是**八张卡同时报错**。
`run_shards.sh` 启动前会检查这个目录非空，缺权重时直接拒跑而不是让你等到失败。

### 整个数据集转换

节点上按顺序（假设已经 pull 好镜像）：

```bash
export IMAGE=registry.example.com/your-team/proxy-extract:0.1.0
export DATA_DIR=/data/handpick29_high_low
export OUT_DIR=/data/proxy_out
export HF_CACHE_DIR=/data/hf-cache

make fetch                        # 一次就够，需要外网
make qc                           # CPU，几分钟，先确认挂载和数据对
make convert N=8                  # 正题：整个 high/ 目录，8 张卡
```

`make convert` 就是 `docker/run_shards.sh` 的封装：每张卡一个容器，
`--shard i/8` 跨步切分 clip 列表，带 `--resume` 和 `--keep-going`。
日志在 `$OUT_DIR/logs/shard-*.log`，产物在 `$OUT_DIR/cond/high/<clip>/`。

中途崩了就**重跑同一条命令**，`--resume` 只补缺的。它靠重新校验文件判断
一条 clip 是否完成，所以写到一半被 kill 的 clip 会校验失败并重做，
不会有半条混进数据集。

跑之前先只跑一条确认端到端是通的，比八张卡一起翻车便宜得多：

```bash
make extract CLIP=26_trevor_seg_0004
make preview CLIP=26_trevor_seg_0004
```

---

## 4. 逐步说明

下面全部用 venv 写法。容器里的等价命令是把 `.venv/bin/python -m proxy_extract`
换成 `docker compose -f docker/docker-compose.yml run --rm extract`，并把路径换成
容器内的 `/work/...`。

### 步骤 0：先确认环境

```bash
nvidia-smi                     # 驱动 >= 525，否则 torch 2.13 起不来
# 每一路交付视频都经 ffmpeg 写出。问代码而不是问 PATH：venv 自带一个静态构建
.venv/bin/python -c "from proxy_extract.proxy import ffmpeg_binary; print(ffmpeg_binary())"
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 期望：True 8
ls handpick29_high_low         # 应该有 camera/ high/ low/ manifest.json
```

数据集目录的约定（QC 命令依赖它）：

```
handpick29_high_low/
├── high/<clip>.mp4      高质量渲染
├── low/<clip>.mp4       低模渲染
├── camera/<clip>.json   GT 相机轨迹（COLMAP 导出）
└── manifest.json
```

### 步骤 1：自检

不需要 GPU、不需要权重，先确认这个 venv 本身是好的：

```bash
.venv/bin/python -m pytest proxy-extract/tests -q
# 期望：373 passed
```

`setup_venv.sh` 结尾已经跑过一次。单独重跑是在改完代码之后。

`torch.cuda.is_available()` 打印 `False` 是宿主机/驱动的问题，见第 8 节。

### 步骤 2：拉权重

```bash
.venv/bin/python scripts/fetch_models.py --set default
```

国内网络慢的话先 `export HF_ENDPOINT=https://hf-mirror.com`。

默认拉的是推荐配置需要的两个模型：Mask2Former（语义）和 Depth Anything V2
Metric Outdoor（深度）。别的组合见 `docker/README.md` 的表。

权重位置由 `HF_HOME` 决定，**显式指到大盘上**，别让它落到家目录：

```bash
export HF_HOME=/data/binghe/cache/huggingface
```

拉完之后批处理建议加 `export HF_HUB_OFFLINE=1` —— 这不是洁癖：8 个 worker 同时打
hub 会互相限速，批处理跑到一半 hub 抽风会留下一个处理了一半的数据集。

### 步骤 3：挑片（camera-qc）

**这是最该先跑的一步**，而且它不需要 GPU 也不需要权重。

它做的事：在视频里跟踪稀疏特征点，用 GT 相机位姿算对极几何，
看特征点偏离对极线多少像素（Sampson 距离）。位姿是可信的，
所以残差大就说明**这一版渲染的几何跟位姿对不上**。

```bash
.venv/bin/python -m proxy_extract camera-qc \
  --dataset handpick29_high_low --track high --report out/camera_qc.json
```

真实输出（29 条 clip 的 high 渲染）：

```
01_john_marston_seg_0014         poses:match     fidelity:keep     sampson    0.57 px  inlier  0.95
02_john_marston_seg_0017         poses:match     fidelity:keep     sampson    0.28 px  inlier  0.98
...
18_michael_seg_0043              poses:loose     fidelity:review   sampson    1.90 px  inlier  0.56
...
29 clips: 28 keep, 1 review, 0 drop
```

换成 low 渲染（`--track low`），同样的位姿、同样的场景：

```
29 clips: 10 keep, 11 review, 8 drop
```

两列的含义不一样，别混：

| 列 | 判定的是 | 阈值 |
| --- | --- | --- |
| `poses:` | 这组位姿是不是这段视频的 | `match` < 2 px 且 inlier > 0.6；`mismatch` >= 10 px |
| `fidelity:` | 这一版渲染的几何有多忠实 | `keep` <= 1 px，`review` <= 3 px，`drop` > 3 px |

**入库只收 `fidelity:keep`。** high 里 28/29 达标而 low 里只有 10/29，
这个差距就是「深度和语义从 high 提取、low 只当 RGB 通道用」这条结论的依据。

### 步骤 4：抽一条 clip

```bash
.venv/bin/python -m proxy_extract extract \
  --video handpick29_high_low/high/26_trevor_seg_0004.mp4 \
  --out out/cond \
  --semantic-backend coarse6 \
  --depth-backend depth_anything
```

一行日志：

```
26_trevor_seg_0004: 124 frames, depth backend_native median 13.21 m, flicker 0.0288 -> 0.0233
```

`flicker` 是相邻帧标签跳变的比例，箭头前后是时域稳定化处理前/后。
降不下来说明分割在这段视频上本来就不稳，值得去看一眼 preview。

参考耗时：单条 124 帧在一台 M 系笔记本（MPS）上 85–95 秒，A100/H800 会快很多。
输出体积约 31 MB / clip（249 个文件）。

### 步骤 5：看结果

```bash
.venv/bin/python -m proxy_extract preview \
  --condition-root out/cond/high/26_trevor_seg_0004 \
  --out out/preview.mp4
```

交付格式（`scenes` 的产物）是另一个命令，见第 13 节的 `scenes-preview`。

preview 会自己从 `extraction_report.json` 里读出用的是 6 类还是 12 类，
用对应的调色板。**不要**手动把 6 类结果按 12 类的颜色看 —— ID 5 在
6 类里是 `hero`，在 12 类里是 `vegetation`，看起来完全合理但是错的。

### 步骤 6：校验

```bash
.venv/bin/python -m proxy_extract validate \
  --condition-root out/cond/high/26_trevor_seg_0004 --expect-frames 124
```

这一步用的是跟 CWM 加载器同一套检查（字节数、分辨率、深度范围、ID 范围）。
过了就是 CWM 能读。

### 步骤 7：多卡批处理

交付场景用 `scripts/run_scenes.sh`（第 12 节），它是 venv 原生的。
condition_root 的批处理目前只有容器版封装 `docker/run_shards.sh`；venv 下手写：

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python -m proxy_extract extract \
    --video handpick29_high_low/high --out out/cond \
    --semantic-backend coarse6 --depth-backend depth_anything \
    --shard $i/8 --resume --keep-going \
    >out/logs/shard-$i.log 2>&1 &
done
wait
```

一张卡一个进程，`--shard i/8` 把 clip 列表按位置切开互不重叠。日志在
`out/logs/shard-*.log`。

三个标志值得知道它们为什么在：

- `--shard i/N`：**跨步**切分而不是连续切分。如果耗时沿列表递增（长片排在后面），
  连续切分会让最后一个 worker 扛下整条尾巴。
- `--resume`：跳过已完成的 clip。判断方式是**重新校验文件**而不是看标记文件，
  所以一个写到一半被 kill 的 clip 会校验失败并被重做，而不是半条混进数据集。
- `--keep-going`：一条 clip 炸了不带走同 shard 剩下的。

崩了直接重跑同一条命令，`--resume` 会只补缺的。

`--video` 可以直接给目录，会自动展开成里面所有的视频文件并按名字排序 ——
排序是必须的，每个 worker 都要推导出完全一致的列表，否则会有 clip 被处理两次
或者一次都没有。

---

## 5. venv 的维护

### 只装不需要模型的部分

合约、分类体系、QC、编码这几层都不 import torch（后端是懒加载的），所以纯 CPU
机器上可以不装那 3 GB：

```bash
EXTRAS=core scripts/setup_venv.sh
```

这时 `--depth-backend`/`--semantic-backend` 只有 `synthetic` 可用，其余会在
调用时报 ImportError，不会在启动时。

### 改完代码之后

`proxy-extract` 是以 editable 装的，改源码不用重装。重装只在这三种情况下需要：

| 情况 | 做什么 |
| --- | --- |
| 改了 `pyproject.toml` 的依赖或 entry point | `.venv/bin/python -m pip install --no-deps -e proxy-extract` |
| 改了 `requirements.txt` 的 pin | `.venv/bin/python -m pip install -r requirements.txt` |
| venv 被移动或拷贝过 | 重建。里面全是绝对路径，改不干净 |

### 别把它跟主环境混起来

`setup_venv.sh` 建的是不继承 `site-packages` 的干净 venv，这是它的全部意义。
两个会破坏这一点的动作：

- 在已经 activate 了 conda/主环境的 shell 里 `pip install` 到全局；
- 用 `$VENV/bin/pip` 而不是 `$VENV/bin/python -m pip`（venv 移动过就会打到别处）。

要确认当前用的到底是哪个解释器：

```bash
.venv/bin/python -c "import proxy_extract, sys; print(sys.prefix); print(proxy_extract.__file__)"
```

`sys.prefix` 必须是这个 venv 的路径。不是的话就是装串了。

---

## 6. 输出怎么读

```
out/cond/high/26_trevor_seg_0004/
├── 000000.depth.f32          336*192*4 = 258048 字节，raw float32，单位米
├── 000000.semantic_id.png    336x192 8-bit，像素值就是类别 ID
├── ...
└── extraction_report.json
```

`.semantic_id.png` 用图片查看器打开是一片纯黑 —— ID 只有 0-5，肉眼看不出来，
这是正常的。要看就用 `preview`。

`extraction_report.json` 里值得看的字段：

| 字段 | 含义 | 不对劲的信号 |
| --- | --- | --- |
| `depth.metric_source` | 米制尺度哪来的 | `backend_native` = 模型自己出的米；`cameras` = 用 GT 相机标定过 |
| `depth.clipped_far_fraction` | 被 256 m 截掉的比例 | 明显大于 0 说明场景超出 CWM 的深度量程 |
| `semantic.flicker_before/after` | 时域抖动 | after 没降下来说明分割不稳 |
| `semantic.class_fractions` | 各类占比 | 某类为 0 或占满，多半是映射或场景不匹配 |
| `semantic.hero_split.resolved` | 主角有没有拆出来 | `false` 时看 `note`，模块选择不猜而不是瞎猜 |
| `semantic.hero_split.merged_frames` | 人物 mask 粘连的帧占比 | 偏高时主角判定不可信 |
| `validation` | 用 CWM 的检查重读一遍 | 有异常会直接抛错，不会静默 |

`semantic.meta.unmapped_source_labels` 很长是正常的：ADE20K 有 150 类，
6 类体系里只显式映射了道路/植被/车/人，其余全部落到 `background`。

---

## 7. 后端怎么选

```
--semantic-backend  coarse6     6 类粗分类（推荐，本项目的目标体系）
                    ade20k      12 类 CWM 原生，Mask2Former
                    cityscapes  12 类，SegFormer，街景场景更准但没有室内类
                    synthetic   假数据，只用来验证接线

--depth-backend     depth_anything_v3  DA3 嵌套模型，米制（默认，见下）
                    depth_anything     DA V2 单帧米制，装完即用的兜底
                    mapanything        多帧+可吃 GT 相机，但取权重会卡（见下）
                    synthetic          假数据
```

选后端的实际约束不是精度，而是**权重能不能在这台机器上拿到**，以及**它肯不肯声明自己
是米制**——`scenes` 会拒绝非米制深度，因为交付视频的 log-z 编码需要真实尺度。

**`depth_anything_v3` 是默认值。** 它是三个里唯一同时满足两个条件的：DINOv2 主干
烘焙在它自己的 `model.safetensors` 里，所以**只要连得上 HF 就能拿全所有权重**，不像
mapanything 那样另外走 `torch.hub`；而且它的嵌套 checkpoint 会用自己预测的焦距完成
换算并置 `is_metric = 1`。

代价有三个，都得先知道：

1. **不在 PyPI，而且它声明的 pin 跟本环境冲突**（`numpy<2`，我们是 2.5.1；
   `requires-python <=3.13`，我们是 3.14）。直接装会把 numpy 降级、把 torch 拽走，
   所以必须绕开它的依赖解析：

   ```bash
   .venv/bin/python -m pip install --no-deps --ignore-requires-python \
       git+https://github.com/ByteDance-Seed/depth-anything-3
   .venv/bin/python -m pip install einops omegaconf addict imageio
   .venv/bin/python scripts/fetch_models.py --set da3      # 6.8 GB
   ```

   `--no-deps` 会漏掉 `gsplat` / `open3d` / `pycolmap` / `moviepy` / `evo`，那些只服务
   高斯导出和多视角位姿对齐。后端会在导入前给这两个子模块装上占位实现，占位被真的调用
   才会报错——单目路径永远不会走到（源码里 `if extrinsics is None: return`）。

2. **权重是 CC BY-NC 4.0**，只能研究用。想要 Apache-2.0 的话只有
   `DA3METRIC-LARGE`（`--set da3-apache`），但它是 DinoV2+DPT、**没有相机头**：输出是
   canonical 深度、从不设置 `is_metric`、也给不出换算所需的焦距，所以 `scenes` 会直接
   拒收。要用它就得由外部补焦距（ABot 的 COLMAP `cameras.txt` 里有像素焦距，且像素
   焦距不受 COLMAP 尺度不确定性影响）——这条路还没实现。

3. **它不输出天空掩码。** Apache 那个 metric-large 会给 `sky`，嵌套这个给的是 `None`。
   所以天空只能由语义分支认定，`duv.mp4` 靠 `ids == sky` 仍然正确，但 `depth.mp4` 的
   0 哨兵拿不到它。这跟 `depth_anything` / `mapanything` 的现状一样，不是新问题。

**`depth_anything`（V2）是不想折腾时的兜底。** `requirements.txt` 覆盖它的依赖（只要
transformers），`fetch_models.py --set default` 拉的也正是它的权重。它逐帧预测，帧与帧
之间的尺度没有绑定，这个 caveat 会写进每份 report 的 `depth.meta.caveat`。

**`mapanything` 要单独装，而且不在 PyPI 上。**

```bash
.venv/bin/python -m pip install 'git+https://github.com/facebookresearch/map-anything'
.venv/bin/python scripts/fetch_models.py --set mapanything
```

权重是 CC-BY-NC 且在 hub 上是 gated 的，要先 `hf auth login` 并接受条款；商用要用
`facebook/map-anything-apache`。

**mapanything 还要第三份权重，而 `fetch_models.py` 拿不到它。** MapAnything 的
DINOv2 主干是走 `torch.hub` 拉的，宿主是 `dl.fbaipublicfiles.com` —— 跟 HF 无关，
所以 `HF_ENDPOINT` 和 `HF_HUB_OFFLINE=1` 对它都不起作用。很多集群的出网白名单只放行
了 HF / PyPI 这一类，这个域名不在里面（跟节点在哪个国家无关），症状是日志停在

```
Using cache found in ~/.cache/torch/hub/facebookresearch_dinov2_main
```

然后**不报错、GPU 零占用**。那行说的是仓库代码已缓存，卡住的是紧随其后的 1.2 GB
权重下载。确认方法：

```bash
ls -la ~/.cache/torch/hub/checkpoints/            # 空的就是没下下来
curl -sI --max-time 10 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
```

两条出路。一是在能出网的机器上跑一次预检，让它把 `torch.hub` 缓存填满，再整个拷到
节点上 —— 这样不用猜 MapAnything 用的是哪个 DINOv2 变体：

```bash
# 能出网的机器
DEPTH=mapanything scripts/run_scenes.sh    # 预检会拉全，然后 Ctrl-C
rsync -a ~/.cache/torch/ node:~/.cache/torch/
```

二是直接下（`TORCH_HOME` 没设时就是这个路径），但要先从 traceback 里确认变体：

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
```

`depth_anything_v3` 和 `depth_anything` 都没有这个问题，它们的权重全在 HF 上。这正是
默认值换成 DA3 的理由：mapanything 的多帧一致性确实更好，但取不到权重的后端等于没有。

DA3 也支持多帧联合重建（它是 any-view 模型）：

```bash
--depth-backend-option window=4        # 窗口内的帧当作同一场景的多个视角
--depth-backend-option process_res=728 # 默认 504；调高更清晰也更慢
```

`window` 大于 1 时该窗口内的尺度绑定在一起。默认 `window=1`，即逐帧——动态场景下
多视角假设不成立，所以不默认打开。

`run_scenes.sh` 启动前会**真的把两个后端各跑一次单帧推理**，所以后端装错或权重没拉
会在几秒内失败，而不是在 2000 条上各失败一次。`--keep-going` 也不再吞 ImportError：
那说明机器不对，不是这条 episode 不对。

---

## 8. 出错了怎么办

venv 相关的先看这几条：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `bad interpreter` / `pip` 找不到 | venv 建好后被移动或拷贝过，`pip` 脚本的 shebang 是过期绝对路径 | 用 `.venv/bin/python -m pip`；venv 移动过就重建 |
| `ModuleNotFoundError: proxy_extract` | 装到了主环境而不是 venv，或没装 | `.venv/bin/python -m pip install --no-deps -e proxy-extract` |
| `error: need Python >= 3.10` | 系统 `python3` 常常是 3.9 | `PYTHON=/path/to/python3.12 scripts/setup_venv.sh` |
| `sys.prefix` 不是 venv 路径 | shell 里 activate 了 conda/主环境 | `deactivate`，或直接用 `.venv/bin/python` 全路径调用 |
| 报错说找不到 ffmpeg | 三条出路都写在报错里 | 见第 2 节「ffmpeg 从哪来」 |
| `No module named 'mapanything'` | 它不在 PyPI，`setup_venv.sh` 不会装它 | `DEPTH=depth_anything`，或按第 7 节从 git 装 |

模型和数据相关：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 日志里 `CUDA initialization: The NVIDIA driver on your system is too old`，随后 `Non-CUDA device detected` | torch 的 wheel 是按比驱动更新的 CUDA 构建的，于是**整批 worker 静默退回 CPU**——`nvidia-smi` 照样列出所有卡，`N_GPUS` 照样数对 | 对比 `python -c "import torch; print(torch.version.cuda)"` 和 `nvidia-smi --query-gpu=driver_version --format=csv`；驱动旧就按驱动的 CUDA 重装 torch，如 `--index-url https://download.pytorch.org/whl/cu124`。容器里改不了驱动，只能换 torch。`run_scenes.sh` 现在会在启动前拦下这种情况 |
| 换过 torch 之后 `OSError: Could not load this library: .../torchaudio/lib/_torchaudio.abi3.so` | torchaudio 的 C++ 扩展是按换掉之前那个 torch 编译的。transformers 导入图像处理器时会路过 `audio_utils`，于是被牵连——跟音频无关 | `pip uninstall -y torchaudio`。它不在 `requirements.txt` 里，transformers 那处 import 由 `is_torchaudio_available()` 守着，包不在就整段跳过。**不要**去装"匹配版本"：torchaudio 停在 2.11，没有配 torch 2.13 的构建 |
| 确实想用 CPU 跑 | — | `N_GPUS=1 ALLOW_CPU=1 scripts/run_scenes.sh` |
| 下权重卡住 / 超时 | hub 网络 | 先确认是不是集群出网白名单的问题（见第 7 节）；国内节点才需要 `export HF_ENDPOINT=https://hf-mirror.com` |
| `OSError: ... preprocessor_config.json` | 权重没拉全就跑了离线模式 | 重跑 fetch，确认 `--set` 覆盖了你用的后端 |
| `no such video: ...` | 路径不对；ABot 那种嵌套目录要加 `--recursive` | 见第 11 节 |
| `no camera tracks under ...` | `camera/` 目录不在或没有 json | 检查数据集目录结构 |
| clip 少于 124 帧 | CWM 窗口就是 124 帧 | 这条 clip 用不了，不是 bug |
| preview 颜色不对 | 手动指定了错的调色板 | 别传，让它自己从 report 读 |
| 交付的语义看着完全不对 | 很可能用了 `synthetic` 占位后端 | 查 report 的 `deliverable` 字段，见第 13 节 |
| 某个 shard 挂了 | 看 `<out>/logs/shard-N.log` | 修完重跑同一条命令，`--resume` 会补 |

---

## 9. 已知限制

写在前面，免得当成 bug 去查：

1. **Docker 镜像没被真正构建过，registry 上也没有。** 写这份文档的机器是
   arm64 且没有 docker daemon。第一次 `docker build` 请当成需要盯着的事，
   而且必须构建成 `linux/amd64`（第 3 节）。**维护的是 venv 路线**（第 2 节），
   它跟镜像装的是同一个 `requirements.txt`，所以容器路线滞后不影响交付。
2. **hero/npc 拆分只在一条 clip 上被真实验证过。** 29 条样例里只有
   `11_john_marston_seg_0313` 有其他人长时间在画面里（44% 的帧）。逻辑本身有
   单元测试覆盖，但「在人多的场景里靠谱吗」这个问题，样本量不足以回答。
3. **`vegetation` 在 low 渲染上的迁移很差。** 这是推荐「从 high 提取」的原因之一，
   细节见 `experiments/README.md`。
4. **深度是逐帧的**，见第 7 节。
5. **GT 相机的平移不是米制的**（COLMAP 尺度）。需要米制尺度时要额外标定，
   `experiments/fig_scale.py` 和 `fig_human_scale.py` 是两个独立的估计方法。

---

## 10. 图册怎么看

**图不在仓库里**（游戏画面版权），要自己画一遍。需要语料库 `handpick29_high_low/`
和一张卡：

```bash
# 先跑一次抽取，fig_hero / fig_condition 读的是它的输出
make extract CLIP=26_trevor_seg_0004

PYTHONPATH=proxy-extract/src python3 experiments/fig_camera_qc.py
PYTHONPATH=proxy-extract/src python3 experiments/fig_appearance.py
# ...其余 fig_*.py 同理，每个脚本的输入输出见 experiments/README.md

make gallery          # 把 experiments/figures/ 组装成一页
open gallery/index.html
```

`make gallery` 只做拷贝和排版：缺哪张图它就报 `figures missing` 退出，
而不是留一个空占位，所以图册要么是完整的实测结果要么根本不生成。

**只想看结论不想重跑**：数字都在 `experiments/*.json` 里，
文字解读在 [`experiments/README.md`](experiments/README.md)，两者都在版本控制内。

图册的顺序和看点：

| # | 图 | 看什么 |
| --- | --- | --- |
| 01 | `condition_output.png` | 正确的 condition 长这样 |
| 02 | `hero_split.png` | 主角是跟出来的，不是分割出来的 |
| 03–04 | `camera_qc_*.png` | 为什么从 high 抽、用对极 QC 卡 low |
| 05 | `appearance_gap.png` | 低模还剩布局和剪影 |
| 06–08 | `extraction_*.png` / `feasibility.png` / `coarse6.png` | 深度能迁，植被迁不过去 |
| 09–10 | `scale_*.png` | GT 相机不是米制的 |
| 12–15 | `duv_*.mp4` | 可播放的 preview |

数字和结论写在 `experiments/README.md`。镜像内部写在 `docker/README.md`。

---

## 11. ABot-World-Explorer-500h

跟 gta-web 不同：这个语料**只有 RGB 和一个 COLMAP 稀疏模型**，深度和语义都没有，
所以整条预测链路对它才是必需的而不是备选。

### 它跟 handpick29 有三处不一样

| | handpick29 / gta-web | ABot |
| --- | --- | --- |
| 布局 | `high/<clip>.mp4` 一层 | `data/<前缀>/<sample_id>/video.mp4` 两层 |
| 每段长度 | 124 帧（正好一个窗口） | 1800 帧 |
| 标注 | 引擎输出的 depth / semantic | 只有 `annotations.tar`（动作、字幕、COLMAP） |

三处都会影响命令怎么写：

**布局要 `--recursive`。** `--video` 默认只看一层目录，指向 ABot 根目录会报
`no .mp4 files in ...`。

**1800 帧要 `--chunk-frames`。** 契约窗口是 124、步长 90，所以合法长度只能是
`124+90n`；1800 帧取 19 个窗口 = **1744 帧**，尾部 56 帧丢弃，这是契约决定的不是 bug。
真正的问题是内存：管线原本一次性把整段读进来，1744 帧在 1344×768 下光 RGB 就是
5.4 GB，由它导出的全分辨率深度栈还要再 7.2 GB。`--chunk-frames` 让模型分批跑、
每批降到 336×192 就丢掉，实测峰值从 7.7 GiB（仅 484 帧时）降到 5.73 GiB（完整 1744 帧）。

分批**不会改变结果**：降采样用的中值/最小值/均值都与正的缩放系数可交换，所以
「先降采样再标定」和原来的顺序逐字节一致，`tests/test_chunking.py` 盯着这件事。
代价是深度后端的跨帧推理会在每个批次接缝处断掉 —— `depth_anything` 是逐帧的，
不受影响；`mapanything` 靠联合观察整段获得时序一致性，分批会削弱它。

**COLMAP 相机不能和分批一起用。** 每批是独立重建，预测出的位姿不共享坐标系，
没法解出一个全局尺度，所以两个一起给会直接报错而不是悄悄算错。ABot 的 COLMAP
平移本来也不是米制的。

### 跑

```bash
python -m proxy_extract extract \
  --video /data/binghe/datasets/ABot-World-Explorer-subset2000/data \
  --recursive \
  --out /data/binghe/datasets/abot_cond \
  --semantic-backend standard11 \
  --depth-backend depth_anything \
  --chunk-frames 124 \
  --emit-videos all \
  --resume --keep-going
```

多卡照旧加 `--shard i/N`，一张卡一个进程。

产物落在 `<out>/<sample_id>/video/`（目录名取自视频的父目录，ABot 里就是
`sample_id`，所以 2000 条不会撞名）。

### 体积要先算再跑

单条 1744 帧约 **435 MiB**（深度 `.f32` 就占 429 MiB），2000 条约 **850 GiB**、
接近 **700 万个文件**。比 RGB 本身（2000 条约 164 GiB）大五倍，落到 ceph 上
inode 也要先确认够。只想先验证的话，深度换成 `--depth-downsample min` 不会省空间，
真要省只能少下 episode。

### 交付视频

`--emit-videos all` 会在每个 condition_root 旁边按 `DATA_F.md` 的编码多写三个文件：

```
depth.mp4      反向 log-z 灰度，near 0.1 / far 256
semantic.mp4   无损 RGB，(R,G,B) = (0,0,id)
proxy.mp4      R = log-z(near 0.1 / far 8000)，G/B = 语义色
```

`--emit-videos proxy` 只写 proxy。已经抽好的 condition_root 想补视频，
用 `python -m proxy_extract videos --condition-root <dir>`，不必重跑模型。

语义视频必须无损 RGB（`libx264rgb`），这不是讲究：小整数 ID 走 YUV 管线会被色度
下采样在类别边界上混成从没预测过的类，而且下游查不出来。`tests/test_proxy.py`
用真实数据核对过三种编码都是逐字节无损的。

> **proxy 的 R 通道方向是推断的，不是文档写明的。** `DATA_F.md` 给了 depth 视频
> 明确的反向公式（近 = 高码值），但 proxy 的 R 只写了 near/far 和「压到 [0,254]，
> 天空 = 255」，没说方向。这里默认沿用同一份文档里唯一给出的那个方向（反向），
> 想要相反的用 `videos --forward-proxy-depth`。**如果 gta-web 录制端的
> `scripts/compose-proxy.mjs` 能拿到，应当以它为准**：方向搞反不会报错，只会
> 悄悄毁掉整个数据集的深度通道。

`ego`（G/B = 128/0）需要知道该段主角在不在开车，这个标志来自 `standard11` 的
player/ped 拆分结果里的 `driving`。ABot 没有 gta-web 那样的 `tag`，所以这完全依赖
预测；不用 `standard11` 时所有载具都会落到普通 `vehicle`（64/0）。

---

## 12. 720p 交付场景（`scenes`）

上一节产出的是 code-world-model 吃的 `condition_root`：336×192 的逐帧 `.f32` +
PNG。这一节产出的是**另一种交付物** —— `DATA_F.md` 规定的那一套，1280×720、
整段长度、四路视频对齐：

```
<out>/
  scenes_manifest.json      scene 编号 ↔ sample_id 的映射，provenance 全在这
  seg_long_000000/
    color.mp4               RGB，libx264 / yuv420p，CRF 16（唯一有损的一路）
    depth.mp4               反向 log-z 灰度，near 0.1 / far 256，libx264 无损
    semantic.mp4            (R,G,B) = (0,0,id)，libx264rgb 无损
    duv.mp4                 R = log-z（near 0.1 / far 8000，天空 255），
                            G/B = 语义色，libx264rgb 无损
    annotation.tar          episode 自带的标注，原样拷贝
    extraction_report.json
  seg_long_000001/
  ...
```

`depth.mp4` 和 `duv.mp4` 的 R 通道**量程和方向都不一样**，别混着读：前者
near 0.1 / far 256 且反向（近 = 高码值），后者 near 0.1 / far 8000 且正向
（近 = 0，远 = 254，255 留给天空）。方向的取舍见第 12 节末尾。

### 跑

多卡直接用封装脚本（venv 原生，自带前置检查、分片、resume、结尾审计）：

```bash
DATA_DIR=/data/binghe/datasets/ABot-World-Explorer-subset2000/data \
OUT_DIR=/data/binghe/datasets/abot_scenes \
scripts/run_scenes.sh
```

它默认按 `nvidia-smi` 数出的卡数开进程。单条手跑：

```bash
.venv/bin/python -m proxy_extract scenes \
  --video /data/binghe/datasets/ABot-World-Explorer-subset2000/data \
  --recursive \
  --out /data/binghe/datasets/abot_scenes \
  --semantic-backend standard11 \
  --depth-backend depth_anything_v3 \
  --resume --keep-going
```

多卡加 `--shard i/N`，一张卡一个进程。

### 跟 `extract` 的四处关键差别

**不做 336×192 降采样。** 降采样会丢掉 93% 的像素，而从这四路视频重做一遍很便宜；
现在就做只会让网格的选择永远无法回头。降采样属于后处理。

**不截帧。** `extract` 必须把 1800 帧截到 `124+90n = 1744` 帧以对齐契约窗口；
`scenes` 按源帧率交付全部 1800 帧，切段也留给后处理。

**1280×720 正好是 1920×1080 的 2/3**，所以缩放不引入形变。更要紧的是，这正是
`semantic.player` 的先验（锚点 `(0.5, 0.55)`、`max_anchor_distance` 等）当初在
gta-web 真实语料上拟合的分辨率 —— gta-web 的 semantic 视频本身就是 1280×720。
所以 player/ped 拆分在这里跑的是原生像素，不是代理网格。

**color 视频从模型看到的同一批解码帧编码。** 不是把源文件另外交给 ffmpeg 缩放：
两个重采样器不会逐像素一致，而一套 RGB 跟自己的 depth 差半个像素的交付数据，
对任何要学对应关系的下游来说比没有更糟。

### 实测（真实 ABot episode 前 600 帧，本机 CPU，合成后端）

| | 无光流 | `--flow-downscale 2`（默认） | `--flow-downscale 1` |
| --- | --- | --- | --- |
| 耗时 | 159 s | 222 s | 412 s |
| 光流净成本 | — | 63 s | 253 s |
| `flicker_after` | 0.0382 | 0.053909 | 0.053934 |
| 四路视频合计 | 138 MiB | 154 MiB | 154 MiB |

**降采样解光流几乎不花代价**：ds=2 和 ds=1 的 flicker 差 0.05%，而光流开销是 1/4，
符合 Farneback 对像素数的平方关系。默认值就按这个定的。

> 这里**不能拿「无光流」的 flicker 更低当成它更好**。`flicker_rate` 数的是逐帧
> 变类的像素比例；不做流补偿时投票会在未对齐的窗口上把移动内容抹平，标签「粘」住
> 了，于是这个指标反而更低 —— 那是边缘涂抹，不是稳定。上面 ds1/ds2 的对比是在
> 同样开启光流的前提下比的，才是可比的。

单帧字节数（600 帧实测除以帧数，proxy 最大是因为 R 通道扛着 log-z 的全部细节）：

| 流 | 600 帧 | 折算 1800 帧 |
| --- | --- | --- |
| `duv.mp4` | 68.3 MiB | 205 MiB |
| `depth.mp4` | 44.4 MiB | 133 MiB |
| `color.mp4` | 29.4 MiB | 88 MiB |
| `semantic.mp4` | 12.2 MiB | 37 MiB |
| 合计 + tar | 155 MiB | **≈ 465 MiB** |

所以 **2000 条约 0.89 TiB**，是源 RGB（约 211 GiB）的 4.3 倍。总量跟
`condition_root` 那条路（约 850 GiB）差不多，但**文件数从约 700 万降到约 1.2 万** ——
深度进了无损视频而不是逐帧 `.f32`，每个 scene 只有 6 个文件。ceph 上这是决定性的差别。

### 内存：每个 worker 约 40 GiB，这决定开几个进程

600 帧实测峰值 RSS **13.3 GiB**。每一项都是「每帧数组 × 帧数」，没有别的量级项，
所以线性外推到 1800 帧是 **约 40 GiB**（第一次在 H200 上跑满 1800 帧时请复核这个数）。

内存随 episode 长度走，不随窗口走，因为时序稳定和主角跟踪都跑在整段上 —— 这样它们
跟测试覆盖的批处理行为完全等价，不需要为流式再引入一套近似。峰值里除了三个主栈
（深度 6.6 GB、标签 1.7 GB、光流引导 1.7 GB）之外，还有稳定器在释放输入前先分配的
输出、`np.concatenate` 的双份、以及 range guard 的临时量。

`--chunk-frames`（默认 64）管的是 GPU 上单次前向的激活量，**不管**这些主机端的栈。

**开 8 个 worker 就要约 320 GiB 主机内存。** 如果这不够，能省的地方按性价比排：
按 `probe` 的帧数预分配以消掉 `concatenate` 的双份（约省 6.6 GB）、
range guard 改成原地（再省 6.6 GB）、深度栈降到 float16（再省 3.3 GB，
而 8-bit 对数量化的步长是 3.1%，float16 的精度远远够）。这些都还没做，
因为当前的判断是主机内存不是瓶颈。

### 会拒绝什么

深度后端返回的若是 up-to-scale（非米制）深度，`scenes` 会**直接报错**而不是交付。
交付视频编的是绝对米制，而 ABot 的 COLMAP 模型自己只定义到一个相似变换，
补不上这个尺度。所以这里必须用能预测米制深度的后端。

### 编号与 provenance

`seg_long_000000` 往上按 **sample_id 字典序**编号（6 位，够整个 500h 语料用），
映射写在 `scenes_manifest.json`。
排序而不是按发现顺序，有两个后果：分片的 worker 不用互相协调就能得出同一套编号；
后来新增 episode 只会插入、不会把已交付的重新编号，而这一点 manifest 的 diff 看得见。

重编号会丢掉数据集自己的标识符，所以 manifest 不是可选的 —— 没有它，交付集就没法
回溯到源语料，某个 scene 出问题也查不到是哪条 episode。

`annotation.tar` 是**原样拷贝**：里面是数据集自己的声明（动作、字幕、COLMAP），
重新打包会让这条管线变成它并未产出的数据的第二个真相来源。

## 13. 验收交付场景

### 交付格式是设计上不可直视的

`semantic.mp4` 把类别 ID 0–10 放在蓝通道，直接播放几乎全黑；`depth.mp4` 是 log-z
灰度斜坡，看起来是一片平灰。所以**不要用播放器判断质量**，用：

```bash
python -m proxy_extract scenes-preview \
  --scene /data/binghe/datasets/abot_scenes/seg_long_000000 \
  --out /tmp/sheet.png --frames 6
```

`.png` 出等距采样的 contact sheet（`color | semantic | depth` 三联 + 类别图例），
适合 scp 回来看；后缀换成 `.mp4` 出同样面板的视频。

### 合成后端会伪造输出，而且看起来完全正常

`synthetic` 深度/语义后端存在的意义是无 GPU 时验证管线，它们产出的视频在结构上
跟真实产物**不可区分** —— 同样的编码、同样的类别 ID、同样的报告字段。这已经害过
一次：一张用 `synthetic` 出的对照图被当成了「真实分割器失效」的证据。

现在有三道防线：

- `run_scenes.sh` 直接**拒绝** `SEMANTIC=synthetic` 或 `DEPTH=synthetic`，除非显式
  设 `ALLOW_SYNTHETIC=1`；
- `extract_scene` 在写盘时抛 `PlaceholderOutput` 警告；
- `extraction_report.json` 里带 `"deliverable": false` 和 `"placeholder_backends"`。

验收任何一批数据，先 `grep -L '"deliverable": true' */extraction_report.json`。

### 真后端在 ABot 上长什么样

在真实 episode 上实测（`standard11` = Mask2Former swin-large ADE20K，本机 CPU
2.9 s/帧），荒野徒步素材的构成大致是：

| | 占比 |
| --- | --- |
| vegetation | 72–88% |
| terrain | 6–25% |
| sky | 2.5–17% |
| person | 1.7–2.2% |

草地归 vegetation 而不是 terrain，是 DATA_F.md 明确规定的（`7 vegetation = 树、草、
灌木`；`8 terrain = 山地、岩石、野外地面`），不是映射错误。

主角判定在同一素材上跑 90 帧连续窗口：`resolved: True`，轨迹全程 90/90 帧，中心距
锚点 0.08 帧宽，中位面积 1.53%，merged 0%。掩码是贴合的人形轮廓。

### 坐骑：`animal` → `vehicle`

标准的 11 类里没有动物类。被骑乘的马是载具，归 `prop` 会告诉世界模型「一个大体积
移动主体是静态杂物」，这是两种可选错误里更有害的一种，所以 `animal` 映射到
`vehicle`（3）。

代价要写明：ADE20K 只有**一个** `animal` 标签，分不出马、鹿、狗、鸟，所以这个映射
连带把野生动物也算作 vehicle。要把坐骑和野生动物分开需要实例掩码，映射表做不到。
实测两条 episode 共 24 帧里 `animal` 命中 0 像素，所以这个代价的实际暴露面很小。
