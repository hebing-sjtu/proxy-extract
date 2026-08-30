# Runbook：从零跑通 proxy-extract

面向第一次接触这个仓库的人。不需要先读论文，也不需要先读源码。

代码注释和 `proxy-extract/README.md`、`experiments/README.md` 是英文的。
**这份是操作手册**：怎么跑、看什么、怎样算对、出错了怎么办。

建议顺序：

1. 有 Docker + NVIDIA 卡：跑 `./docker/first_run.sh`（约 15–30 分钟，首次含构建和拉权重）
2. 没有卡、只想摸命令：用第 5 节的本地 CPU 跑法

`make help` 是同一套步骤的单词入口。

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

## 2. 一键跑通（推荐）

前提：有 Docker、有 NVIDIA 驱动（`nvidia-smi` 能打印出卡）、仓库根下有 `handpick29_high_low/`。

```bash
cd /path/to/fastvideo_datapipe
chmod +x docker/first_run.sh docker/run_shards.sh
./docker/first_run.sh
```

它按顺序做五件事：构建镜像 → 自检（应看到 **229 passed**）→ 拉权重 → 相机 QC → 抽一条 clip 并渲 preview。

**这是给「本机就有卡」的场景准备的。** 要在远端服务器上拉镜像跑，看第 3 节。

跑完打开：

| 文件 | 是什么 |
| --- | --- |
| `out/preview.mp4` | 你刚抽出的结果：左深度、右语义、底栏图例 |
| `out/camera_qc.json` | 29 条 clip 的对极残差 |
| `out/cond/high/26_trevor_seg_0004/` | 给 CWM 吃的 condition_root |

`preview.mp4` 应当是三行：源画面、暖色为近的深度、6 类色块。颜色对不上先看第 8 节，不要先改代码。

等价的单词写法：`make build test fetch qc extract preview`。

容器默认以 root 写文件。`first_run.sh` 已经设了你的 uid；手打 compose 时先：

```bash
export DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)
```

---

## 3. 在服务器节点上部署

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
make test           # 期望 229 passed
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

### 步骤 0：先确认环境

```bash
nvidia-smi                     # 驱动 >= 525，否则 torch 2.13 起不来
docker --version
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

### 步骤 1：构建镜像

```bash
docker compose -f docker/docker-compose.yml build
```

镜像内部细节、为什么不用 `nvidia/cuda` 基础镜像、怎么换 CUDA 版本，
见 [`docker/README.md`](docker/README.md)。

**先跑一次不需要 GPU、不需要权重的自检**，确认镜像本身没问题：

```bash
docker compose -f docker/docker-compose.yml run --rm test
# 期望：229 passed
```

再确认卡能被看到：

```bash
docker run --rm --gpus all proxy-extract:0.1.0 \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 期望：True 8
```

打印 `False` 的话是宿主机的问题不是镜像的问题，见第 8 节。

### 步骤 2：拉权重

```bash
docker compose -f docker/docker-compose.yml run --rm fetch
```

国内网络慢的话先 `export HF_ENDPOINT=https://hf-mirror.com`，compose 会透传。

默认拉的是推荐配置需要的两个模型：Mask2Former（语义）和 Depth Anything V2
Metric Outdoor（深度）。别的组合见 `docker/README.md` 的表。

权重放在宿主机的 `.hf-cache/`（可用 `HF_CACHE_DIR` 改），重建镜像不会丢。拉完之后所有抽取都跑在
`HF_HUB_OFFLINE=1` 下 —— 这不是洁癖：8 个 worker 同时打 hub 会互相限速，
批处理跑到一半 hub 抽风会留下一个处理了一半的数据集。

### 步骤 3：挑片（camera-qc）

**这是最该先跑的一步**，而且它不需要 GPU 也不需要权重。

它做的事：在视频里跟踪稀疏特征点，用 GT 相机位姿算对极几何，
看特征点偏离对极线多少像素（Sampson 距离）。位姿是可信的，
所以残差大就说明**这一版渲染的几何跟位姿对不上**。

```bash
docker compose -f docker/docker-compose.yml run --rm qc
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
docker compose -f docker/docker-compose.yml run --rm extract \
  extract --video /work/data/high/26_trevor_seg_0004.mp4 \
          --out /work/out/cond \
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
docker compose -f docker/docker-compose.yml run --rm extract \
  preview --condition-root /work/out/cond/high/26_trevor_seg_0004 \
          --out /work/out/preview.mp4
```

preview 会自己从 `extraction_report.json` 里读出用的是 6 类还是 12 类，
用对应的调色板。**不要**手动把 6 类结果按 12 类的颜色看 —— ID 5 在
6 类里是 `hero`，在 12 类里是 `vegetation`，看起来完全合理但是错的。

### 步骤 6：校验

```bash
docker compose -f docker/docker-compose.yml run --rm extract \
  validate --condition-root /work/out/cond/high/26_trevor_seg_0004 --expect-frames 124
```

这一步用的是跟 CWM 加载器同一套检查（字节数、分辨率、深度范围、ID 范围）。
过了就是 CWM 能读。

### 步骤 7：多卡批处理

```bash
./docker/run_shards.sh /work/data/high /work/out/cond 8
```

每张卡一个容器，`--shard i/8` 把 clip 列表按位置切开互不重叠。日志在
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

## 5. 不用 Docker 的本地跑法

只想看合约/分类/QC 这些不需要模型的部分，纯 CPU 就够：

```bash
python3 -m venv .venv
.venv/bin/pip install -e proxy-extract
.venv/bin/pip install pytest && .venv/bin/python -m pytest proxy-extract/tests -q
```

想跑真模型再装 `pip install torch transformers accelerate`。
`synthetic` 后端可以在完全没有权重的情况下跑通整条链路，用来验证接线：

```bash
.venv/bin/python -m proxy_extract extract \
  --video handpick29_high_low/high/26_trevor_seg_0004.mp4 \
  --out /tmp/smoke --semantic-backend synthetic --depth-backend synthetic
```

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

--depth-backend     depth_anything  单帧米制深度（当前验证过的）
                    mapanything     多帧+可吃 GT 相机（更好，但权重受限，见下）
                    synthetic       假数据
```

两个必须知道的坑：

**depth_anything 是逐帧预测的**，帧与帧之间的尺度没有绑定。静态场景问题不大，
但如果下游对时序深度一致性敏感，得换 mapanything 或者用 GT 相机做尺度标定。
这个 caveat 会写进每份 report 的 `depth.meta.caveat`。

**mapanything 的默认权重是 CC-BY-NC 且在 hub 上是 gated 的。**
商用要用 `facebook/map-anything-apache`。

---

## 8. 出错了怎么办

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `torch.cuda.is_available()` 是 `False` | 宿主机没装 nvidia-container-toolkit，或装完没重启 docker | 装好后 `systemctl restart docker` |
| CUDA 起不来，驱动报版本低 | torch 2.13 要驱动 >= 525 | 升驱动，或按 `docker/README.md` 换 torch+cuXXX |
| 下权重卡住 / 超时 | hub 网络 | `export HF_ENDPOINT=https://hf-mirror.com` 后重跑 fetch |
| `OSError: ... preprocessor_config.json` | 权重没拉全就跑了离线模式 | 重跑 fetch，确认 `--set` 覆盖了你用的后端 |
| `no such video: ...` | 路径是宿主机路径不是容器路径 | 容器里数据在 `/work/data`，不是 `handpick29_high_low` |
| `no camera tracks under ...` | `camera/` 目录不在或没有 json | 检查数据集目录结构 |
| clip 少于 124 帧 | CWM 窗口就是 124 帧 | 这条 clip 用不了，不是 bug |
| preview 颜色不对 | 手动指定了错的调色板 | 别传，让它自己从 report 读 |
| 某个 shard 挂了 | 看 `out/logs/shard-N.log` | 修完重跑同一条命令，`--resume` 会补 |

---

## 9. 已知限制

写在前面，免得当成 bug 去查：

1. **Docker 镜像没被真正构建过，registry 上也没有。** 写这份文档的机器是
   arm64 且没有 docker daemon。版本 pin 来自一个能跑的本地环境，布局是常规
   布局，但第一次 `docker build` 请当成需要盯着的事，而且必须构建成
   `linux/amd64`（第 3 节）。
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
