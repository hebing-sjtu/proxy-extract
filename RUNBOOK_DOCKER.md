# Runbook：在 FastVideo 官方 Docker 里跑（双节点）

这份是 [`RUNBOOK.md`](RUNBOOK.md) 的一个变体，只覆盖一种部署：**镜像已经给好了
Python 和 torch，驱动不可改，所以这套管线必须搬进去住，而不是自己建环境。**

先给结论，后面都是它的展开：

> **不要跑 `scripts/setup_venv.sh`，不要 `pip install -r requirements.txt`。**
> 这两条都会把 torch 换掉，而在这台机器上换 torch 会让 16 张卡**静默消失**。
> 用 `scripts/setup_docker_env.sh`。

---

## 0. 这台机器是什么

节点 0 上实测（`node-0-0`，节点 1 同镜像）：

| | 值 |
| --- | --- |
| Python | 3.12.14 |
| torch | **2.12.0+cu126** |
| torch 编译时的 CUDA | **12.6** |
| `torch.cuda.is_available()` | True |
| venv | `$VIRTUAL_ENV`（进 shell 时已激活） |
| FastVideo | `/workspace/FastVideo` |
| 卡 | 2 节点 × 8 = 16 |

两个数决定了后面所有选择：**torch 2.12.0** 和 **cu126**。

## 1. 为什么不能动 torch

不是洁癖，是这台机器上有三个具体后果。

**一、`requirements.txt` 钉的是 `torch==2.13.0`，而 PyPI 给这个版本的默认 wheel 是
cu130 构建的。** CUDA 只在小版本间兼容，cu130 需要 580+ 驱动；这台是 12.6。装上去的
结果不是报错，是 `torch.cuda.is_available()` 变成 `False`，16 张卡全部从视野里消失，
所有东西退回 CPU，**而提示只有一条 UserWarning**。`requirements.txt` 里那段长注释讲的
就是这件事。驱动不可改，所以这一步一旦做了就只能靠重装 torch 往回退。

**二、镜像的 flash-attn 轮子是按 `cu126torch2.12` 编出来的**（见
`FastVideo/docker/Dockerfile` 里 `FLASH_ATTN_WHEEL_TAG` 必须跟 `UV_TORCH_BACKEND`
对齐那段）。换掉 torch 就废掉镜像自己的训练路径 —— 而这台机器装这个镜像正是为了训练。

**三、换来的收益是零。** 所有后端要的只是 `torch>=2.4`，2.12.0 完全够。也就是说，把
torch 换成我们钉的版本，代价是丢掉 GPU 和 flash-attn，收益是没有。

同理不要装 `opencv-python-headless`。镜像已经有 `opencv-python`，两者是**两个发行包提
供同一个 `cv2` 模块**，装进同一个环境会得到两份 `cv2`，出问题时极难定位。镜像那份功能
上是超集（只多了我们用不到的 GUI）。

## 2. 装（两个节点各做一遍）

```bash
# 第一次：
cd /workspace
git clone https://github.com/hebing-sjtu/proxy-extract.git fastvideo_datapipe
cd fastvideo_datapipe

# 已经克隆过、要拿最新修复再装一遍：
cd /workspace/fastvideo_datapipe && git pull

scripts/setup_docker_env.sh --with-flicker
```

装之前**先 `git pull`**。这个脚本重复执行是安全的（装过的不会再装），所以修好一个节点
上的问题以后，两个节点都重跑一遍是最省事的做法。

`--with-flicker` 会连 `moge3`（深度）和 `sam2`（语义）一起装；这两个是治闪烁的那对，
都只有 git 源，不在 PyPI 上。只想先把管线本身跑通就不加这个参数。

先看它打算做什么再让它动手：

```bash
DRY_RUN=1 scripts/setup_docker_env.sh --with-flicker
```

这个脚本做的事，和它**拒绝**做的事一样重要：

- 从当前环境把 torch / torchvision / torchaudio / transformers / tokenizers /
  accelerate / diffusers / numpy / flash-attn / flashinfer 的**实际版本**抄成一个
  constraints 文件，后面所有安装都带着 `-c` 走。这不是偏好，是围栏：某个依赖真要动
  torch，pip 会在这里**报错**，而不是悄悄升上去、几天后表现成「这个节点训不动了」。
- `pip install --no-deps -e proxy-extract`。`--no-deps` 是因为四个运行时依赖
  （numpy / pillow / cv2 / imageio-ffmpeg）镜像全都有，已经逐个 import 验过了；不加
  它 pip 会为了照字面满足 pyproject 去抓 `opencv-python-headless`，正是上面说的第二份
  `cv2`。
- 装完**再读一次 torch 版本，和装之前比**。只要动了就报错并给出回退命令。这条检查是
  这个脚本存在的理由，不是装饰。
- 最后跑一遍测试套件当门禁。这个环境的 numpy / opencv / pillow 跟管线合不合，这是最快
  的答案，而且不需要 GPU。真被无关问题卡住可以 `SKIP_TESTS=1` 绕过，但绕过之后在信任
  任何输出之前要自己补跑。

## 3. 体检：哪些 warning 是正常的

```bash
python scripts/doctor.py
```

**预期会有 pin 冲突的 warning，而且不用管。** 镜像钉 `accelerate==1.0.1`，我们
`requirements.txt` 记的是 `1.14.0`；`doctor.py` 的 `check_conflicting_pins` 会把这类
差异都列出来，因为在**自建环境**里它确实是个坑。在这里不是：这些版本是镜像挑的，而
镜像的选择优先。

必须是绿的只有这几条：

| 项 | 要求 |
| --- | --- |
| `proxy_extract` 能否导入 | ok |
| ffmpeg | ok |
| `depth encode` | **ok（bit-exact）**，见下 |
| torch / cuda available | ok，且 **2.12.0+cu126** |
| `moge3 (depth flicker)` | ok（装了 `--with-flicker` 的话） |
| `sam2 (semantic flicker)` | ok（同上） |
| `DATA_DIR` 有几条 episode | 非 0 |

`moge3` 那行如果说 macOS 不支持，说明 `DATA_DIR`/脚本被在本地跑了 —— 在这个镜像里不会
出现，MoGe-3 的 FlexGEMM 依赖 Triton，Triton 没有 macOS wheel，但 Linux 上没这个问题。

### `depth encode` 这行是干什么的

这个镜像的 `/usr/bin/ffmpeg` 是 Ubuntu 22.04 的 **4.4.2**，它写深度时有个会安静毁数据的
毛病。深度码值是「装成像素的数字」，x264 不吃 `gray`，所以码值被放进亮度平面；这只有在
文件**标明是全范围**时才可逆。ffmpeg 7/8 会写成 `yuvj420p` 并打上 `pc` 标记，4.4.2 写的
是 `yuv420p` 且 `color_range=unknown` —— 存进去的码值其实没动，但没人知道这件事，于是任何
解码器都按有限范围把 16..235 拉回 0..255，两端截断、码值合并。**产出看起来完全正常，而每
一个深度值都是错的。**

所以现在深度那条流不再依赖默认行为：格式明确写成 `yuvj420p` 并显式带 `-color_range pc`。
更要紧的是加了一道运行时闸门 —— 真正编码前会用一张含全部 256 个码值的测试帧过一遍这台机
器上的 ffmpeg，读回来不是逐位相同就不用它。`/usr/bin/ffmpeg` 过不了的话会自动改用
`imageio-ffmpeg` 自带的 7.1（镜像里就有，已验证没问题），并在 stderr 说明换了；如果一个都
过不了，它会**拒绝编码**而不是交付被缩放过的深度。

`$FFMPEG` 是例外：你显式指名的二进制不会被悄悄替换掉，过不了闸门就直接报错。

## 4. 拉权重（两个节点各做一遍）

```bash
export HF_HOME=/data/binghe/cache/huggingface     # 指到大盘上
python scripts/fetch_models.py --set flicker      # MoGe-3 + SAM 2.1
```

`flicker` 这组就是 `Ruicheng/moge-3-vitl` 加 `facebook/sam2.1-hiera-large`，都不需要
hub 登录。国内网络慢先 `export HF_ENDPOINT=https://hf-mirror.com`。

拉全之后正式跑加 `HF_HUB_OFFLINE=1`：不只是整洁，128 个 worker 各自去撞 hub 会把自己
限流，跑到一半赶上 hub 抖动会留下一个处理了一半的数据集。

## 5. 先试跑：单节点，8 条 episode

**不要跳过这一步。** 唯一目的是定 `WORKERS_PER_GPU`，而这个数我给不了（见下）。

```bash
export DATA_DIR=/data/binghe/datasets/ABot-World-Explorer-subset2000/data
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips

make clip-episodes LIMIT=8 WORKERS_PER_GPU=2 \
    DEPTH=moge3 REFINER=sam2 PROXY_DUV=1
make proxy-duv-audit
```

**`WORKERS_PER_GPU` 的默认值 6~8 是拿 DA3 + Mask2Former 量的，对这套配置偏大。**
SAM 2 的 video predictor 会为每个 masklet 维护一份跨整段的记忆（上限 48 个 × 124 帧），
每个 worker 的显存占用比原来高不少，8 个大概会 OOM。所以从 2 起步，跑起来之后看：

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5
```

利用率没到顶而显存还有富余就往上加。**加过头的后果是有界的**：一个 worker 一个进程,
CUDA OOM 只杀一个 shard，重跑会从已经写好的片接着走。

### 但显存往往不是上限

看到 `60/140 GiB` 就加 worker 是最容易踩的一脚，因为先撑不住的通常是另外两样，而它们
都不会给出干净的报错：

**宿主内存。** 这条路线上一次调用就是一个窗口（不能切，见第 7 节），那一批彩色帧在内存里
同时存在三份 —— 一份在算、两份被 prefetch 预读 —— 再加上由它导出的 float32 深度栈。128
帧 1344×768 大约是 1.2 + 0.5 + 0.13 GiB，所以按**每 worker 2.5 GiB**估：

| `WORKERS_PER_GPU` | 8 卡的 worker 数 | 约需宿主内存 |
| --- | --- | --- |
| 6 | 48 | 117 GiB |
| 8 | 64 | 156 GiB |
| 10 | 80 | 195 GiB |
| 12 | 96 | 234 GiB |

启动器现在会读 `/proc/meminfo` 的 `MemAvailable` 做预检，不够就拒绝启动并算给你上限;
banner 里的 `memory` 一行是它的估算。量准了可以用 `MIB_PER_WORKER_RAM=` 覆盖。内存超卖
不会干净地失败：它开始换页，所有 worker 一起变慢，显卡反而空着。

**CPU 核数。** 每个 worker 都要在 CPU 上解 H.264 来喂自己的卡，`THREADS_PER_WORKER` 是
`nproc / worker 数`（上限 4）。低于每 worker 一个核之后，worker 就是在排队等核，加得再多
也不会更快 —— 而现象恰恰是显卡看起来更忙、总时间没变短，很容易被读成"还有余量"。低于
这条线时启动器会提示。

先用 `nproc` 和 `free -g` 把这两个天花板算出来，再决定加到多少。

试跑完要看的三件事：

1. `make proxy-duv-audit` 退出码 0，没有 warning。
2. 随便挑一片的 `clip_report.json`，`proxy_duv.taxonomy` 是 `cwm12`。
3. 同一份 report 里 `depth.meta` 的 `fov_source` 是 `probed`、`scale_locked` 是
   `true`、`frames_in_call` 约等于 124 + 2×`temporal_radius`。第三个数如果明显更小，
   说明有人传了 `--chunk-frames` 把一段切开了，见第 7 节。

## 6. 全量：双节点

每个节点一条命令，**只有 `NODE_RANK` 不同**：

```bash
# node 0
NODE_COUNT=2 NODE_RANK=0 make clip-episodes \
    DEPTH=moge3 REFINER=sam2 PROXY_DUV=1 WORKERS_PER_GPU=<第 5 节定的>

# node 1
NODE_COUNT=2 NODE_RANK=1 make clip-episodes \
    DEPTH=moge3 REFINER=sam2 PROXY_DUV=1 WORKERS_PER_GPU=<同上>
```

分片是全局的：`--shard i/N` 按位置切 episode 列表，每个 worker 自己从 `DATA_DIR`
重新列一遍（排过序），所以两台机器之间不需要任何通信。本节点拿
`[NODE_RANK × n_workers, +n_workers)` 这一段，日志按全局编号落在
`$CLIPS_DIR/logs/shard-<全局号>.log`，两台写进同一个 `CLIPS_DIR` 不会撞名字。

**三件事必须一致，否则会静默漏掉一部分语料 —— 没有任何报错，只有最后审计数目偏少：**

1. **两边看到的 episode 列表必须完全相同。** 共享挂载最省事。
2. **`NODE_COUNT` 两边相同，`NODE_RANK` 互不相同。**
3. **每个节点的 worker 数必须相同。** 全局分片数是按本节点的 worker 数乘 `NODE_COUNT`
   算的，一台 8 卡配一台 4 卡会让后半个语料没人认领。脚本校验 `NODE_RANK` 的范围，但
   它看不见另一台机器，这一条只能靠约定。

`CLIPS_DIR` 建议放共享盘：`--resume` 和审计都是看盘上已有的成品，放一起才能两边都正确
跳过已完成的片，最后的数才是全局的。各写本地盘也能跑，事后 rsync 到一处再审计。

**磁盘：`PROXY_DUV=1` 是大头。** 逐帧的 float32 深度不压缩，一片 124 帧约 31 MiB，
加上两个视频一片约 39 MiB —— 10000 片是 **约 390 GiB**，不是不开这个参数时的 80 GiB。
启动器的 preflight 已经按这个算，盘不够会直接拒绝启动。

## 7. 三个别动的默认值

**`--chunk-frames` 不要传。** 这条路线上一次调用就是一个窗口，而这正是 MoGe-3 的相机
锁、尺度锁和 SAM 2 的记忆需要的粒度。切开它会在每个边界重探一次相机、重调一次尺度，
深度上留一道台阶，SAM 2 的 masklet 也会断。report 里的 `frames_in_call` 就是用来事后
确认这件事的。

**`DEPTH=moge3` 和 `REFINER=sam2` 要一起开。** 深度闪烁和语义闪烁是两件独立的事，各
治一半；只开一个就只好一半。

**别用 `SEMANTIC=synthetic` 或 `DEPTH=synthetic` 试通路然后忘记改回来。** 启动器会拦
（要 `ALLOW_SYNTHETIC=1`），因为合成后端的输出结构上完全合法、看起来也正常，但内容是
编的。

## 8. 验收

```bash
make clips-audit                                  # 切完的 / 半截的
make proxy-duv-manifest PROMPTS=prompts.json      # encode_manifest.jsonl
make proxy-duv-audit                              # 第 8 节的验收检查
```

**`proxy-duv-audit` 是这里唯一值得认真看的一条。** 它的逐段检查只是把消费方自己的断言
重跑一遍；真正有价值的是**跨段**的深度中位数散布 —— 逐段归一化会通过所有逐帧检查
（每个文件都合法、形状对、范围对、类别 id 都在 12 以内），只有把段之间放在一起比才看得
出来，而这是唯一一个能让整批数据作废的错误（`PROXY_DUV_SPEC.md` 第 2 节）。有 warning
就退非 0，所以可以直接当 CI 门禁。

`--prompts` 忘了不会报错，但会得到一份没有 prompt 的 manifest —— 它能加载、能训练，
只是不是训练想要的。命令会把没带 prompt 的段数打出来。

## 9. 出错了先看哪

| 症状 | 大概是什么 |
| --- | --- |
| `torch.cuda.is_available()` 变成 False | 有人装了别的 torch。第 1 节。`pip install --reinstall 'torch==2.12.0'` |
| 每个 shard 都是同一个 ImportError | `moge`/`sam2` 没装。启动器有预检会先拦，见第 2 节 |
| 装的时候 `test_delivery.py` 报 `depth codes changed` | 系统 ffmpeg 4.4.2 丢了全范围标记。已修（第 3 节「`depth encode`」）；`git pull` 后重跑。细节用 `python scripts/diagnose_depth_encode.py` |
| stderr 出现 `not using the ffmpeg from PATH` | 正常，闸门在绕开 4.4.2 改用 7.1。不用管 |
| `Refusing to encode` / `depth encode` FAIL | 这台机器没有一个能用的 ffmpeg。`pip install imageio-ffmpeg` 或 `export FFMPEG=` 指一个 7 以上的 |
| 卡上全是 OOM | `WORKERS_PER_GPU` 太大。第 5 节 |
| 全部 GPU 0% 占用、没有任何报错 | 叠了 worker 没限线程。`RUNBOOK.md` 第 3 节「线程」 |
| 审计数目比 2000×5 少 | 双节点三条约定之一没对上。第 6 节 |
| 深度每隔 N 帧一道台阶 | 传了 `--chunk-frames`。第 7 节 |
| `proxy-duv-audit` 报 median spread | 逐段归一化，整批作废。第 8 节 |
| 盘满 | `PROXY_DUV=1` 的体积。第 6 节 |

`make doctor` 是把所有前置条件一次查完、各自给修法、不中途退出的那个；启动器的
preflight 是遇到第一个问题就退（对启动器是对的：用错的环境起跑会白烧几个小时）。

## 10. 速查

```bash
# 一次性（每节点）
cd /workspace && git clone https://github.com/hebing-sjtu/proxy-extract.git fastvideo_datapipe
cd fastvideo_datapipe && scripts/setup_docker_env.sh --with-flicker

# 重装 / 拿最新修复（每节点，可重复执行）
cd /workspace/fastvideo_datapipe && git pull && scripts/setup_docker_env.sh --with-flicker
python scripts/doctor.py
export HF_HOME=/data/binghe/cache/huggingface
python scripts/fetch_models.py --set flicker

# 每次跑
export DATA_DIR=/data/binghe/datasets/ABot-World-Explorer-subset2000/data
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips
export HF_HUB_OFFLINE=1

make clip-episodes LIMIT=8 WORKERS_PER_GPU=2 DEPTH=moge3 REFINER=sam2 PROXY_DUV=1   # 试
NODE_COUNT=2 NODE_RANK=$R make clip-episodes DEPTH=moge3 REFINER=sam2 PROXY_DUV=1   # 全量

# 收货
make clips-audit && make proxy-duv-manifest PROMPTS=prompts.json && make proxy-duv-audit
```
