# ABot MoGe3 + SAM2 数据交付与 FastVideo 训练格式

本文描述 `/data/binghe/datasets/ABot-sub-2000-clips-moge3` 完成全部后处理后的
**实际磁盘布局**，以及 FastVideo 把它编码成训练 cache 后的第二层布局。

这里有两层数据，不能混：

1. **可追溯源数据**：RGB、逐帧米制深度、逐帧 CWM12 语义和文本。
2. **FastVideo cache**：VAE latent、Qwen3-VL text embedding 和 token tags；训练只读这一层。

最重要的边界是：新数据训练走每片 `duv/` 目录，对应 manifest key
`proxy_duv`。`proxy/duv.mp4` 是旧 DATA_F / Standard11 兼容产物，不能代替它。

---

## 1. 数据集根目录

推荐固定为：

```text
/data/binghe/datasets/ABot-sub-2000-clips-moge3/
├── semantic.json
├── encode_manifest.jsonl
├── captions_audit.json
├── proxy_duv_audit.json
├── clip_000000_0/
├── clip_000000_1/
├── ...
├── logs/
├── clips_manifest.<shard>-of-<count>.json
└── review/
```

根目录文件的作用：

- `semantic.json`
  - 全语料唯一的 CWM12 `semantic_id → [U,V]` 映射。
  - 记录 DUV 网格 `336×192`、U/V 字节和通道位置。
  - FastVideo 编码器不读取它；编码器使用自己代码里的同一组权威常量。它用于交付说明和漂移检查。
- `encode_manifest.jsonl`
  - 一行一个 clip，路径相对于数据集根目录。
  - 是源数据进入 `encode_proxy_samples.py` 的索引。
  - 原始版本可能包含没有 prompt 的行；训练前必须生成只保留有效 prompt、且按 episode 切分的
    `_fastvideo/train.jsonl` 和 `_fastvideo/val.jsonl`，见第 6 节。
- `captions_audit.json`
  - 文本处理统计：`captioned / missing / failed / warned / exported`。
- `proxy_duv_audit.json`
  - 深度、语义、范围和跨片深度中位数检查。
- `logs/` 和 `clips_manifest.*.json`
  - 生产过程记录，不参与 FastVideo 编码或训练。
- `review/`
  - 人工检查用的 RGB | depth | semantic 并排 MP4 和 `selection.json`，不参与训练。

根目录不应该出现正在使用的 `.work/`。完成的 clip 也不应该残留 `.work/`。

---

## 2. 单个 clip 的最终结构

```text
clip_000414_2/
├── target/
│   ├── rgb.mp4
│   └── anchor.png
├── proxy/
│   └── duv.mp4
├── duv/
│   ├── 000000.depth.f32
│   ├── 000000.semantic_id.png
│   ├── 000001.depth.f32
│   ├── 000001.semantic_id.png
│   ├── ...
│   ├── 000123.depth.f32
│   └── 000123.semantic_id.png
├── annotations/
│   ├── action.json
│   ├── caption.json
│   ├── cameras.npz
│   ├── prompt.json
│   └── prompt_sheet.jpg
├── prompt.txt
└── clip_report.json
```

`annotations/` 以及其中每一个文件都可能缺席。训练必需的不是整个目录，而是通过质量检查后导出的
`prompt.txt`。

### 2.1 `target/rgb.mp4`

- H.264 `yuv420p`
- `1344×768`
- `124` 帧
- `24 fps`
- 时长约 `5.17 s`
- FastVideo 编码成 `vae_latent`

### 2.2 `target/anchor.png`

- `1344×768`
- target 第 0 帧的无损版本

当前 `proxy-duv-manifest` 有意不写 `anchor` key，让 FastVideo 从 target 第 0 帧回退得到
appearance anchor。不要在不同 cache 中一部分显式使用 PNG、一部分使用 MP4 解码帧。

### 2.3 `duv/*.depth.f32`

- 每片 `124` 个
- shape：`[192, 336]`
- little-endian float32，C order，无 header
- 每帧固定 `192 × 336 × 4 = 258048` 字节
- 数值：米制、相机空间正 view-z
- `0` 表示无效或天空
- 全语料使用同一个米制标度，禁止逐帧或逐片 min/max 归一化

FastVideo 在编码时把它映射为反向 log depth：

```text
near = 0.3 m
far  = 256 m
invalid = 0
```

### 2.4 `duv/*.semantic_id.png`

- 每片 `124` 个
- `336×192`
- PNG mode `L`
- 每像素就是 CWM12 类别 id，范围 `[0, 12)`
- 与 depth、target 逐帧对齐

对应的 U/V 网格：

```text
U = [32, 96, 160, 224]
V = [43, 128, 213]

u = U[id % 4]
v = V[id // 4]
```

FastVideo 打包 DUV 时：

```text
RGB = [depth_code, semantic_u, semantic_v]
```

即 U 在 G 通道，V 在 B 通道，U 变化最快。

### 2.5 `proxy/duv.mp4`

这是为了旧 DATA_F 路线和人工预览保留的兼容产物：

- `336×192`
- 124 帧、24 fps
- Standard11 / 旧 depth 编码

**新训练不得使用它。** 以下 manifest 是错的：

```json
{"proxy_duv_video": "clip_000414_2/proxy/duv.mp4"}
```

新训练必须使用：

```json
{"proxy_duv": "clip_000414_2/duv"}
```

### 2.6 文本文件

文本链路：

```text
annotations/prompt.json
        │ captions-recompile --reverify
        │ captions-export --write-txt
        ▼
prompt.txt
        │ proxy-duv-manifest
        ▼
manifest 行的 prompt 字段
        │ encode_proxy_samples.py
        ▼
.pt 的 text_embedding + text_token_tags
```

- `annotations/prompt.json`
  - VLM 生成的结构化逐秒描述。
  - 包含 evidence、事件、检查结果、编译文本和 provenance。
  - 不直接进入训练。
- `prompt.txt`
  - 真正进入 Qwen3-VL 的 CWM 用户句。
  - 默认是一个带窗口时间戳的单行文本：

```text
[0.00s-5.17s] Third-person open-world video game. ...
```

  - UTF-8、无 BOM、无末尾换行；多行变体使用 CRLF。
  - `checks.fail` 非空的 caption 默认不导出这个文件。
- `annotations/caption.json`
  - 原始 episode 级描述，描述约 60 秒。
  - 不能回退成 5.17 秒训练窗口的 prompt。

### 2.7 `clip_report.json`

记录：

- episode / window 编号
- `source_ordinals`
- fps、帧数和几何
- depth / semantic 后端
- `moge3`、`sam2` 和 `proxy_duv` provenance
- 源视频与源 annotations 路径

它用于审计、resume 和 prompt 复用校验，不被 FastVideo 训练 loader 读取。

---

## 3. 文本质量门禁

`clip-prompts` 用 DUV evidence 对 caption 做两档检查：

- `fail`
  - DUV 明确否定文本，例如文字说有车，但对应秒没有 vehicle 像素。
  - 默认不写 `prompt.txt`。
- `warn`
  - 有合理解释的弱冲突，例如跟踪误差或相机运动导致深度场变化。
  - 保留样本，但记录在 audit。

`score` 会记录，但当前没有按 score 自动设淘汰阈值。

本轮在新 MoGe3 + SAM2 DUV 上重新验证并导出后的实测结果：

```text
clips       9985
captioned   9721
missing      264
failed        61
warned       893
exported    9660
mean_score  0.944
```

因此原始 `encode_manifest.jsonl` 有 `9985` 行，其中 `325 = 264 + 61` 行没有可训练的
`prompt`；文本门禁后最多有 `9660` 个候选 clip。`warned` 仍保留。最终训练数量还要扣除
episode 级 validation holdout；同时必须先完成 DUV audit，不能在这一步写死。

检查命令：

```bash
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips-moge3

python -m clip_prompts captions-audit \
  --clips "$CLIPS_DIR" \
  --report "$CLIPS_DIR/captions_audit.json"
```

验收关系：

```text
missing = 0                         # 理想值；否则这些片没有结构化 caption
exported = captioned - failed       # 通过质量门禁的训练文本
warned 可以非 0
```

注意：`make proxy-duv-manifest` 扫描的是完整媒体树。没有 `prompt.txt` 的 clip 仍可能以无
`prompt` 字段的形式出现在原始 manifest 里，而 FastVideo encoder 明确要求 `prompt`。因此训练使用的
manifest 必须过滤 prompt，而不是直接假设根目录的 manifest 已经完成文本筛选。

---

## 4. DUV 质量门禁

```bash
make clips-audit CLIPS_DIR="$CLIPS_DIR"
make proxy-duv-audit CLIPS_DIR="$CLIPS_DIR" AUDIT_WORKERS=32
python scripts/write_semantic_uv.py "$CLIPS_DIR" --check
```

`proxy-duv-audit` 的验证与统计共用一次帧读取，并按 clip 多进程并行。224 核节点先用
`AUDIT_WORKERS=32`；该任务受 `/data` 小文件吞吐限制，不应直接开到 224。

接受条件：

- 没有不完整 clip
- `.work/` 数量为 0
- `proxy-duv-audit` 退出码为 0
- `warnings` 为空
- 跨片深度 median spread 没有超过门槛
- 根目录有且只有一份 `semantic.json`

`proxy_duv_audit.json` 的价值不只是结构检查。逐片归一化的深度可以让每一帧都合法，只有跨片比较
深度中位数才能发现整批标度不可比较。

当前文本门禁已经完成；最终并行 DUV audit 的结果尚未记录到本文。在
`proxy_duv_audit.json` 确认 `failed = 0` 且 fatal warning 为空之前，这批数据仍属于“编码前待验收”，
不能仅凭 `9985` 个目录就视为全部可训练。

---

## 5. 原始 encode manifest

生成：

```bash
make proxy-duv-manifest CLIPS_DIR="$CLIPS_DIR"
```

每行形式：

```json
{
  "name": "clip_000414_2",
  "id": "clip_000414_2",
  "target": "clip_000414_2/target/rgb.mp4",
  "proxy_duv": "clip_000414_2/duv",
  "prompt": "[0.00s-5.17s] Third-person open-world video game. ..."
}
```

约束：

- 所有路径相对于 `CLIPS_DIR`
- `proxy`、`proxy_duv`、`proxy_duv_video` 三者恰好一个
- 本数据必须是 `proxy_duv`
- 有效训练行必须包含非空 `prompt`
- `anchor` 省略时 FastVideo 使用 target 第 0 帧
- ABot 单窗口使用 `--cwm-system w0`

---

## 6. 生成 FastVideo 专用 train / val manifest

切分必须按 episode，而不是按 clip。同一 episode 的五片共享环境、天气和外观；按 clip 随机切分会
让验证集包含训练中已经见过的同一段世界。

下面从原始 manifest：

1. 丢掉没有 prompt 的行；
2. 按 episode 留出 24 个验证 episode；
3. 写入数据根目录 `_fastvideo/`。

```bash
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips-moge3

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CLIPS_DIR"])
rows = [
    json.loads(line)
    for line in (root / "encode_manifest.jsonl").read_text().splitlines()
    if line.strip()
]

with_prompt = [
    row for row in rows
    if isinstance(row.get("prompt"), str) and row["prompt"].strip()
]
dropped = len(rows) - len(with_prompt)
if not with_prompt:
    raise SystemExit("no manifest row has a prompt; finish captions-export before splitting")

def episode(row):
    # clip_000414_2 -> 000414
    return str(row["name"]).split("_")[-2]

episodes = sorted({episode(row) for row in with_prompt})
val_count = min(24, len(episodes))
val_episodes = {
    episodes[int(index * len(episodes) / val_count)]
    for index in range(val_count)
}

train = [row for row in with_prompt if episode(row) not in val_episodes]
val = [row for row in with_prompt if episode(row) in val_episodes]

out = root / "_fastvideo"
out.mkdir(exist_ok=True)
for name, split in (("train.jsonl", train), ("val.jsonl", val)):
    (out / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split),
        encoding="utf-8",
    )

assert {episode(row) for row in train}.isdisjoint(
    {episode(row) for row in val}
)
print({
    "source": len(rows),
    "without_prompt_dropped": dropped,
    "train": len(train),
    "val": len(val),
    "val_episodes": len(val_episodes),
})
PY
```

最终新增：

```text
ABot-sub-2000-clips-moge3/
└── _fastvideo/
    ├── train.jsonl
    └── val.jsonl
```

训练前检查：

```bash
wc -l "$CLIPS_DIR"/_fastvideo/{train,val}.jsonl

python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["CLIPS_DIR"])
for name in ("train.jsonl", "val.jsonl"):
    rows = [json.loads(x) for x in (root / "_fastvideo" / name).read_text().splitlines()]
    assert rows
    assert all(row.get("prompt") for row in rows)
    assert all("proxy_duv" in row for row in rows)
    assert all("proxy_duv_video" not in row for row in rows)
    print(name, len(rows))
PY
```

---

## 7. FastVideo 编码 cache

FastVideo 不在训练时现场跑 VAE 或 Qwen3-VL。先把每片编码成一个 `.pt`：

```bash
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips-moge3
export CACHE_DIR=/data/binghe/h3_proxy/cache/abot_moge3_sam2_w0_qwen2
export MODEL_PATH=/data/models/MiniMax-H3

cd /workspace/FastVideo

python scripts/h3_proxy/prepare_models/verify_h3_snapshot.py \
  --path "$MODEL_PATH" --profile ref2va

NUM_SHARDS=8 STAGGER_SEC=45 \
scripts/h3_proxy/prepare_data/encode_proxy_shards.sh \
  --manifest "$CLIPS_DIR/_fastvideo/train.jsonl" \
  --root "$CLIPS_DIR" \
  --output "$CACHE_DIR" \
  --model-path "$MODEL_PATH" \
  --num-frames 124 \
  --height 768 \
  --width 1344 \
  --proxy-height 192 \
  --proxy-width 336 \
  --anchor-short-edge 2048 \
  --qwen-video-fps 2 \
  --cwm-system w0
```

这里的 `--qwen-video-fps 2` 只控制 Qwen3-VL 观看 `<Video 1>` 时的时间采样率。
源视频、target VAE 和 proxy VAE 仍使用完整的 124 帧（24 fps），不会降采样为 2 fps。

这批 cache 的固定时间合同是：

```text
源 target/proxy 时间线     24 fps，124 帧，约 5.17 秒
target/proxy VAE 输入      全部 124 帧
Qwen <Video 1> 采样率      2 fps
info.qwen_video_fps        2.0
CWM system role            w0
given latent frames        1
```

目录名显式带 `_qwen2`，避免现有 24-FPS Qwen cache 被 resume 逻辑跳过后混入。普通编码遇到
已存在的 `.pt` 会跳过，所以不要把 2-FPS 命令指向一个曾用 24 FPS 编码过的目录。

`encode_proxy_shards.sh`：

- 默认每张可见 GPU 一个进程
- shard `i` 处理 manifest 的 `i::N`
- 已存在的 `.pt` 自动跳过
- 中断后重跑同一命令即可 resume
- 日志默认写到 cache 同级 `encode_logs/`

编码后的布局：

```text
/data/binghe/h3_proxy/
├── cache/
│   ├── abot_moge3_sam2_w0_qwen2/
│   │   ├── clip_000000_0.pt
│   │   ├── clip_000000_1.pt
│   │   └── ...
│   └── encode_logs/
│       ├── shard_0.log
│       ├── shard_1.log
│       └── ...
```

每个 `.pt` 是一个字典，主要字段：

```text
vae_latent          target/rgb.mp4 的 VAE latent
proxy_latent        逐帧 depth + semantic 打包后 DUV 的 VAE latent
anchor_latent       appearance anchor 的 keyframe latent
text_embedding      Qwen3-VL 文本/视觉联合 embedding
text_token_tags     每个 token 的 modality tag
info.num_frames
info.pixel_size
info.qwen_video_fps
info.prompt
info.cwm_system
```

本数据没有可直接使用的米制相机轨迹，因此不应该出现：

```text
camera_extrinsics
camera_intrinsics
```

编码完成后：

```bash
python scripts/h3_proxy/describe_cache.py \
  "$CACHE_DIR" \
  --manifest "$CLIPS_DIR/_fastvideo/train.jsonl"
```

必须确认：

- cache `.pt` 数量等于 train manifest 行数
- target 为 `768×1344`
- proxy 为 `192×336`
- `num_frames = 124`
- `qwen_video_fps = 2`
- `cwm_system = w0`
- 整个 cache 的几何只有一种

数据量级约为每片十几 MiB；约一万片通常是约百余 GiB，必须放持久化 `/data`，不要放
`/workspace`。

---

## 8. FastVideo 训练读取的最终形态

训练 YAML 的 `training.data.data_path` 指向 cache，而不是原始 clip：

```yaml
training:
  data:
    data_path: /data/binghe/h3_proxy/cache/abot_moge3_sam2_w0_qwen2
    preprocessed_data_type: t2va
    num_frames: 124
    num_latent_t: 37
    num_height: 768
    num_width: 1344
```

模型设置必须与 cache 的文本合同一致：

```yaml
models:
  student:
    enable_anchor: true
    num_given_latent_frames: 1
    enable_camera_controlnet: false
```

说明：

- `w0` 对应 `num_given_latent_frames: 1`
- `wn` 对应 `num_given_latent_frames: 10`
- 本批单窗口 caption 按 `w0` 编码，不要训练时改成 `wn`
- `align_proxy_reference_time` 是模型实验变量，不改变磁盘 cache；训练与验证必须一致

可以从
`examples/train/scenario/h3_proxy/proxy_bd_finetune_abot.yaml` 复制一份新配置，但该文件的注释和
旧 cache 路径描述的是 `proxy_duv_video`。源数据编码时必须使用本文的 `proxy_duv` manifest，
并把 `data_path` 换成新 cache。

训练只读取 `.pt`，不会再读取：

- `duv/*.depth.f32`
- `duv/*.semantic_id.png`
- `prompt.txt`
- `prompt.json`
- `semantic.json`
- `clip_report.json`

这些源文件仍必须保留，用于重编码、修文本、审计和定位坏样本。

---

## 9. 当前验证路径的已知限制

FastVideo 的 `encode_proxy_samples.py` 已经支持逐帧 `proxy_duv`，因此训练 cache 可以正确生成。

但是当前：

- `clip_dir_to_validation_json.py` 写的是 `proxy/duv.mp4`
- validation callback 的 `proxy_path` 只接受视频
- 它不能直接读取 `clip_*/duv/` 的逐帧 depth + semantic

因此，**不要用现有 `clip_dir_to_validation_json.py` 对这批新 cache 做效果结论**。那会训练时使用
CWM12 逐帧 DUV，验证时却使用旧 Standard11 `proxy/duv.mp4`，两边输入编码不同，结果没有可比性。

在可视化验证前必须完成二选一：

1. 扩展 validation callback，让 `proxy_path` 支持逐帧 `duv/`；或
2. 只为 held-out validation clips 生成由 FastVideo 自己的 CWM packer 打包、bit-exact
   `libx264rgb` 编码的 DUV 视频。

在该项完成前，可以验证训练 loss 和 cache 完整性，但不能把现有 callback 产出的视觉效果当作这批新
DUV 的真实验证效果。

---

## 10. 训练前最终检查单

依次执行：

```bash
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips-moge3
export CACHE_DIR=/data/binghe/h3_proxy/cache/abot_moge3_sam2_w0_qwen2

cd /workspace/fastvideo_datapipe

python -m clip_prompts captions-audit \
  --clips "$CLIPS_DIR" \
  --report "$CLIPS_DIR/captions_audit.json"

make clips-audit CLIPS_DIR="$CLIPS_DIR"
make proxy-duv-audit CLIPS_DIR="$CLIPS_DIR" AUDIT_WORKERS=32
python scripts/write_semantic_uv.py "$CLIPS_DIR" --check

cd /workspace/FastVideo
python scripts/h3_proxy/describe_cache.py \
  "$CACHE_DIR" \
  --manifest "$CLIPS_DIR/_fastvideo/train.jsonl"
```

最终必须满足：

- source clip 无 `.work/`
- depth / semantic 各 124 帧
- DUV audit 无 fatal warning
- train manifest 每行有 `prompt`
- train manifest 每行只有 `proxy_duv`，没有 `proxy_duv_video`
- train / val episode 集合不相交
- cache 数量等于 train manifest 行数
- cache 几何、fps 和 `cwm_system` 单一且正确
- 训练配置的 `data_path` 指向新 cache
- 验证没有偷用旧 `proxy/duv.mp4`

