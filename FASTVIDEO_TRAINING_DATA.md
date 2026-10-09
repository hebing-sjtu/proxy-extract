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
`prompt`；基础文本门禁后最多有 `9660` 个候选 clip。本轮采用更严格的训练门禁：`missing`、
`failed`、`warned` 全部丢弃，并要求 `score >= 0.90`。

最终 strict split 实测结果：

```text
source                         9985
text missing/failed/warned     1218
score below 0.90                432
eligible                       8335
train                          8238
val                              97
val episodes                     24
```

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

首次并行 audit 已确认 `9985/9985` 个 segment 结构完整、每片至少 124 帧、`failed = 0`。
同时发现旧 writer 没有强制
将语义天空位置的 depth 置 0；旧 audit 用整片 depth 有效率间接推断天空，因此也会误报没有天空、
但 depth 100% 有效的画面。

修复工具现在直接读取 `semantic_id == sky` 的像素，只原子重写对应 depth 为 0，非天空深度和
semantic PNG 不变。操作可中断并重跑：

```bash
make proxy-duv-repair-sky CLIPS_DIR="$CLIPS_DIR" AUDIT_WORKERS=32
make proxy-duv-audit CLIPS_DIR="$CLIPS_DIR" AUDIT_WORKERS=32
```

修复后的第二次 audit 已通过：

```text
segments       9985
audited        9985
failed            0
median_spread  2.227
warnings           0
notices          463
```

`463` 条 notice 都是不阻塞的 `no sky` 提示；没有 semantic sky 的 clip 本身不是 depth 损坏。

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

1. 只保留通过 DUV audit 的行；
2. 丢掉 caption `missing / failed / warned`；
3. 丢掉 `score < 0.90`；
4. 有 `quality_audit.json`（第 11.3 节 VLM 质量门禁）时，只保留其中 `accepted` 的 clip；
5. 按 episode 留出 24 个验证 episode；
6. 写入数据根目录 `_fastvideo/`。

```bash
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips-moge3

python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

from clip_prompts.contract import Caption

root = Path(os.environ["CLIPS_DIR"])
source = [
    json.loads(line)
    for line in (root / "encode_manifest.jsonl").read_text().splitlines()
    if line.strip()
]

audit = json.loads((root / "proxy_duv_audit.json").read_text())
captions = json.loads((root / "captions_audit.json").read_text())
assert audit["failed"] == 0
assert not audit["warnings"]

good_duv = {row["seg"] for row in audit["segment_stats"]}
text_rejected = set(captions["missing"]) | set(captions["failed"]) | set(captions["warned"])
quality_path = root / "quality_audit.json"
quality_ok = set(json.loads(quality_path.read_text())["accepted"]) if quality_path.exists() else None
eligible = []
rejected = Counter()

for row in source:
    name = row["name"]
    if name not in good_duv:
        rejected["duv"] += 1
        continue
    if quality_ok is not None and name not in quality_ok:
        rejected["vlm_quality"] += 1
        continue
    if name in text_rejected:
        rejected["text_missing_failed_or_warned"] += 1
        continue
    if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
        rejected["no_manifest_prompt"] += 1
        continue
    if "proxy_duv" not in row or "proxy_duv_video" in row:
        rejected["wrong_proxy_type"] += 1
        continue
    try:
        caption = Caption.read(root / name / "annotations" / "prompt.json")
        score = caption.provenance.get("score")
    except Exception:
        rejected["unreadable_caption"] += 1
        continue
    if not isinstance(score, (int, float)) or score < 0.90:
        rejected["score_below_0.90"] += 1
        continue
    eligible.append(row)

def episode(row):
    return str(row["name"]).rsplit("_", 1)[0]

episodes = sorted({episode(row) for row in eligible})
val_count = min(24, len(episodes))
val_episodes = {
    episodes[int(index * len(episodes) / val_count)]
    for index in range(val_count)
}

train = [row for row in eligible if episode(row) not in val_episodes]
val = [row for row in eligible if episode(row) in val_episodes]

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
summary = {
    "source": len(source),
    "eligible": len(eligible),
    "train": len(train),
    "val": len(val),
    "val_episodes": len(val_episodes),
    "score_threshold": 0.90,
    "quality_gate": quality_ok is not None,
    "rejected": dict(rejected),
}
(out / "split_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

最终新增：

```text
ABot-sub-2000-clips-moge3/
└── _fastvideo/
    ├── train.jsonl
    ├── val.jsonl
    └── split_summary.json
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
export LOG_DIR="${CACHE_DIR}_logs"

cd /workspace/FastVideo

python scripts/h3_proxy/prepare_models/verify_h3_snapshot.py \
  --path "$MODEL_PATH" --profile ref2va

NUM_SHARDS=8 STAGGER_SEC=45 LOG_DIR="$LOG_DIR" \
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

双节点各 8 张 H200 时，两边必须看到同一个 manifest、`CACHE_DIR` 和 `LOG_DIR`。两边运行同一条命令，
只改 `NODE_RANK`；节点 0 取全局 shard `0..7`，节点 1 取 `8..15`：

```bash
# 两个节点都设置；节点 0 填 0，节点 1 填 1
export NODE_COUNT=2
export NODE_RANK=0

# 每节点 8 个 encoder 进程。限制每进程 CPU 线程，避免 8×224 线程过度订阅。
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false

NUM_SHARDS=8 NODE_COUNT="$NODE_COUNT" NODE_RANK="$NODE_RANK" \
STAGGER_SEC=60 LOG_DIR="$LOG_DIR" \
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

建议先启动节点 0，约 30 秒后再启动节点 1，避免两节点同时从存储加载 16 份约 64 GB 的
Qwen3-VL。不要在两边都使用 `NODE_RANK=0`，否则会重复处理 shard `0..7`。多节点下某个节点先结束时
cache 尚未达到 8238 是正常的；等两边都结束后统一运行 `describe_cache.py`。

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
- shard `i` 处理 manifest 的 `i::N`；双节点的 `N=16`
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


---

## 11. 720p 原分辨率版本（omni：depth / semantic 分开作 reference）

为数据复用，另切一份标准 720p 语料：源视频 1920×1080 一次缩到 **1280×720**（精确 2/3），
RGB、逐帧 depth、逐帧 semantic、`proxy/duv.mp4` **全部 1280×720**，不再降到 336×192。
Wan 等其它模型直接用 1280×720 不裁剪；H3 cache 编码时再中心裁剪到 1280×704。

### 11.1 切片

```bash
cd /workspace/fastvideo_datapipe && git pull
export DATA_DIR=/data/binghe/datasets/ABot-World-Explorer-subset2000/data
export CLIPS_DIR=/data/binghe/datasets/ABot-sub-2000-clips-moge3-720p
export HF_HOME=/data/binghe/cache/huggingface HF_HUB_OFFLINE=1   # 权重在这；pod 默认缓存是空的

make clip-episodes LIMIT=8 WORKERS_PER_GPU=2 DEPTH=moge3 REFINER=sam2 PROXY_DUV=1 \
  WORK_SIZE=1280x720 DUV_SIZE=native CLIPS_DIR="$CLIPS_DIR"                 # 先试
NODE_COUNT=4 NODE_RANK=$R WORK_ROOT=/workspace/clip-work make clip-episodes WORKERS_PER_GPU=10 \
  DEPTH=moge3 REFINER=sam2 PROXY_DUV=1 \
  WORK_SIZE=1280x720 DUV_SIZE=native CLIPS_DIR="$CLIPS_DIR"                 # 全量，R=0..3
```

- `CLIPS_DIR` 必须是新目录：一个根目录只能有一种 DUV 网格，`semantic.json` 会记录
  `1280×720`，往里写 336×192 的帧会直接报错；`clip_report.json` 的 `duv_size` 也参与 resume
  判断，旧网格的 clip 不会被当成已切好。
- 体积：每帧 depth 3.69 MB，每片约 457 MB，一万片约 4.6 TB，必须放 `/data`。
- 收货与第 3–6 节完全相同（`clips-audit`、`proxy-duv-manifest`、`proxy-duv-audit`、captions、
  按 episode 切分），只是 `CLIPS_DIR` 换成新根目录；切分前先过第 11.3 节的 VLM 质量门禁。
- 文本不必重跑 VLM：窗口只取决于 episode 和 `--per-scene/--frames/--fps`，与分辨率无关
  （smoke 实测 `source_ordinals` 与 768p 版逐片相同）。`reuse_prompts.py` 只在两边
  `source_ordinals` 相同时复制 `annotations/prompt.json`，再用新 DUV 本地复核（不调 VLM）后导出：

  ```bash
  python scripts/reuse_prompts.py /data/binghe/datasets/ABot-sub-2000-clips-moge3 "$CLIPS_DIR"
  python -m clip_prompts captions-recompile --clips "$CLIPS_DIR" --reverify
  python -m clip_prompts captions-export --clips "$CLIPS_DIR" --write-txt
  python -m clip_prompts captions-audit --clips "$CLIPS_DIR" --report "$CLIPS_DIR/captions_audit.json"
  ```

  复核里行人的最小连通块按画面面积缩放（336×192 时 8 像素，1280×720 时 114 像素），
  所以原分辨率 DUV 和旧网格数出的人数一致。

### 11.1.1 SolarWM 镜像（torch 2.6）上的环境

SolarWM 训练镜像是 Python 3.10 + torch 2.6/triton 3.2。MoGe-3 的 FlexGEMM 需要更新的 triton
（报 `'dtype' object has no attribute 'itemsize'`），FastVideo 又钉 torch 2.12，所以两者都装进
独立的 `/opt/fv-venv`（Python 3.12 + torch 2.12），镜像自带的 `/opt/venv` 不动：

```bash
cd /workspace/FastVideo && git pull
python -m pip install uv
python -m uv venv --python 3.12 --seed /opt/fv-venv
python -m uv pip install --python /opt/fv-venv/bin/python -e .
cd /workspace/fastvideo_datapipe && git pull
source /opt/fv-venv/bin/activate && VENV=/opt/fv-venv scripts/setup_docker_env.sh --with-flicker
```

镜像的 `LD_LIBRARY_PATH` 以 `/lib/x86_64-linux-gnu`（系统 cuDNN 9.1）开头，会盖住 torch 2.12
自带的 cuDNN 9.20，卷积直接报 `cuDNN version incompatibility`。所以每次都用一个把 venv 自带
NVIDIA 库排在前面的入口，而不是裸 `activate`：

```bash
cat > /opt/fv-venv/env.sh <<'SH'
source /opt/fv-venv/bin/activate
_nv=/opt/fv-venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$_nv/cudnn/lib:$_nv/cu13/lib:$_nv/cusparselt/lib:$_nv/nccl/lib:$_nv/nvshmem/lib:$LD_LIBRARY_PATH"
SH
source /opt/fv-venv/env.sh     # clip-episodes、captions、encode 前都先 source 它
```

SAM 2 必须带编译好的 `sam2._C`，否则它只打一行 warning 就跳过 mask 补洞，产出和旧语料不一致；
现在 refiner 和启动器 preflight 都会直接拒绝。git 装的 SAM 2 在没有匹配 nvcc 时会静默不编，
所以用和 torch 同版本的 nvcc（torch 2.12 是 CUDA 13.0）重编，头文件用 venv 自带的：

```bash
apt-get install -y cuda-nvcc-13-0 cuda-cudart-dev-13-0
source /opt/fv-venv/env.sh
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
  CPATH=/opt/fv-venv/lib/python3.12/site-packages/nvidia/cu13/include \
  TORCH_CUDA_ARCH_LIST=9.0 SAM2_BUILD_ALLOW_ERRORS=0
/opt/venv/bin/python -m uv pip install --python /opt/fv-venv/bin/python setuptools wheel
/opt/venv/bin/python -m uv pip install --python /opt/fv-venv/bin/python --no-build-isolation \
  --no-deps --reinstall --no-cache git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
python -c "from sam2 import _C"
```

四节点时 `NODE_COUNT=4`，各 pod `NODE_RANK=0..3`，`WORKERS_PER_GPU` 必须一致。

吞吐（4×8 H200 实测）：`.work` 默认放在 clip 目录里，也就是 GCS 挂载上；1280×720 时每片要把
600 MB 以上的中间帧写一遍、读两遍，GPU 大半时间在等。`WORK_ROOT=/workspace/clip-work`
把它放到 pod 本地盘，再把 `WORKERS_PER_GPU` 从 6 提到 10（每卡约 55–75 GB 显存，主机内存约
470 GB/节点）：GPU 利用率从 12–82% 波动升到 75–100%，吞吐从约 26 片/分钟升到约 38 片/分钟。
此时瓶颈是 GPU，CPU 仍有约 60% 空闲是正常的。

### 11.2 H3 omni cache（1280×704）

```bash
cd /workspace/FastVideo && git pull
source /opt/fv-venv/env.sh            # SolarWM 镜像上；FastVideo 镜像里直接用自带 venv
export CACHE_DIR=/data/binghe/h3_proxy/cache/abot_720p_omni_704_qwen2

NUM_SHARDS=8 NODE_COUNT=4 NODE_RANK=$R STAGGER_SEC=60 LOG_DIR="${CACHE_DIR}_logs" \
scripts/h3_proxy/prepare_data/encode_proxy_shards.sh \
  --manifest "$CLIPS_DIR/_fastvideo/train.jsonl" \
  --root "$CLIPS_DIR" \
  --output "$CACHE_DIR" \
  --model-path /data/models/MiniMax-H3 \
  --num-frames 124 \
  --height 704 --width 1280 \
  --proxy-height 704 --proxy-width 1280 \
  --fit center-crop \
  --proxy-references depth semantic \
  --cwm-system w0_depth_semantic \
  --anchor-short-edge 2048 \
  --qwen-video-fps 2
```

- `--fit center-crop`：每路先等比缩放到恰好覆盖目标网格再居中裁剪；720→704 只裁上下各 8 行，
  不缩放。depth / semantic 只裁不缩，必须已是 target 分辨率，否则报错。anchor 先裁到 1280:704
  再缩放到短边 2048（3712×2048）。
- `--proxy-references depth semantic`：同一份 `duv/` 拆成两路 video reference：
  `<Video 1>` 灰度 depth（DUV 的 log-depth 通道复制三份），`<Video 2>` 纯色 semantic
  （12 类放在 RGB 3×2×2 格点上）。cache 存 `proxy_latents` `[2,24,37,44,80]` 和
  `info.proxy_references`，不再有 `proxy_latent`。
- `--cwm-system w0_depth_semantic`：新的锁哈希 system prompt，按 `<Picture 1>` anchor、
  `<Video 1>` depth、`<Video 2>` semantic 描述三路参考；w0/wn 只描述一路 proxy，和两路参考
  组合会被拒绝。

训练在 SolarWM：`configs/examples/minimax_h3/stage0p5-124f-ref2va-omni-704p-sp2.yaml`
（SP2，2×8 卡全局 batch 8），见 SolarWM `docs/backends/minimax-h3.md`。

### 11.3 VLM 质量门禁与工作台

结构性 audit 只保证每帧格式合法；闪烁、错标、漏检只能在时序上、或对照画面才看得出来。
所以每片在切分前先由 VLM 看一遍：

- `quality-judge` 为每片渲染一个 2×2 审片视频 `annotations/quality_panel.mp4`
  （RGB｜depth turbo 色，近暖远蓝、天空黑｜semantic 12 类固定色｜semantic 半透明叠 RGB，
  每格 640×360、24 fps），以 8 fps 发给 Gemini（默认 `gemini-3.8-flash`，Vertex）。
- VLM 给四项 1–5 分：`depth_temporal`、`depth_accuracy`、`semantic_temporal`、
  `semantic_accuracy`，外加带时间段的问题列表（`minor`/`major`）和自己的 verdict。
- 判定：VLM 判 reject、任一分 < 3、或有任一 `major` 问题，就拒绝；score = 四项均值 / 5。
  阈值写进每片的 `quality.json`（`--min-score`、`--allow-major`、`--ignore-verdict` 可调）。
- 本地另算两项不经 VLM 的时序指标，用于排序和交叉核对：`depth_jitter`（log-depth 码的
  三帧二阶差分中位数，1 码≈4.4% 距离；匀速运动相消，只剩抖动）和 `semantic_flicker`
  （t 帧标签与 t±3 不同、而 t±3 彼此相同的像素比例，即持续 ≤5 帧的闪烁；单调运动的边缘
  不计入）。窗口取 ±3 而不是 ±1：SAM 2 传播已经消掉单帧闪烁，±1 在实测 292 片上几乎全为 0，
  VLM 报的闪烁都持续数帧。指标改版只重算指标（`remeasured`），不重新调用 VLM。
- 只评审已有 `clip_report.json` 的完整 clip；同一 prompt 版本评过的会复用，所以可以边切边评、
  反复重跑。每片约 18 s，24 并发约 45 片/分钟，只用 CPU 和网络。

```bash
cd /workspace/fastvideo_datapipe && git pull --ff-only
source /opt/fv-venv/env.sh
clip-prompts quality-judge --clips "$CLIPS_DIR" --workers 24 --keep-going \
  --env-file /data/binghe/secrets/vertex.env
clip-prompts quality-audit --clips "$CLIPS_DIR"      # 写 $CLIPS_DIR/quality_audit.json，第 6 节切分读它
```

工作台（只监听 pod 的 127.0.0.1，经 `kubectl port-forward` 访问）：

```bash
# pod 上
clip-prompts workbench --clips "$CLIPS_DIR" --port 8765
# 本机
kubectl -n ultron-ls-gcp-aw port-forward pod/$(bcs name 154751:0) 8765:8765
open http://localhost:8765/
```

- 顶部计数（全部／接收／拒绝／待评／人工／未完成）可点击筛选；score 直方图按区间筛选；
  表格可按任一分数或本地指标排序；`#clip_xxx` 直接定位某片。
- 右侧播放审片视频（0.25×/0.5×、逐帧），列出四项分数、VLM 结论、拒绝原因、问题列表
  （点击跳到对应时间并 0.5× 播放）、本地指标和来源。
- 人工复核：`a` 接收、`r` 拒绝、`c` 清除，写 `annotations/quality_override.json`；
  人工结论优先于 VLM，`quality-audit` 和切分都按最终结论。`j`/`k` 上下切换。

720p 语料前 148 片实测：接收 72%。拒绝几乎都来自 semantic：道路／停车场被标成
infrastructure、狗被标成 vehicle、主角没被分出来（`hero_split` 未 resolve）；depth 很少触发。
