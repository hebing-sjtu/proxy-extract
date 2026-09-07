# clip-prompts

给 `ABot-sub-2000-clips` 的每一片写一份**带时间戳的结构化 caption**，落在
`<clip>/annotations/prompt.json`，用于 H3 的 finetune。

时间按 **1 秒**切。切片格式见 `../DATA_CLIPS.md`，本文只讲 caption。

代码注释是英文的，和 `proxy-extract` 一致。

---

## 1. 一句话说清楚它跟老的 `prompt.json` 差在哪

老的是 `contract: "scene"` v3，事件写 `chunks: [1,2,3]`——**chunk 是把这一片均分四
份的第几份**。这里是 `contract: "timeline"` v4，事件写 `bins: [1,2,3]`——**bin 恒等
于一秒**。

差别不是换个字段名：

| | v3 `scene` | v4 `timeline` |
|---|---|---|
| 时间单位 | 片长的 1/4，5.17s 片里是 1.29s，60s 片里是 15s | 恒定 1 秒 |
| 相机 | 没有 | `channel: "camera"`，和 subject 同一套规则 |
| 朝向 | 混在 `phrase` 里（"walks forward"） | `facing` 闭集，屏幕空间 |
| 连续动作的逐段文本 | 同一句话复制 N 遍 | 起始 / 延续 / 收尾分开措辞 |
| 能不能验 | 不能 | `evidence` + `checks`，DUV 免费对账 |
| 能不能切窗 | 不能 | `slice_to()`，切完还是一份合法 caption |

最后一条是为了「后续直接处理 60s 长视频」：**一条 60s episode 只调一次 VLM，切出来
的每个训练窗口的 caption 由代码切出来**，不是每个窗口再调一次。

---

## 2. 装 & 跑

```bash
pip install -e clip-prompts            # numpy + opencv，没有别的
```

VLM 的传输层**不在本包里**，用的是 `low_high_pipeline/src/mllm`（Vertex / LiteLLM /
DashScope 三选一，带 token 刷新和 429 退避）。同一个配额上挂两套重试策略迟早出事，
所以这里不复制一份。找不到的话报的是「目录不存在」而不是 ImportError：

```bash
export CLIP_PROMPTS_MLLM=/path/to/low_high_pipeline/src   # 只有不在同级目录时才需要
```

密钥读 `low_high_pipeline/.env.local`、本仓库 `.env` / `.env.local`，顺序如此。

```bash
# 0) 先不花钱：看 DUV 表和将要发出去的 prompt 长什么样
python -m clip_prompts captions-evidence --clip /data/.../clip_000000_0
python -m clip_prompts captions-evidence --clip /data/.../clip_000000_0 --json
python -m clip_prompts captions-evidence --clip /data/.../clip_000000_0 --sheet /tmp/sheet.jpg

# 1) 先跑 4 片看结果
python -m clip_prompts captions --clips /data/binghe/datasets/ABot-sub-2000-clips \
  --limit 4 --workers 4 --keep-sheet --report /tmp/smoke.json

# 2) 全量。--keep-going 让一片坏只损失一片
python -m clip_prompts captions --clips /data/binghe/datasets/ABot-sub-2000-clips \
  --workers 16 --keep-going --report /tmp/captions.json

# 3) 验收
python -m clip_prompts captions-audit --clips /data/.../ABot-sub-2000-clips
python -m clip_prompts captions-audit --clips /data/.../ABot-sub-2000-clips --list failed

# 4) 看一片
python -m clip_prompts captions-show --prompt /data/.../clip_000000_0/annotations/prompt.json
```

**`captions` 是唯一花钱的命令。** 已经有 `prompt.json` 的片默认跳过，`--overwrite`
才重跑。改了措辞规则不用重调 VLM：

```bash
python -m clip_prompts captions-recompile --clips /data/.../ABot-sub-2000-clips
```

切窗（60s 那条路线用的就是它）：

```bash
python -m clip_prompts captions-slice --prompt .../prompt.json --start 10 --stop 15 --out win.json
```

### 第一次上节点，按这个顺序看

第 0 步先跑，**它一分钱不花**，能挡掉大部分接不上的问题：路径、DUV 解得对不对、
bin 网格是不是 5 个、发出去的 prompt 读起来通不通。

然后 `--limit 4 --keep-sheet` 跑四片，四件事要亲自看：

1. **`prompt_sheet.jpg`**（`--keep-sheet` 才留）。每格烙着 bin 号，拿它跟
   `compiled.timed.script` 逐行对，确认事件没有整体早一秒或晚一秒——这是最容易出
   而且最难在下游发现的错。
2. **`checks.fail`**。应该是空的。不空说明 VLM 在无视 evidence 表，先改 prompt 而
   不是放宽检查。
3. **`checks.warn` 的分布**。全都在报"说没动但深度场在动"，那是
   `STATIC_CODE_DELTA` 该调，不是 caption 错。
4. **`provenance.attempts`**。大面积是 2 或 3 说明结构约束太紧或 schema 说得不够
   清楚，一次调用能过的比例是这条管线的主要成本项。

`--report` 写的那份 JSON 里有每片的 `attempts` / `fail` / `warn` / `score`，
`usage` 在每片的 `provenance` 里，够估全量的 token 账。

---

## 3. `prompt.json` 里有什么

完整样例在 `example/prompt.json`（是代码真跑出来的，不是手写的）。

```
contract    "timeline"
version     4
compiler    1          ← render.py 的版本。措辞改了它变，可以据此挑片重编
window      这一片是谁的哪一段：clip / episode / t0 / frames / fps / source_ordinals
timeline    bin 网格：每个 bin 的 [start, stop) 秒 和 [first_frame, stop_frame) 帧
scene       medium / environment / lighting，各有 _rich 长版
entities    id / kind / look / look_rich / protagonist / bins（在场的秒，不是动作的秒）
events      subject 和 camera 两个 channel 混在一个列表里
evidence    DUV 逐秒实测，没有模型参与
checks      evidence 和 caption 对不上的地方，分 fail / warn
compiled    lean / rich / timed 三种渲染，外加 conditioning
provenance  模型、后端、重试次数、token、score、时间
```

### events

```json
{
  "id": "e1", "channel": "subject", "entity": "char_0",
  "verb": "walk", "phrase": "forward along the road",
  "phrase_rich": "steadily along the asphalt roadway toward an intersection",
  "facing": "back", "bins": [1, 2, 3, 4], "t": [1.0, 5.167],
  "confidence": 0.9, "starts_before": false, "continues_after": false
}
```

三个地方值得单独说：

**`t` 是算出来的，不是模型写的。** 让语言模型报浮点秒数，它会给你一个很像样的数
字；让它从一个列出来的 bin 列表里挑索引，它挑的是能指出来的东西。所以 VLM 只写
`bins`，`t` 由 `bind()` 从网格填。

**`phrase` 不含动词，也不含主语。** `verb` 是词元，`vocab.py` 存变位，`render.py`
负责拼。这样同一个事件在它开始的那秒和延续的那几秒可以措辞不同——见下一节。

**`starts_before` / `continues_after` 只有切窗会置位。** 分两个 flag 而不是一个
`truncated`，因为渲染要区分：窗口打开时已经在走的动作不能写成「开始走」，窗口关闭
时还在走的不能写成「走完了」。一个 flag 会把其中一个写错。

### compiled

```
[0s-1s]     A man in a dark green jacket stands still in the middle of the roadway.
            The camera follows from behind at shoulder height.
[1s-2s]     The man walks forward along the road.
            The camera keeps following from behind at shoulder height.
[2s-3s]     The man keeps walking forward along the road.  ...
[3s-4s]     The man is still walking forward along the road.  ...
[4s-5.167s] The man is still walking forward along the road.  ...
```

- `lean` / `rich`：同一批事件的两个详略级别，用来在一个训练集里混长短 condition。
- `timed`：`bins` 是结构化的 `{bin, t, text}`，`script` 是拼好的一整块。
  默认标记是 `[0.00s-1.00s]`，**这不是随便挑的**：H3 released examples 里窗口 prompt
  本来就长这样（`code-world-model/examples/config.multiwindow.example.json` 是
  `"[0.00s-5.17s] ..."` 和 `"[3.75s-8.92s] ..."`），两位小数、绝对时间。逐秒 caption
  因此是在**延长模型见过的约定**，不是教它一套新记号。
  上游的时间是绝对的，所以 `script` 会把 `window.t0` 加上去（`timed.offset` 记着加了
  多少），而 `timed.bins[].t` 保持片内相对——结构化的那份才是真值。
  想换写法是 `render.script(marker=...)` 一个参数，不用重跑语料。
- 第一次提到某个实体用完整描述，之后用 `the man`。切片会重新建立这个「第一次」。

`[4s-5.167s]` 那个尾巴不是 bug：124 帧 24fps 是 5.1667 秒，最后 0.1667 秒只有 4 帧，
不够描述任何东西，所以并进前一个 bin，这一片是 5 个 bin 而不是 6 个。尾巴既不四舍
五入也不丢掉，bin 里写的是它真实的起止。门槛是 `--min-tail-seconds`（默认 0.5）。

---

## 4. DUV 当免费的裁判

`evidence.py` 解 `proxy/duv.mp4`，逐秒给出：主角在不在、几个路人、有没有车、天空/
路面/植被占比、深度 p10/p50/p90、以及相邻帧深度码的平均变化（「世界动没动」）。
**一次模型调用都不花，一块 GPU 都不用。**

它同时是两样东西：

- **喂给 VLM 的先验**：告诉它第 3 秒有 1 个主角、2 个路人、没有车，它就不会把街上
  「应该有」的停车脑补出来。
- **收到回复后的裁判**：说了车但那几秒一个 vehicle 像素都没有 → `fail`。

`checks` 分两档，这个分法是重点：

- `fail`：DUV 直接打脸的（凭空的车、不在场的主角）。是幻觉，训练集里留着就是教模型
  生成 condition 没要求的东西。
- `warn`：有无辜解释的（路人数差 1 是 blob 太小或者两个人重叠；说「都没动」但深度
  场在动是跟拍相机在动）。值得统计，不值得为它丢片。

跨语料看哪一档在涨，含义不一样：`fail` 涨是 prompt 抓不住 evidence 了，`warn` 涨
通常是 `verify.py` 里某个阈值该挪了。

两个已知限制，来自 `DATA_CLIPS.md`：DUV 的调色板**不是单射的**（building / ground /
terrain / water / prop 五类共用 (0,0)），所以这里只报「static other」；主角/路人的
区分来自一个会失手的追踪器，`hero_resolved=False` 时主角那条检查自动关掉。

`STATIC_CODE_DELTA`（世界动没动的阈值）是本包里**唯一没有在这份语料上拟合过**的
数字，而且只驱动 warn。拟它的办法是跑几百片 `captions-audit` 看那条 warn 触发得多
频繁。

---

### compiled.conditioning —— 控制信号的含义

H3 除了文本还看 `proxy/duv.mp4`，文本得说清楚那一路是什么，否则"跟着视频走"就是在
邀请模型去渲染调色板。这件事拆成两半：

```json
"conditioning": {
  "card": "duv.abot.v1",
  "stream": "proxy/duv.mp4",
  "contents": "In this clip the control video holds road surface, vegetation, static structure, sky and one protagonist, and no vehicles."
}
```

**编码含义是整个语料的一句话，只存在 `conditioning.py` 里，每片只存一个 card id。**
把那段话写进 9985 个文件，是 9985 个要一起改的地方；更要命的是，**在每个训练样本上
都一模一样的前缀不携带任何信息**，却在每个样本上都要付 token，然后变成 finetune 后
模型离不开的一句咒语——推理时漏掉就出分布。

**这一片实际有什么是逐片不同的**，那部分从 `evidence` 编译出来，真有信息，值它的
token。开车的片和走路的片这句话不一样。

拼回完整段落用 `conditioning.full_text()`，或者：

```bash
python -m clip_prompts captions-show --prompt .../prompt.json --conditioning
```

它是 `compiled` 下独立的一块，**没有粘进 `lean` / `rich`**，所以训练时可以带、可以
不带、可以按比例混，不用重编语料。措辞上刻意写成"控制信号的含义是 X"而不是"画面看
起来像 X"——`NOTES.md` 第 1 条，职责混了就会去渲染那些平涂色块。

---

## 5. 60 秒长视频怎么接

契约在时间切片下是封闭的，所以是这样：

```
一条 60s episode → 60 个 bin → 一次 VLM 调用 → 一份 prompt.json
                                                    │
                     ┌──────────────────────────────┼──────────────────────┐
                slice_to(0,5)                 slice_to(12,17)        slice_to(55,60)
                     │                              │                      │
              窗口自己的 caption            窗口自己的 caption      窗口自己的 caption
```

`slice_to` 做四件事：bin 重新从 0 编号、事件按窗口取交集、被切开的事件打上
`starts_before` / `continues_after`、窗口里根本没出现的实体删掉。切出来的东西
`problems()` 是空的——它就是一份合法 caption，测试里对 0/10/55 三个起点都验了。

代价是 60 个 bin 的回复比 5 个长得多。真跑长片建议 12s 左右一段带重叠地问，再把
结果拼起来（`Caption` 是 frozen dataclass，拼接是纯数据操作，没有实现在这里）。

---

## 6. 对老 v3 格式的几点意见

按影响排序，前四条已经落在 v4 里了：

1. **chunk 索引换成秒。** 同一个 `chunks: [1]` 在 5s 片里是 1.29s、在 60s 片里是
   15s，同时训这两种，模型学不出 chunk 值多少钱。
2. **`chunks: [1,2,3]` 渲染出三句一模一样的话**——`compiled.lean.chunks` 里
   `"The man walks forward."` 连着三遍。这是**负价值的监督**：模型学到的是「时间戳
   不预测任何东西」。v4 用起始/延续/收尾的变位解决，不需要 VLM 多写一个 token。
   注意：真的十秒不变的动作，中间那几行**仍然是重复的**，这是故意的——重复是事实，
   区分它们的是时间戳。轮换同义词会让文本好看，同时教会模型「换个说法」是信号。
3. **完全没有相机。** 第三人称开放世界里，相机是大多数帧里最大的运动。
   `NOTES.md` 第 2 条本身就是「语义图锁不住走位，必须另写一段运动说明」。
4. **朝向没有独立字段。** `"walks forward"` 正是 `NOTES.md` 第 10 条花了钱买回来的
   那个歧义：对写的人是「走进画面」，对读的模型是「脸朝镜头」。
5. **`target` / `caused_by` / `channel` / `mode` 全是 null，`initial.bindings` 是
   空的。** 训练契约里的死字段比没有这个字段更糟：它在邀请 VLM 往里编结构。要么
   定义闭集并校验，要么删掉。v4 只留了一个 `target`，并且校验它指向存在的实体。
6. **没有置信度，也没有弃权的出口。** 一万片规模上 VLM 一定会幻觉。没有
   `confidence`、没有 `unknown`，你既不能挑掉最差的一批，也逼着模型在看不清时必须
   编一个。
7. **没有任何东西是可验的。** 这是最可惜的一条——`duv.mp4` 就躺在同一个目录里，
   逐帧告诉你有几个人、有没有车、主角在不在。不用它，等于把唯一一个免费的真值源
   扔了。
8. **`enrichment` 是第二遍改写。** 短版和长版可能互相矛盾。一次要两个长度，几百个
   token 的事，把这类矛盾整个消掉。
9. **`derived_from: "minimax_h3/output.mp4"`。** 那份 caption 描述的是 **H3 生成的**
   视频，不是真实素材。用生成数据 finetune 生成模型要么是有意为之要么是事故，无论
   哪种都不能悄悄混。v4 的 `provenance` 和 `window.source_video` 把这件事写明。
10. **`compiled` 存下来是对的**（确定性、可复现），但要记编译器版本，否则没法知道
    哪些片的文本是哪一版规则出的。v4 是顶层的 `compiler` 字段。

还有一条不是格式问题而是训练问题，得自己拿主意：

**H3 的 prompt 写手明确要求「不要用时钟时间戳」**，因为推理时的 prompt 里没有时间
戳。带时间戳 finetune 是在**教它一个新能力**，不是在对齐它原来的分布。所以
`compiled` 里同时留了不带时间戳的 `lean.global` / `rich.global`，建议按比例混进去，
免得模型变成「没有时间戳就不会写了」。

---

## 7. 目录

```
src/clip_prompts/
    timeline.py   1 秒网格、span↔bin、切窗算术
    contract.py   v4 schema、校验、slice_to
    vocab.py      facing / verb / kind 闭集和变位表
    evidence.py   DUV 逐秒实测 + contact sheet
    prompts.py    发给 VLM 的指令和回复 schema
    observe.py    调用 + 解析 + 修复循环
    verify.py     caption 对 evidence 的对账
    conditioning.py  控制信号含义：语料级 card + 逐片内容
    render.py     结构 → lean / rich / timed 文本
    layout.py     片目录解析，prompt.json 放哪
    cli.py        六个子命令
example/prompt.json
tests/
```

跑测试：

```bash
python -m pytest clip-prompts/tests -q
```

不需要 GPU、不需要网络、不需要语料——80 个用例全部在合成数据上跑。
