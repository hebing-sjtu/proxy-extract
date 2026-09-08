# 补丁：把 `prompt.json` 投影成 CWM 用户 caption

写给要实现这一层的 datapipe agent。**这里只规定文字如何最终落地**；不要重跑
VLM，不要改 DUV / RGB，不要碰 FastVideo 的 conditioner。`prompt.json` 的结构
已经够用，缺的是一层确定性的导出。

对照实现时打开这三份，不要凭记忆：

| 文件 | 它锁死的东西 |
| --- | --- |
| `code-world-model/INFERENCE.md` | 用户 caption 的时间戳、CRLF、system 不可覆盖 |
| `code-world-model/src/cwm_h3_inference/config.py` 的 `_canonical_caption` | 行尾规范化的逐字节规则 |
| `code-world-model/src/cwm_h3_inference/presentation.py` | Qwen 里 system / user 怎么叠 |
| `code-world-model/src/cwm_h3_inference/prompts/system_w0.txt` / `system_wn.txt` | frozen system 正文 |
| `example/prompt.json` | 本仓库的黄金结构样例 |
| `src/clip_prompts/render.py` | 标记语法已经和发布例子对齐 |

本文件是合同。实现之后用第 8 节的验收句自己打勾，不要另写一份「差不多」的格式。

---

## 0. 一句话

CWM 喂给 Qwen 的用户句是**一个字符串**：前缀一个窗口时间戳，后面一段只描述
**这一窗**的散文。结构化留在 `prompt.json` 里；进模型的永远是这个扁字符串。

datapipe 的产出止于这个用户句（外加在 `compiled` 里记下它）。  
`AWM_PROXY_CONTROL` 那段 system **不写进用户句**，也不写进 `prompt.txt`。

---

## 1. 边界：谁写什么

```
prompt.json (结构, 已有)
        │
        ▼  本补丁（纯函数 + 写盘）
compiled.cwm + <clip>/prompt.txt     ← 用户 caption，CRLF
        │
        ▼  FastVideo（另一条线，本补丁不做）
Qwen chat:
    system  = system_w0.txt 或 system_wn.txt
    user    = <Picture 1> + <Video 1> + 上面的用户 caption
```

FastVideo 今天的 `build_ref2va_presentation` 是扁拼接、没有 system 块。那是
训练侧的缺口，不是 caption 导出的缺口。**不要**为了「现在 FastVideo 没有
system」就把 `AWM_PROXY_CONTROL` 粘到用户句前面——CWM 推理是 system 角色，
粘进 user 会变成另一套 token，以后加 chat template 还会出现两次。

`compiled.conditioning`（`duv.abot.v1` card）同样不进用户句。CWM 把「Video 1
是粗几何、不是成片」写在 system 里。card 继续只通过 `conditioning.full_text()`
给消融用。

---

## 2. 最终用户句长什么样

### 2.1 默认变体 `window`（对齐发布例子，大多数样本用这个）

一行。标记和 `code-world-model/examples/config.example.json` 同构：

```
[0.00s-5.17s] Third-person open-world video game. City street. Bright daylight. A man in a dark green jacket stands still in the middle of the roadway, then the camera follows from behind at shoulder height, then the man walks forward along the road.
```

这就是 `example/prompt.json` 的 `compiled.lean.global` 前面加上**整窗**时间戳。
中间一个空格，没有换行。`captions-show --style timed` 打出来的逐秒 `script`
不是这个默认值。

黄金断言（对 `example/prompt.json` 必须 Exact）：

```
[0.00s-5.17s] Third-person open-world video game. City street. Bright daylight. A man in a dark green jacket stands still in the middle of the roadway, then the camera follows from behind at shoulder height, then the man walks forward along the road.
```

### 2.2 少数变体 `timed`（同一套记号，切细）

`compiled.timed.script` 原样，已经是：

```
[0.00s-1.00s] …
[1.00s-2.00s] …
[4.00s-5.17s] …
```

五行的并集必须等于该窗的 `window` 戳。这是延长 CWM 记号，不是新记号。只按
比例混进训练集，默认不要当全量。

### 2.3 不要导出的东西

| 不要 | 为什么 |
| --- | --- |
| `annotations/caption.json` | episode 级 60 秒文案，会教模型在 5 秒里扫完整集 |
| `compiled.conditioning` 全文 | 常数前缀，推理咒语；对等物在 CWM system |
| `AWM_PROXY_CONTROL` 粘在用户句前 | 角色错了 |
| 把 `clip_report.window`（0..4）当成 CWM 窗号 | 见第 3 节 |
| 新时间戳语法（无小数、相对秒、`t=`） | `render.UPSTREAM_MARKER` 已经锁死 |
| 把 JSON / entities / bins 喂给模型 | CWM 不吃结构 |

`compiled.rich.global` 可以另开 `--prose rich`，不是默认。  
`timed.facing`（`Opening screen facing: back.`）默认**不**进 `window` 句；CWM
发布例子没有这行。需要时用 `--facing` 接到散文后面、同一段落，不要另起时间戳。

---

## 3. 唯一会算错的两处时钟

### 3.1 ABot 的 `window: 2` 不是 CWM 的 window 2

`ABot-sub-2000-clips` 一条 episode 切 5 片、互不重叠。每片是一次**独立的**
124 帧 take。`cli.py` 的 `_window()` 已经把 `t0` 写成 `0.0`，包括
`clip_000414_2`。

CWM 的 `window_index` / `system_wn` 只用于 **Retake34 续跑**：同一条 take、
`start_frame` 每次 +90、用户戳是 `[3.75s-8.92s]`。

因此：

```
cwm_system = "wn"  当且仅当  float(window.t0) > 0
cwm_system = "w0"  否则
```

**禁止**用 `window.window`（片号 0..4）或目录名里的 `_2` 去选 `system_wn`。
那会把 80% 的 ABot 片标成续跑，和像素（每片都有自己的 `anchor.png` 作第 0
帧）矛盾。

本批 subset 导出应全部是 `w0` + `[0.00s-5.17s]`。`compiled.cwm.system` 只是
给 FastVideo 看的标签，不进 `prompt.txt`。

### 3.2 整窗戳用输出绝对时间，两位小数

和 `render.UPSTREAM_MARKER` 同一套：`[{start:.2f}s-{stop:.2f}s]`。

```
start = float(window.t0 or 0)
stop  = start + duration
duration = float(window.duration)
         否则 grid 最后一档的 stop（片内秒，未加 offset）
```

`124 / 24 = 5.1666…`，`:.2f` 得到 `5.17`。续跑窗 `t0=3.75`、时长 5.17 →
`[3.75s-8.92s]`，与 `config.multiwindow.example.json` 一致。

`timed.bins[].t` 保持片内相对；只有写到用户句上的标记才加 `t0`。  
`window.duration` 在样例里是 `5.167`，格式化两位小数后必须是 `5.17`，不要
先 round 成 `5.17` 再格式化出 `5.17` 的另一条路径——只用 `:.2f`。

---

## 4. 行尾：进盘的字节要和 CWM 一样

从 `cwm_h3_inference.config._canonical_caption` 原样搬过来，不要「差不多」：

```python
def canonical_caption(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("caption must be a non-empty string")
    normalized = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "\r\n")
```

- JSON 里的 `compiled.cwm.user` **存 LF 形式**（`strip` 后、换行是 `\n`）。
  JSON 转义 CRLF 既难看又容易被编辑器改掉。
- 写 `prompt.txt` 时再跑 `canonical_caption`，UTF-8、无 BOM。
- 文件末尾：`canonical_caption` 的 `strip()` 会去掉尾空白，所以默认变体
  **没有**末尾换行。不要再补一个 `\n`。FastVideo 的
  `clip_dir_to_encode_manifest.read_prompt` 会再 `strip()` 一次，多一个换行
  会被吃掉，但和 CWM 缓存逐字节对比时会分叉。
- `timed` 变体：行与行之间是 CRLF，最后一行后面没有多余 CRLF。

`DATA_CLIPS.md` 第 5.4 节已经写了这条规矩，导出必须遵守。

---

## 5. 写到磁盘的哪里

### 5.1 `compiled.cwm`（`prompt.json` 里，和 lean/rich/timed 并列）

`render.compile_all` 增一块。`COMPILER_VERSION` 从 1  bump 到 **2**（文本
集合变了，即使 lean/rich/timed 字节没变）。已有语料用现成的
`captions-recompile` 回填，禁止为此重跑 VLM。

```json
"cwm": {
  "variant": "window",
  "system": "w0",
  "t": [0.0, 5.167],
  "user": "[0.00s-5.17s] Third-person … along the road.",
  "timed": "[0.00s-1.00s] …\n[1.00s-2.00s] …\n…"
}
```

| 字段 | 规则 |
| --- | --- |
| `variant` | 这份 `user` 用的是 `window` 还是 `timed`。compile 时默认 `"window"` |
| `system` | `"w0"` / `"wn"`，按第 3.1 节，**只是标签** |
| `t` | `[start, stop]`，秒，未做 `:.2f` 的真值（`0.0` 和 `0.0+duration`） |
| `user` | 默认变体的 LF 字符串，训练主路径读这个 |
| `timed` | `compiled.timed.script` 的副本（LF），给 mix / 消融，避免再拼一次 |

`user` 在 `t0=0` 的 124 帧片上必须以 `[0.00s-5.17s] ` 开头。  
`slice_to(3.75, 8.92)` 之后必须以 `[3.75s-8.92s] ` 开头，且 `system` 为 `"wn"`。

不要把 card 文本、facing、rich 散文默默拼进 `user`。

### 5.2 `<clip>/prompt.txt`

FastVideo `clip_dir_to_encode_manifest.py` 已经**先读**这个文件，再回落
`caption.json`。导出写在这里，现有 manifest 构建器不用改就能吃到 CWM 句。

- 路径：`<clip>/prompt.txt`（和 `clip_report.json` 同级，**不是**
  `annotations/prompt.txt`）。`layout.Clip` 加一个 `prompt_txt` 属性。
- 内容：`canonical_caption(compiled.cwm.user)`，或 `--style timed` 时对
  `compiled.cwm.timed` 做同样规范化。
- 已有 `prompt.txt`：默认跳过；`--overwrite` 才覆盖。不要静默覆盖人手写的。
- `checks.fail` 非空：默认**不写** `prompt.txt`（幻觉会进训练）。
  `--keep-failed` 才写。`compiled.cwm` 仍然可以在 recompile 时生成，过滤发生
  在写 `prompt.txt` 这一步。

禁止回落写 `caption.json` 的内容。一片没有可用 `prompt.json` 就跳过并计数，
不要发明一句 fallback 游记。

---

## 6. 建议的代码形状

新模块 `src/clip_prompts/cwm_export.py`，纯函数，不读盘也可以测：

```python
WINDOW_MARKER = "[{start:.2f}s-{stop:.2f}s]"   # 与 render.UPSTREAM_MARKER 相同

def system_id(caption: Caption) -> str: ...
def window_span(caption: Caption) -> tuple[float, float]: ...
def window_user(caption: Caption, *, prose: str = "lean") -> str: ...
def timed_user(caption: Caption) -> str: ...
def canonical_caption(value: str) -> str: ...
def compile_cwm(caption: Caption, *, variant: str = "window") -> dict: ...
def write_prompt_txt(path: Path, user: str) -> None: ...
```

`window_user`：

```
f"{WINDOW_MARKER.format(start=t0, stop=t1)} {compiled['lean']['global'].strip()}"
```

`prose="rich"` 换 `compiled['rich']['global']`。global 为空则失败，不要只输出
一个秃戳。

`compile_all` 末尾：

```python
out["cwm"] = compile_cwm(caption, variant="window")
```

`captions-recompile` 已经会调 `compile_all`，回填旧语料不用新命令也能做。
仍要加一个显式命令，方便只写 `prompt.txt`、不改每个人心里的「recompile = 重渲
lean/rich」。

### CLI：`captions-export`

```
python -m clip_prompts captions-export \
    --clips /data/binghe/datasets/ABot-sub-2000-clips \
    --style window \
    --write-txt
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--clips` | 必填 | 根目录或单片 |
| `--style` | `window` | `window` / `timed` |
| `--prose` | `lean` | `window` 变体用 lean 还是 rich global |
| `--write-txt` | off | 写 `<clip>/prompt.txt` |
| `--overwrite` | off | 覆盖已有 `prompt.txt` |
| `--keep-failed` | off | `checks.fail` 非空也写 txt |
| `--limit` | | 按名字序前 N 片 |
| `--report` | | JSON：written / skipped / failed / missing |

`captions-show` 加 `--style cwm`（或 `--cwm`），打印 LF 的 `user`，并在
stderr 打一行 `system=w0`（提醒实现者它不在用户句里）。不要在 stdout 混进
system 正文。

`captions-audit` 最好加一档 `exported`：有 `prompt.txt` 且开头匹配
`[0.00s-` 或更一般的 `^\[[0-9]+\.[0-9]{2}s-`。不是必须，有则加。

---

## 7. 测试（不需要 GPU、不需要语料盘）

新文件 `tests/test_cwm_export.py`，用现有 `caption` / `long_caption` fixture
和 `example/prompt.json`。

必须覆盖：

1. `example/prompt.json` → `window_user` 与第 2.1 节黄金字符串 **Exact**。
2. 该样例 `system_id == "w0"`，尽管 `window.window == 2`。
3. `canonical_caption` 把 `\n` 变成 `\r\n`，idempotent，空串 raise。
4. `slice_to(10, 15)`（现有 long fixture）→ 用户句以 `[10.00s-15.00s] `
   开头、`system_id == "wn"`；`timed` 第一行仍是 `[10.00s-11.00s]`。
5. `compile_all` 含 `cwm.user` / `cwm.timed`，`compiler` 为 2。
6. `write_prompt_txt` 落盘字节 `== canonical_caption(user).encode("utf-8")`，
   无 BOM、无多余尾换行。
7. `checks.fail` 非空时默认不写 txt（用临时目录测 CLI 或抽一层
   `should_write_txt`）。

`test_render.py` 里现有标记测试不要改语义。若 `COMPILER_VERSION` 进了
`Caption.write` 的断言，一起更新。

---

## 8. 验收

做完这些，下面每句都为真：

- `python -m clip_prompts captions-show --prompt example/prompt.json --style cwm`
  打出第 2.1 节那一整行，前面没有 system，后面没有 card。
- 对 `example/prompt.json` 导出的 `prompt.txt`，`hexdump` 里没有 `0a` 夹在
  两个非 `0d` 之间（单行默认变体根本没有换行）；若测 `timed`，每个换行都是
  `0d 0a`。
- `ABot-sub-2000-clips` 上全量 export 后，`prompt.txt` 以 `[0.00s-5.17s] `
  开头的比例 ≈ 有合格 `prompt.json` 且 `fail` 为空的片数；**没有**
  `[12.00s-` 这种把片号当成续跑时钟的文件。
- `captions-recompile` 能给旧 `compiler: 1` 的文件补上 `compiled.cwm`，不调
  网络。
- grep 导出文本，没有 `AWM_PROXY_CONTROL`，没有 `logarithmic depth code`。
- README 目录表加上 `cwm_export.py` 和本文件；不要把本文件的合同再抄一份
  进 `DATA_CLIPS.md`，只在 5.4 节加一句「训练用用户句见
  `clip-prompts/CWM_TEXT_EXPORT.md`」。

---

## 9. 明确不做

- 不改 VLM prompt、不改 `observe.py` / `verify.py`。
- 不把 DUV 转成 CWM 的 `.depth.f32` + `semantic_id.png`（另一条线）。
- 不改 FastVideo 的 chat template / text cache。导出齐了之后，训练侧要
  **重编 text embedding**；VAE latent 不用动。把这句话写在 `--report` 的
  打印里提醒调用者即可。
- 不在 datapipe 里实现 80/20 mix 的随机抽样器。`--style` 一次一种；mix 是
  两次 export 或训练 loader 的事。需要的话 `--report` 带上每片 `variant`。
- 不新增第二种标记语法。

---

## 10. 给 FastVideo 的接口（只写在这里，本补丁不实现）

读盘优先级保持：`<clip>/prompt.txt` →（不要）`caption.json`。  
有 `prompt.txt` 之后应关掉 episode caption 回落（`--no-episode-caption`）。

编码用户句之后还要：

1. 按 `compiled.cwm.system` 装入对应的 CWM system 文件（本批全是 w0）。
2. `apply_chat_template([system, user])`，user 正文里 Picture 1 在 Video 1
   前，caption 出现恰好一次，没有 assistant 轮。
3. 送进 Qwen 前再跑一遍与第 4 节相同的 `canonical_caption`。

没有这三步，导出的用户句仍然对，但和 CWM 发布 LoRA 的 token 对不齐。那是
FastVideo 的补丁，不要在 clip-prompts 里用前缀假装做完。
