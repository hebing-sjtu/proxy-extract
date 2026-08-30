"""Copy the figures a newcomer needs into one folder, with a page that captions them.

The source files stay where the experiment scripts wrote them. This step only
assembles a viewing copy, so a broken export cannot silently replace a measured
figure with an empty placeholder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "figures"
DST = ROOT / "gallery"

# Order is the tour: first "what a correct run looks like", then the evidence
# that decided how the pipeline is wired, then the two known weak spots.
ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        "condition_output.png",
        "01 正确结果长这样",
        "从磁盘读回来的 condition：源画面、对数深度（暖色=近）、6 类语义。"
        "你自己跑完第一步，应该得到和这一排同结构的东西。",
    ),
    (
        "hero_split.png",
        "02 主角是跟出来的，不是分割出来的",
        "样本里唯一有路人的片段。红=hero，黄=npc。右侧走过的人没有被升成主角。",
    ),
    (
        "camera_qc_overview.png",
        "03 相机 QC：high 几乎全过，low 过一半",
        "对极残差。蓝条是 high，橙条是 low。入库只收 low≤1 px 的 keep 档。",
    ),
    (
        "camera_qc_detail.png",
        "04 残差落在画面哪里",
        "稀疏轨迹按偏离对极线的距离上色。左列对齐，右列 low 已经漂了。",
    ),
    (
        "appearance_gap.png",
        "05 低模还剩什么",
        "同一帧的 high / low。布局和剪影在，颜色和纹理几乎没了。",
    ),
    (
        "extraction_high_vs_low.png",
        "06 同一套模型，两路画面",
        "上三行 high，下三行 low。深度跟着走，语义在植被上散掉。",
    ),
    (
        "feasibility.png",
        "07 12 类：深度能迁，语义不能",
        "逐类 IoU。road/sky 还能看，vegetation 掉到 15%。",
    ),
    (
        "coarse6.png",
        "08 收成 6 类之后",
        "background/road 上来了，vegetation 仍是 21%。右边：这批样本几乎测不了 hero/npc。",
    ),
    (
        "scale_calibration.png",
        "09 GT 相机不是米制的",
        "三角化深度 vs 单目米制。单位在 0.8–2.0 米之间，而且逐 clip 不同。",
    ),
    (
        "scale_crosscheck.png",
        "10 第二把尺子对不上第一把",
        "人物身高估的尺度和上一张对不上。能定的只有「别拿相机去定米」。",
    ),
    (
        "duv_readback.png",
        "11 12 类 DUV 回读",
        "早期用 12 类体系写出的 condition，用 CWM 解码器读回来。",
    ),
)


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>proxy-extract 可视化</title>
<style>
  body {{ font: 16px/1.5 ui-sans-serif, system-ui; max-width: 980px; margin: 32px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  .item {{ margin: 2.4rem 0 3rem; }}
  .item h2 {{ font-size: 1.05rem; margin: 0 0 .4rem; }}
  .item p {{ color: #444; margin: 0 0 .8rem; }}
  img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
  video {{ width: 100%; background: #111; }}
  code {{ background: #f3f3f3; padding: 0 .25em; }}
</style>
</head>
<body>
<h1>proxy-extract 可视化</h1>
<p>这些图是管线在 <code>handpick29_high_low</code> 上跑出来的实测结果，
不是示意。操作步骤见仓库根目录的 <code>RUNBOOK.md</code>。</p>
{body}
</body>
</html>
"""


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"no figures directory at {SRC}")

    DST.mkdir(parents=True, exist_ok=True)
    missing = [name for name, _, _ in ITEMS if not (SRC / name).exists()]
    if missing:
        raise SystemExit(f"figures missing, run the experiments first: {missing}")

    blocks = []
    for name, title, caption in ITEMS:
        dest_name = name
        shutil.copy2(SRC / name, DST / dest_name)
        blocks.append(
            f'<section class="item"><h2>{title}</h2><p>{caption}</p>'
            f'<img src="{dest_name}" alt="{title}"></section>'
        )

    videos = [
        ("duv_c6_26_trevor_seg_0004.mp4", "12 6 类 preview：广场（Trevor）"),
        ("duv_c6_11_john_marston_seg_0313.mp4", "13 6 类 preview：有路人的街道（John）"),
        ("duv_high.mp4", "14 12 类 preview（high）"),
        ("duv_low.mp4", "15 12 类 preview（low）"),
    ]
    for name, title in videos:
        src = SRC / name
        if not src.exists():
            continue
        shutil.copy2(src, DST / name)
        blocks.append(
            f'<section class="item"><h2>{title}</h2>'
            f'<video controls muted loop src="{name}"></video></section>'
        )

    (DST / "index.html").write_text(HTML.format(body="\n".join(blocks)), encoding="utf-8")
    print(f"wrote {len(ITEMS)} figures + gallery page -> {DST / 'index.html'}")


if __name__ == "__main__":
    main()
