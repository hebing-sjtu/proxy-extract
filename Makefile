# One-word entry points for the first time through. Every target is a thin
# wrapper around a command already documented in RUNBOOK.md — if a target does
# something the runbook does not mention, that is a bug in one of the two.

# The maintained deployment path. These nodes already carry a working main
# environment, so the pipeline installs into its own directory and is uninstalled
# by deleting it.
VENV ?= $(CURDIR)/.venv
VPY  := $(VENV)/bin/python

# Where the corpus is and where the delivered set goes. The same defaults
# scripts/run_scenes.sh and scripts/doctor.py carry, restated here so that
# `make scenes DATA_DIR=...` reads the way it looks like it should.
DATA_DIR ?= /data/binghe/datasets/ABot-World-Explorer-subset2000/data
OUT_DIR  ?= /data/binghe/datasets/ABot-seg-long-2000

# One episode per worker holds ~40 GiB of host RAM, so this is bounded by
# MemTotal rather than by VRAM; run_scenes.sh checks it and warns.
WORKERS_PER_GPU ?= 1

.PHONY: help venv venv-core venv-test venv-fetch doctor scenes scenes-audit preview

help:
	@echo "VENV     = $(VENV)"
	@echo "DATA_DIR = $(DATA_DIR)"
	@echo "OUT_DIR  = $(OUT_DIR)"
	@echo
	@echo "装环境（RUNBOOK 第 1-2 节）"
	@echo "  make venv        建 .venv，装死 pin 的依赖，跑自检"
	@echo "  make venv-core   同上但不装 torch（只跑合约/分类/编码）"
	@echo "  make venv-test   在 .venv 里跑测试"
	@echo "  make venv-fetch  拉权重"
	@echo "  make doctor      一次查完所有前置条件，不中途退出"
	@echo
	@echo "交付（RUNBOOK 第 3-4 节）"
	@echo "  make scenes         多卡跑 720p 交付场景"
	@echo "  make scenes-audit   统计 complete/incomplete/missing"
	@echo "  make preview        把 SCENE= 渲成可看的 contact sheet"
	@echo
	@echo "路径用 DATA_DIR= 和 OUT_DIR= 覆盖，worker 数用 WORKERS_PER_GPU=。"
	@echo "其余传给 scenes 的 flag 走 SCENES_ARGS=，见 RUNBOOK 第 6 节。"

# DA3=1 because depth_anything_v3 is the default depth backend and cannot go in
# requirements.txt; see RUNBOOK section 5.
venv:
	DA3=1 scripts/setup_venv.sh

venv-core:
	EXTRAS=core scripts/setup_venv.sh

venv-test:
	$(VPY) -m pytest proxy-extract/tests -q

venv-fetch:
	$(VPY) scripts/fetch_models.py --set default

doctor:
	DATA_DIR=$(DATA_DIR) OUT_DIR=$(OUT_DIR) $(VPY) scripts/doctor.py

scenes:
	DATA_DIR=$(DATA_DIR) OUT_DIR=$(OUT_DIR) WORKERS_PER_GPU=$(WORKERS_PER_GPU) \
	  scripts/run_scenes.sh

scenes-audit:
	$(VPY) -m proxy_extract scenes-audit --out $(OUT_DIR)

# The delivery format is unviewable by construction, so this is the only way to
# judge a run by eye. See RUNBOOK section 4.
SCENE ?= $(OUT_DIR)/seg_000000

preview:
	$(VPY) -m proxy_extract scenes-preview --scene $(SCENE) --out /tmp/sheet.png --frames 6
	@echo "open /tmp/sheet.png"
