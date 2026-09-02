# One-word entry points for the first time through. Every target is a thin
# wrapper around a command already documented in RUNBOOK.md — if a target does
# something the runbook does not mention, that is a bug in one of the two.

# Image coordinates. Set REGISTRY to publish somewhere a node can pull from:
#   make push REGISTRY=registry.example.com/team
# Leaving it empty keeps the image local, which is all a single node needs.
NAME     ?= proxy-extract
VERSION  ?= 0.1.0
REGISTRY ?=
IMAGE    ?= $(if $(REGISTRY),$(REGISTRY)/,)$(NAME):$(VERSION)

# The GPU nodes are linux/amd64 and the torch wheel that bundles CUDA is only
# published for that platform, so an image built on an arm64 laptop cannot run
# there at all. Always state the target platform rather than inheriting the
# builder's.
PLATFORM ?= linux/amd64

# Weights live here on the host, shared by compose and run_shards.sh.
HF_CACHE_DIR ?= $(CURDIR)/.hf-cache

COMPOSE := IMAGE=$(IMAGE) HF_CACHE_DIR=$(HF_CACHE_DIR) docker compose -f docker/docker-compose.yml
UID_ENV := DOCKER_UID=$(shell id -u) DOCKER_GID=$(shell id -g)

# A representative clip: urban, one person, already measured end-to-end.
CLIP ?= 26_trevor_seg_0004

# The maintained deployment path. Targets below that do not need a container go
# through this interpreter rather than compose.
VENV   ?= $(CURDIR)/.venv
VPY    := $(VENV)/bin/python

.PHONY: help venv venv-core venv-test venv-fetch scenes scenes-audit \
        image version build buildx push pull save load test fetch qc extract \
        convert preview gallery shards

help:
	@echo "IMAGE = $(IMAGE)   PLATFORM = $(PLATFORM)"
	@echo "VENV  = $(VENV)"
	@echo
	@echo "venv（维护的主路径，见 RUNBOOK 第 2 节）"
	@echo "  make venv        建 .venv，装死 pin 的依赖，跑自检"
	@echo "  make venv-core   同上但不装 torch（只跑合约/分类/QC/编码）"
	@echo "  make venv-test   在 .venv 里跑测试（应看到 378 passed）"
	@echo "  make venv-fetch  用 .venv 拉权重"
	@echo
	@echo "交付数据（RUNBOOK 第 12/13 节）"
	@echo "  make scenes       多卡跑 720p 交付场景，DATA_DIR= OUT_DIR= 覆盖路径"
	@echo "  make scenes-audit 统计 complete/incomplete/missing"
	@echo
	@echo "镜像（备选路线，RUNBOOK 第 3 节）"
	@echo "  make build     在本机构建（本机就是 $(PLATFORM) 时用这个）"
	@echo "  make buildx    交叉构建到 $(PLATFORM)（在 arm64 Mac 上必须用这个）"
	@echo "  make push      推到 registry，需要 REGISTRY=..."
	@echo "  make pull      从 registry 拉，需要 REGISTRY=..."
	@echo "  make save      导出 $(NAME)-$(VERSION).tar.gz（没有 registry 时）"
	@echo "  make load      导入上面那个 tar"
	@echo
	@echo "容器里跑"
	@echo "  make test      不需要 GPU / 权重的自检（应看到 378 passed）"
	@echo "  make fetch     下载权重到 $(HF_CACHE_DIR)"
	@echo "  make qc        对 high 渲染做相机 QC"
	@echo "  make extract   抽一条 clip（CLIP=$(CLIP)）"
	@echo "  make convert   整个 high/ 目录，多卡（N=8）"
	@echo "  make preview   把 extract 的结果渲成 out/preview.mp4"
	@echo "  make gallery   把实测图册拷到 gallery/index.html"
	@echo
	@echo "服务器上的数据/输出路径用 DATA_DIR= 和 OUT_DIR= 覆盖。"

venv:
	scripts/setup_venv.sh

venv-core:
	EXTRAS=core scripts/setup_venv.sh

venv-test:
	$(VPY) -m pytest proxy-extract/tests -q

venv-fetch:
	$(VPY) scripts/fetch_models.py --set default

scenes:
	scripts/run_scenes.sh

scenes-audit:
	$(VPY) -m proxy_extract scenes-audit --out $(OUT_DIR)

image:
	@echo $(IMAGE)

# CI reads this so the published tag is not a second copy of the version.
version:
	@echo $(VERSION)

build:
	$(COMPOSE) build

# Separate from `build` because buildx needs an explicit platform and, when
# pushing a cross-built image, cannot also load it into the local daemon.
buildx:
	docker buildx build --platform $(PLATFORM) \
	  -f docker/Dockerfile -t $(IMAGE) $(if $(REGISTRY),--push,--load) .

push:
	@test -n "$(REGISTRY)" || { echo "set REGISTRY=..., e.g. make push REGISTRY=registry.example.com/team"; exit 1; }
	docker push $(IMAGE)

pull:
	@test -n "$(REGISTRY)" || { echo "set REGISTRY=..., e.g. make pull REGISTRY=registry.example.com/team"; exit 1; }
	docker pull $(IMAGE)

save:
	docker save $(IMAGE) | gzip > $(NAME)-$(VERSION).tar.gz
	@echo "scp $(NAME)-$(VERSION).tar.gz to the node, then: make load"

load:
	gunzip -c $(NAME)-$(VERSION).tar.gz | docker load

test:
	$(UID_ENV) $(COMPOSE) run --rm test

fetch:
	mkdir -p $(HF_CACHE_DIR)
	$(UID_ENV) $(COMPOSE) run --rm fetch

qc:
	mkdir -p out
	$(UID_ENV) $(COMPOSE) run --rm qc

extract:
	mkdir -p out
	$(UID_ENV) $(COMPOSE) run --rm extract \
	  extract --video /work/data/high/$(CLIP).mp4 \
	          --out /work/out/cond \
	          --semantic-backend coarse6 \
	          --depth-backend depth_anything

preview:
	mkdir -p out
	$(UID_ENV) $(COMPOSE) run --rm extract \
	  preview --condition-root /work/out/cond/high/$(CLIP) \
	          --out /work/out/preview.mp4
	@echo "open out/preview.mp4"

gallery:
	PYTHONPATH=proxy-extract/src python3 experiments/export_gallery.py
	@echo "open gallery/index.html"

# The real job: every clip under high/, one worker per GPU, resumable.
convert shards:
	mkdir -p $(or $(OUT_DIR),out)
	IMAGE=$(IMAGE) HF_CACHE_DIR=$(HF_CACHE_DIR) \
	  $(if $(DATA_DIR),DATA_DIR=$(DATA_DIR),) $(if $(OUT_DIR),OUT_DIR=$(OUT_DIR),) \
	  ./docker/run_shards.sh /work/data/high /work/out/cond $(or $(N),8)
