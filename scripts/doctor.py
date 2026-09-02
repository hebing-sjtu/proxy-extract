"""Check every prerequisite the delivery run needs, and report all of them.

`run_scenes.sh` refuses to launch on the first thing it finds wrong, which is
right for a launcher — a run that starts with the wrong torch wastes hours — but
it means bringing a fresh node up costs one round trip per problem. This checks
everything and prints a verdict per item, so one run shows the whole gap.

    python scripts/doctor.py
    DATA_DIR=... OUT_DIR=... python scripts/doctor.py

Nothing here mutates anything or downloads weights.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.environ.get("DATA_DIR", "/data/binghe/datasets/ABot-World-Explorer-subset2000/data")
)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/data/binghe/datasets/ABot-seg-long-2000"))

# Kept equal to run_scenes.sh, which sizes the run with them.
GIB_PER_WORKER = 11
MIB_PER_SCENE = 5200
WORKERS_PER_GPU = 6


@dataclass
class Result:
    name: str
    state: str  # "ok" | "warn" | "fail"
    detail: str
    fix: str = ""


def _installed(module: str) -> bool:
    """Is the module importable without importing it? Torch takes seconds."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> Result:
    version = ".".join(str(part) for part in sys.version_info[:3])
    inside_venv = sys.prefix != sys.base_prefix
    where = "venv" if inside_venv else "NOT a venv"
    if sys.version_info < (3, 10):
        return Result(
            "python", "fail", f"{version} ({where})", "need >= 3.10; scripts/setup_venv.sh"
        )
    if not inside_venv:
        return Result(
            "python",
            "warn",
            f"{version}, {where}: {sys.executable}",
            "run this with .venv/bin/python so it checks the environment the pipeline uses",
        )
    return Result("python", "ok", f"{version} in {sys.prefix}")


def check_package() -> Result:
    try:
        import proxy_extract
    except ImportError as error:
        return Result(
            "proxy_extract", "fail", str(error), "scripts/setup_venv.sh, or pip install -e proxy-extract"
        )
    return Result("proxy_extract", "ok", str(Path(proxy_extract.__file__).parent))


def check_ffmpeg() -> Result:
    try:
        from proxy_extract.proxy import EncodeError, ffmpeg_binary
    except ImportError:
        return Result("ffmpeg", "fail", "proxy_extract not importable", "fix proxy_extract first")
    try:
        binary = ffmpeg_binary()
    except Exception as error:  # EncodeError, but do not let this check crash
        return Result("ffmpeg", "fail", str(error).splitlines()[0], "pip install imageio-ffmpeg")
    bundled = "imageio" in binary
    return Result("ffmpeg", "ok", f"{binary}{' (bundled)' if bundled else ''}")


def check_torch() -> list[Result]:
    if not _installed("torch"):
        return [Result("torch", "fail", "not installed", "EXTRAS=full scripts/setup_venv.sh")]
    import torch

    results = [Result("torch", "ok", f"{torch.__version__}, built against CUDA {torch.version.cuda}")]

    driver = ""
    driver_cuda = ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            driver = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = ""
        try:
            banner = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=30
            ).stdout
            found = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", banner)
            driver_cuda = found.group(1) if found else ""
        except (OSError, subprocess.SubprocessError):
            driver_cuda = ""

    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
    except Exception as error:
        available, count = False, 0
        results.append(Result("cuda", "fail", f"{type(error).__name__}: {error}", ""))
        return results

    if available:
        results.append(Result("cuda", "ok", f"{count} device(s), driver {driver or '?'}"))
    elif shutil.which("nvidia-smi"):
        # A cu12x wheel runs on any 12.x driver >= 525.60.13, so the fix is to
        # match the driver's major, not its minor. Across majors nothing works.
        tag = {"12": "cu126", "13": "cu130"}.get(driver_cuda.split(".")[0], "cu126")
        version = torch.__version__.split("+")[0]
        vision = importlib.metadata.version("torchvision") if _installed("torchvision") else ""
        command = f"          torch=={version}+{tag}"
        if vision:
            command += f" torchvision=={vision.split('+')[0]}+{tag}"
        results.append(
            Result(
                "cuda",
                "fail",
                f"nvidia-smi is here (driver {driver or '?'}, up to CUDA {driver_cuda or '?'}) "
                f"but torch is built for CUDA {torch.version.cuda} and sees no devices",
                f"install the {tag} build of the same versions:\n"
                f"      pip install --index-url https://download.pytorch.org/whl/{tag} \\\n"
                f"{command}",
            )
        )
    else:
        results.append(
            Result("cuda", "warn", "no nvidia-smi on this host", "CPU only; this workload needs a GPU")
        )
    return results


def check_torchaudio() -> Result | None:
    """Only interesting when it is installed and broken.

    Reinstalling torch for a different CUDA leaves torchaudio's C++ extension
    linked against the old build, and `transformers` imports torchaudio while
    loading an image processor - so a broken one takes the semantic backend down
    with it even though nothing here uses audio.
    """
    if not _installed("torchaudio"):
        return None
    try:
        import torchaudio  # noqa: F401
    except Exception as error:
        return Result(
            "torchaudio",
            "fail",
            f"installed but will not load: {type(error).__name__}",
            "pip uninstall -y torchaudio  (nothing here uses it; transformers copes without it)",
        )
    return Result("torchaudio", "ok", "loads (unused, but harmless)")


def check_backends() -> list[Result]:
    results = []

    if _installed("depth_anything_3"):
        results.append(Result("depth backend", "ok", "depth_anything_3 installed"))
    elif _installed("transformers"):
        results.append(
            Result(
                "depth backend",
                "warn",
                "depth_anything_3 missing; only DEPTH=depth_anything (V2) is available",
                "DA3=1 scripts/setup_venv.sh   (or run with DEPTH=depth_anything)",
            )
        )
    else:
        results.append(
            Result("depth backend", "fail", "neither depth_anything_3 nor transformers", "scripts/setup_venv.sh")
        )

    if _installed("transformers"):
        results.append(Result("semantic backend", "ok", "transformers installed"))
    else:
        results.append(
            Result("semantic backend", "fail", "transformers missing", "scripts/setup_venv.sh")
        )
    return results


def check_weights() -> list[Result]:
    """Are the checkpoints already local? A missing one is a download, not an error."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = home / "hub"
    results = [
        Result(
            "HF_HOME",
            "ok" if hub.exists() else "warn",
            str(home) + ("" if hub.exists() else " (no hub/ yet: nothing downloaded)"),
            "" if hub.exists() else "python scripts/fetch_models.py --set da3",
        )
    ]

    wanted = {
        "depth weights": "models--depth-anything--DA3NESTED-GIANT-LARGE-1.1",
        "semantic weights": "models--facebook--mask2former-swin-large-ade-semantic",
    }
    for label, directory in wanted.items():
        path = hub / directory
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            results.append(Result(label, "ok", f"{size / 1e9:.1f} GB in {directory}"))
        else:
            results.append(
                Result(
                    label,
                    "warn",
                    f"not in {hub}",
                    "will download on first use; to pre-fetch: "
                    "python scripts/fetch_models.py --set da3",
                )
            )
    return results


def check_data() -> list[Result]:
    results = []
    if not DATA_DIR.is_dir():
        return [
            Result("data", "fail", f"{DATA_DIR} is not a directory", "set DATA_DIR to the episode root")
        ]
    episodes = sum(1 for _ in DATA_DIR.rglob("*.mp4"))
    state = "ok" if episodes else "fail"
    results.append(
        Result(
            "data",
            state,
            f"{episodes} mp4 under {DATA_DIR}",
            "" if episodes else "no episodes found; check the path and --recursive",
        )
    )

    out = OUT_DIR if OUT_DIR.exists() else OUT_DIR.parent
    if not out.exists():
        results.append(Result("out", "fail", f"neither {OUT_DIR} nor its parent exists", "mkdir -p it"))
    else:
        usage = shutil.disk_usage(out)
        # The per-frame directories, not the videos, are what fills a disk:
        # depth and semantic are raw arrays at a fixed 2.6 MiB a frame between
        # them. KEEP_FRAMES=none is most of this back, but assume the default.
        need_gib = episodes * MIB_PER_SCENE / 1024
        free_gib = usage.free / 1024**3
        enough = free_gib >= need_gib
        results.append(
            Result(
                "out",
                "ok" if enough else "fail",
                f"{free_gib:.0f} GiB free at {out}, {episodes} episodes need ~{need_gib:.0f} GiB",
                "" if enough else "point OUT_DIR at a bigger filesystem",
            )
        )
    return results


def check_memory() -> Result | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal"):
            total_gib = int(line.split()[1]) // 1024 // 1024
            fits = total_gib // GIB_PER_WORKER
            return Result(
                "host RAM",
                "ok" if fits >= 1 else "fail",
                f"{total_gib} GiB, about {fits} worker(s) of {GIB_PER_WORKER} GiB fit",
                "" if fits >= 1 else "not enough RAM for even one worker",
            )
    return None


# What DA3 imports on the path this pipeline drives it down: load the model,
# call `inference`. Everything else it declares belongs to its demo app, its
# benchmark suite, its exporters, or its Gaussian-splatting branch, none of
# which is reachable from here — `xformers` and `e3nn` are behind try/except
# with working fallbacks, and the two submodules that hard-import the rest are
# stubbed by the backend. So `pip check` listing a dozen missing packages is
# the expected state of a correct install, not a problem to fix.
DA3_RUNTIME_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "einops": "einops",
    "addict": "addict",
    "omegaconf": "omegaconf",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "imageio": "imageio",
    "tqdm": "tqdm",
}


def check_da3_dependencies() -> Result | None:
    """DA3 is installed with --no-deps, so verify the few deps that matter."""
    if not _installed("depth_anything_3"):
        return None

    missing = [package for module, package in DA3_RUNTIME_MODULES.items() if not _installed(module)]
    if missing:
        return Result(
            "da3 deps",
            "fail",
            f"depth_anything_3 is installed but cannot run: missing {', '.join(missing)}",
            f"pip install {' '.join(missing)}",
        )
    return Result(
        "da3 deps",
        "ok",
        "all runtime deps present (`pip check` also lists open3d, pycolmap, "
        "moviepy, evo, fastapi ... — unused here)",
    )


# The packages whose version the pipeline is actually sensitive to. A stale pin
# on anything else is somebody else's business.
PINNED = ("torch", "torchvision", "numpy", "transformers", "accelerate", "tokenizers")


def check_conflicting_pins() -> list[Result]:
    """Find installed distributions that demand a different torch than this one.

    This is the failure that arrives as a wall of pip text and then gets
    ignored: another project sharing the environment pins torch==2.12 and
    accelerate==1.0.1, pip installs them over ours, and the semantic backend
    breaks somewhere unrelated. Reported per offending distribution, because
    the fix is to remove that distribution rather than to chase each line.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:  # packaging ships with pip, but do not insist
        return []

    installed = {
        name.lower().replace("_", "-"): distribution.version
        for name, distribution in (
            (dist.metadata["Name"] or "", dist) for dist in importlib.metadata.distributions()
        )
        if name
    }

    conflicts: dict[str, list[str]] = {}
    for distribution in importlib.metadata.distributions():
        source = (distribution.metadata["Name"] or "").lower()
        # DA3's own pins are known to be wrong for this environment and are the
        # reason it is installed with --no-deps; check_da3_dependencies covers
        # what it really needs.
        if not source or source == "depth-anything-3":
            continue
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            # Requirements gated on an extra are only real if that extra was
            # asked for, and nothing here installs extras.
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            wanted = requirement.name.lower().replace("_", "-")
            if wanted not in PINNED:
                continue
            have = installed.get(wanted)
            if have and not requirement.specifier.contains(have, prereleases=True):
                conflicts.setdefault(source, []).append(f"{wanted}{requirement.specifier} (have {have})")

    results = []
    for source in sorted(conflicts):
        results.append(
            Result(
                "conflict",
                "warn",
                f"{source} wants {', '.join(sorted(conflicts[source]))}",
                f"nothing here needs {source}; in a venv for this pipeline alone: "
                f"pip uninstall -y {source}",
            )
        )
    return results


def main() -> int:
    results: list[Result] = [check_python(), check_package(), check_ffmpeg()]
    results += check_torch()
    audio = check_torchaudio()
    if audio:
        results.append(audio)
    results += check_backends()
    da3 = check_da3_dependencies()
    if da3:
        results.append(da3)
    results += check_conflicting_pins()
    results += check_weights()
    results += check_data()
    memory = check_memory()
    if memory:
        results.append(memory)

    mark = {"ok": "  ok  ", "warn": " warn ", "fail": " FAIL "}
    print()
    for result in results:
        print(f"[{mark[result.state]}] {result.name:18s} {result.detail}")
        if result.fix and result.state != "ok":
            print(f"{'':10s} -> {result.fix}")
    print()

    failures = [r for r in results if r.state == "fail"]
    warnings_ = [r for r in results if r.state == "warn"]
    if failures:
        print(f"{len(failures)} blocker(s): {', '.join(r.name for r in failures)}")
        print("Fix these, then re-run this script. Nothing will launch until they pass.")
        return 1
    if warnings_:
        print(f"no blockers; {len(warnings_)} thing(s) worth reading above.")
    else:
        print("everything checks out.")
    print(f"\nNext:  WORKERS_PER_GPU={WORKERS_PER_GPU} scripts/run_scenes.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
