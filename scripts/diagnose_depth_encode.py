#!/usr/bin/env python3
"""Find out which side of the depth round trip a node is breaking.

`test_depth_codes_survive_the_gray_encode_bit_exactly` asserts that all 256
depth codes survive `depth.mp4`. When it fails there are two possibilities with
very different consequences, and the test alone does not say which:

  the **encode** is lossy   the delivered files are corrupt, and every run on
                            this node is scrap. Depth is a number pretending to
                            be a pixel; a range squeeze from 0..255 into 16..235
                            is invisible per frame and wrong everywhere.

  the **decode** is lossy   the files are fine and only this node's reader is
                            wrong. Delivery is safe, but preview/validate and
                            anything else that reads depth back here lies.

So this writes one file and reads it back two ways - through ffmpeg itself and
through OpenCV - and reports them separately. ffmpeg is the reference: it wrote
the file, so if it cannot read its own codes back the encode is at fault.

    python scripts/diagnose_depth_encode.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

HEIGHT, WIDTH = 128, 192
FRAMES = 6


def staircase(shift: int = 0) -> np.ndarray:
    """A frame holding every one of the 256 codes, in bands."""
    frame = np.zeros((HEIGHT, WIDTH), np.uint8)
    for code in range(256):
        row0, row1 = (code * HEIGHT) // 256, ((code + 1) * HEIGHT) // 256
        frame[row0:row1, :] = (code + shift) % 256
    return frame


def describe(want: np.ndarray, got: np.ndarray) -> str:
    """Say *how* the codes moved, not just that they did.

    The shape of the damage is the diagnosis: a limited-range squeeze maps 0 to
    16 and 255 to 235 and is linear in between, which is a completely different
    finding from a couple of codes off by one at a block boundary.
    """
    if np.array_equal(want, got):
        return "bit-exact"

    changed = int(np.count_nonzero(want != got))
    total = want.size
    worst = int(np.abs(want.astype(int) - got.astype(int)).max())
    lines = [
        (
            f"CHANGED: {changed}/{total} pixels ({100.0 * changed / total:.1f}%), "
            f"largest error {worst}"
        ),
        f"  input  range {int(want.min())}..{int(want.max())}",
        f"  output range {int(got.min())}..{int(got.max())}",
    ]

    # A range mix-up is a straight line through the codes, so fit one rather
    # than pattern-matching output bounds. That catches both directions, and
    # the two directions mean different things: a gain below 1 is full data
    # written into a limited-range plane (the encode threw the codes away),
    # while a gain above 1 is limited-range maths applied to full-range data
    # (the file is intact and the reader expanded it).
    # Fitted away from the rails: an expansion clips at 0 and 255, and those
    # flat shoulders would drag the line towards gain 1 and hide exactly the
    # case being looked for.
    x = want.astype(np.float64).ravel()
    y = got.astype(np.float64).ravel()
    interior = (y > 1) & (y < 254)
    if interior.sum() >= 64:
        x, y = x[interior], y[interior]
    if np.ptp(x) > 0:
        gain, offset = np.polyfit(x, y, 1)
        residual = float(np.abs(y - (gain * x + offset)).max())
        if residual <= 2.0 and (abs(gain - 1.0) > 0.01 or abs(offset) > 1.0):
            lines.append(f"  ^ the codes were REMAPPED: got = {gain:.4f} * want + {offset:.2f}")
            limited_to_full = 255.0 / (235.0 - 16.0)  # ~1.164
            full_to_limited = 1.0 / limited_to_full  # ~0.859
            if abs(gain - full_to_limited) < 0.02:
                lines.append(
                    "    That gain is a FULL->LIMITED squeeze (0..255 into 16..235).\n"
                    "    Depth codes have been quantised away and cannot be recovered."
                )
            elif abs(gain - limited_to_full) < 0.02:
                lines.append(
                    "    That gain is a LIMITED->FULL expansion: something read this\n"
                    "    full-range file as if it were limited range, and clipped both\n"
                    "    ends. The stored codes are fine; the reader is wrong."
                )
    return "\n".join(lines)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy-extract" / "src"))
    from proxy_extract import proxy

    binary = proxy.ffmpeg_binary()
    version = subprocess.run(
        [binary, "-version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    print("=== the tools ===")
    print(f"ffmpeg binary  {binary}")
    print(f"ffmpeg version {version[0] if version else 'unknown'}")
    try:
        import cv2

        print(f"cv2 version    {cv2.__version__}")
        print(f"cv2 file       {cv2.__file__}")
    except ImportError:
        print("cv2            NOT INSTALLED")
        return 1

    frames = [staircase(shift) for shift in range(FRAMES)]

    with TemporaryDirectory() as scratch:
        path = Path(scratch) / "depth.mp4"
        encoder = proxy.open_encoder(path, WIDTH, HEIGHT, 30.0, kind="depth")
        for frame in frames:
            encoder.write(frame)
        encoder.close()

        print("\n=== what was written ===")
        # The sibling of the ffmpeg that wrote the file, so the report describes
        # the build in use. Only the file name is rewritten, so a path like
        # `.../imageio_ffmpeg/binaries/ffmpeg-v7` keeps its directory. That
        # sibling often does not exist - imageio-ffmpeg ships no ffprobe - so
        # PATH is the fallback, and its absence is not a failure either way:
        # this block is a description, and the round trips below are the test.
        name = Path(binary).name.replace("ffmpeg", "ffprobe")
        candidates = [str(Path(binary).with_name(name)), "ffprobe"]
        probe = None
        for candidate in candidates:
            try:
                probe = subprocess.run(
                    [
                        candidate,
                        "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=pix_fmt,color_range,color_space",
                        "-of", "default=noprint_wrappers=1", str(path),
                    ],
                    capture_output=True, text=True, check=False,
                )
                break
            except (FileNotFoundError, PermissionError):
                continue
        if probe is not None and probe.returncode == 0 and probe.stdout.strip():
            for line in probe.stdout.strip().splitlines():
                print(f"  {line}")
            if "color_range=pc" not in probe.stdout:
                print(
                    "  ^ color_range is not `pc` (full). `gray` through libx264 is only\n"
                    "    lossless when it lands in a FULL-range plane; limited range\n"
                    "    quantises every code."
                )
        else:
            print("  (no ffprobe beside this ffmpeg; skipping)")

        # ---- read back through ffmpeg, which is the reference -------------
        print("\n=== read back by ffmpeg (is the FILE right?) ===")
        raw = subprocess.run(
            [
                binary, "-v", "error", "-i", str(path),
                "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ],
            capture_output=True,
            check=False,
        )
        if raw.returncode != 0:
            print(f"  ffmpeg failed to decode: {raw.stderr.decode(errors='replace')[:400]}")
            return 1
        decoded = np.frombuffer(raw.stdout, np.uint8)
        expected = HEIGHT * WIDTH * FRAMES
        if decoded.size != expected:
            print(f"  decoded {decoded.size} bytes, expected {expected}")
            return 1
        decoded = decoded.reshape(FRAMES, HEIGHT, WIDTH)
        ffmpeg_ok = True
        for index, (want, got) in enumerate(zip(frames, decoded)):
            verdict = describe(want, got)
            if verdict != "bit-exact":
                ffmpeg_ok = False
                print(f"  frame {index}: {verdict}")
                break
        if ffmpeg_ok:
            print("  bit-exact: the file itself carries every code unchanged")

        # ---- read back through OpenCV, which is what the test uses --------
        print("\n=== read back by OpenCV (is this node's READER right?) ===")
        capture = cv2.VideoCapture(str(path))
        read = []
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                read.append(bgr[:, :, 0])
        finally:
            capture.release()
        cv2_ok = len(read) == FRAMES
        if not cv2_ok:
            print(f"  read {len(read)} frames, expected {FRAMES}")
        for index, (want, got) in enumerate(zip(frames, read)):
            verdict = describe(want, got)
            if verdict != "bit-exact":
                cv2_ok = False
                print(f"  frame {index}: {verdict}")
                break
        if cv2_ok:
            print("  bit-exact")

    print("\n=== verdict ===")
    if ffmpeg_ok and cv2_ok:
        print("Both sides are clean. The failing test is not reproducing here.")
        return 0
    if not ffmpeg_ok:
        print(
            "THE ENCODE IS LOSSY. Do not run a delivery on this node: depth.mp4\n"
            "would carry rescaled codes, which no downstream check can detect.\n"
            "Report the `what was written` block above."
        )
        return 1
    print(
        "The FILE is correct and only this node's OpenCV misreads it.\n"
        "Delivery is therefore safe, but anything that reads depth back here\n"
        "(preview, validate, the QC in this suite) is unreliable. Report the\n"
        "cv2 version above."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
