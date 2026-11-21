"""clip doctor: check every piece the pipeline needs and say plainly what is missing."""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def doctor():
    ok_all = True

    def line(ok, what, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok) or ok is None
        mark = "OK  " if ok else ("WARN" if ok is None else "FAIL")
        print(f"[{mark}] {what}{(': ' + detail) if detail else ''}")

    v = sys.version_info
    line(v >= (3, 10), "python", f"{v.major}.{v.minor}.{v.micro} at {sys.executable}")
    try:
        from . import util
        out = subprocess.run([util.FFMPEG, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        line(True, "ffmpeg", out[:60])
        line(Path(util.FFPROBE).exists() or shutil.which("ffprobe"), "ffprobe", util.FFPROBE)
        yv = subprocess.run([util.YTDLP, "--version"], capture_output=True, text=True).stdout.strip()
        line(bool(yv), "yt-dlp", yv)
    except Exception as e:  # noqa: BLE001
        line(False, "ffmpeg / yt-dlp", f"{e} -> run setup again")
        util = None
    deno = shutil.which("deno")
    line(bool(deno), "deno (YouTube needs it to unlock video formats)", deno or "not found -> run setup again")
    for mod, name in (("faster_whisper", "faster-whisper"), ("cv2", "opencv"), ("numpy", "numpy"),
                      ("yt_dlp_ejs", "yt-dlp-ejs")):
        try:
            importlib.import_module(mod)
            line(True, name)
        except Exception as e:  # noqa: BLE001
            line(False, name, f"{e} -> run setup again")
    try:
        import cv2
        line(hasattr(cv2, "FaceDetectorYN"), "opencv face detector", cv2.__version__)
    except Exception:  # noqa: BLE001
        pass
    fonts = list((ROOT / "fonts").glob("*.ttf"))
    line(len(fonts) >= 10, "caption fonts", f"{len(fonts)} files")
    line((ROOT / "models" / "face_detection_yunet_2023mar.onnx").exists(), "face model (YuNet)")
    wm = list((ROOT / "models").glob("models--Systran--faster-whisper-*"))
    line(True if wm else None, "speech model (whisper)", ", ".join(p.name.split("--")[-1] for p in wm)
         or "not downloaded yet (downloads on first use, ~500 MB)")
    try:
        from .llm import _claude_bin, backend
        b = backend()
        detail = ("Claude Code CLI " + str(_claude_bin())) if b == "claude" else \
            "the AI agent operating this folder answers the questions (see AGENTS.md)"
        line(True, f"AI backend = {b}", detail)
    except Exception as e:  # noqa: BLE001
        line(False, "AI backend", str(e))
    free = shutil.disk_usage(ROOT).free / 1e9
    line(free > 10 or (None if free > 4 else False), "free disk space", f"{free:.0f} GB (long streams need ~5-10 GB per run)")
    print("\nALL GOOD" if ok_all else "\nSOMETHING IS MISSING: see the FAIL lines above")
    return 0 if ok_all else 1
