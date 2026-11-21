import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
READY = ROOT / "ready"
MODELS = ROOT / "models"
VENV_BIN = Path(sys.executable).parent
LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")


def _find_exe(name, patterns):
    env = os.environ.get(f"CLIPPER_{name.upper().replace('-', '_')}")
    if env and Path(env).exists():
        return env
    p = shutil.which(name)
    if p:
        return p
    for pat in patterns:
        hits = sorted(glob.glob(os.path.expandvars(pat)))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"{name} not found; install it or add it to PATH")


# winget installs land here; PATH is often not refreshed in the shell that ran the installer
FFMPEG = _find_exe("ffmpeg", [LOCALAPPDATA + r"\Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin\ffmpeg.exe",
                              r"C:\ffmpeg\bin\ffmpeg.exe", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"])
_probe = Path(FFMPEG).with_name("ffprobe.exe" if FFMPEG.lower().endswith(".exe") else "ffprobe")
FFPROBE = str(_probe) if _probe.exists() else _find_exe("ffprobe", [])
_ytd = [VENV_BIN / "yt-dlp.exe", VENV_BIN / "yt-dlp"]
YTDLP = str(next((p for p in _ytd if p.exists()), None) or _find_exe("yt-dlp", []))

# make ffmpeg + deno (yt-dlp's JS challenge runtime) visible to child processes
_extra = [str(Path(FFMPEG).parent), LOCALAPPDATA + r"\Microsoft\WinGet\Links",
          str(Path.home() / ".deno" / "bin")]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["PATH"] = os.pathsep.join(_extra + [os.environ.get("PATH", "")])


def ytdlp():
    """yt-dlp + optional login cookies (needed for age-restricted videos, e.g. many GTA streams).
    Put a Netscape cookies.txt in the project root, or set "cookies_from_browser": "firefox" in config.json."""
    args = [YTDLP]
    if (ROOT / "cookies.txt").exists():
        args += ["--cookies", str(ROOT / "cookies.txt")]
    else:
        cfg = jload(ROOT / "config.json", {}) or {}
        if cfg.get("cookies_from_browser"):
            args += ["--cookies-from-browser", cfg["cookies_from_browser"]]
    return args


_LOGFILE = None


def set_logfile(path):
    """Also append every log line to this file (runs/<id>/log.txt), so progress survives a stuck terminal."""
    global _LOGFILE
    _LOGFILE = Path(path)


def log(*a):
    line = " ".join([time.strftime("[%H:%M:%S]"), *map(str, a)])
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)
    if _LOGFILE:
        try:
            with open(_LOGFILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def run(cmd, cwd=None, timeout=None, check=True, quiet=False):
    """Run a command, return CompletedProcess (text). Raises with the stderr tail on failure."""
    cmd = [str(c) for c in cmd]
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        if check:
            raise
        log(f"  (timed out after {timeout}s: {Path(cmd[0]).name}; continuing without it)")
        return subprocess.CompletedProcess(cmd, -9, stdout="", stderr="timeout")
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-3000:]
        raise RuntimeError(f"command failed ({p.returncode}): {Path(cmd[0]).name} ...\n{tail}")
    return p


def jload(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def jsave(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def hms(sec):
    sec = max(0, int(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def probe_duration(path):
    p = run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path])
    return float(p.stdout.strip() or 0)


def probe_video(path):
    p = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate", "-of", "json", path])
    s = json.loads(p.stdout)["streams"][0]
    num, den = s.get("r_frame_rate", "30/1").split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den or 1)


def is_url(s):
    return str(s).startswith(("http://", "https://"))
