"""Review studio: a local web app to screen clips, tweak them, and send the kept ones to be finished.

Pressing Send renders the full-quality finals right here in the studio's own worker (no agent has to be waiting),
so ready/<run>/ fills up on its own. The operating agent only applies the user's note and, in agent mode, writes the
per-platform post text afterwards (clip finalize <run>)."""
import json
import math
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import config
from .config import FONTS, LOOKS
from .llm import backend, cost_summary
from .render import FONTS_DIR, music_tracks, render_clip
from .util import READY, ROOT, RUNS, jload, jsave, log

STYLE_KEYS = ("look", "font", "caption_size", "highlight", "uppercase", "words_per_chunk", "hook_style", "grade",
              "speed", "music", "music_mood", "caption_y", "captions", "pop", "teaser", "teaser_start", "teaser_end")
FRAME_KEYS = ("layout", "cx", "cy", "fw")
TEXT_KEYS = ("hook", "caption", "hashtags", "note")
EDITABLE = {"status", "start", "end", *STYLE_KEYS, *FRAME_KEYS, *TEXT_KEYS}
SIG_KEYS = ("hook", "start", "end", *STYLE_KEYS, *FRAME_KEYS)
MOODS = ("epic", "emotional", "hype", "chill", "lofi", "comedy")
jobs, JOBQ, LOCK = {}, queue.Queue(), threading.Lock()


def visual_sig(clip):
    """Everything that changes the rendered video (caption/hashtags only change the .txt)."""
    return json.dumps({k: clip.get(k) for k in SIG_KEYS}, sort_keys=True)


def run_look(rid):
    return (jload(RUNS / rid / "clips.json") or {}).get("look") or "streamer"


def effective_clips(rid):
    rd = RUNS / rid
    clips = (jload(rd / "clips.json") or {}).get("clips", [])
    dec = jload(rd / "decisions.json", {})
    out = []
    for c in clips:
        e = {**c, **dec.get(str(c["n"]), {})}
        d = rd / "clips" / f"{c['n']:02d}"
        pv = d / "preview.mp4"
        e["preview"] = f"/runs/{rid}/clips/{c['n']:02d}/preview.mp4" if pv.exists() else None
        e["poster"] = f"/runs/{rid}/clips/{c['n']:02d}/poster.jpg" if (d / "poster.jpg").exists() else None
        e["v"] = int(pv.stat().st_mtime) if pv.exists() else 0
        e["preview_sig_ok"] = e.get("preview_sig", visual_sig(c)) == visual_sig(e)
        e["final"] = (d / "final.mp4").exists() and e.get("final_sig") == visual_sig(e)
        e["analysis"] = jload(d / "analysis.json") or {}
        e["rendered"] = jload(d / "preview_meta.json") or {}
        e["job"] = jobs.get(f"{rid}:{c['n']}")
        out.append(e)
    return out


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


def export_ready(rid, clip):
    dst = READY / rid
    dst.mkdir(parents=True, exist_ok=True)
    base = f"{clip['n']:02d}_{slug(clip.get('title') or clip['hook'])}"
    for old in dst.glob(f"{clip['n']:02d}_*"):
        if not old.name.startswith(base):
            old.unlink()
    shutil.copy2(RUNS / rid / "clips" / f"{clip['n']:02d}" / "final.mp4", dst / f"{base}.mp4")
    from .publish import post_copy, post_text
    # never writes copy itself: "clip finalize" makes one batched call for all kept clips (cheaper, and the posts
    # are written side by side); until then the .txt holds the draft caption, after it the cached final text
    copy = post_copy(rid, [clip], allow_llm=False)[clip["n"]]
    (dst / f"{base}.txt").write_text(post_text(copy), encoding="utf-8")
    text = copy["tiktok"]
    man = [m for m in jload(dst / "manifest.json", []) if m["n"] != clip["n"]]
    man.append({"n": clip["n"], "file": f"{base}.mp4", "caption": text, "hook": clip["hook"],
                "tiktok": copy["tiktok"], "instagram": copy["instagram"],
                "youtube_title": copy["youtube_title"], "youtube_description": copy["youtube_description"],
                "duration": round((clip["end"] - clip["start"]) / float(clip.get("speed") or 1), 1)})
    jsave(dst / "manifest.json", sorted(man, key=lambda m: m["n"]))


def set_decision(rid, n, **fields):
    with LOCK:
        dec = jload(RUNS / rid / "decisions.json", {})
        dec.setdefault(str(n), {}).update(fields)
        jsave(RUNS / rid / "decisions.json", dec)


def render_one(rid, n, final, cfg=None):
    cfg = cfg or config.load()
    info = jload(RUNS / rid / "info.json")
    clip = next(c for c in effective_clips(rid) if c["n"] == n)
    if final and clip["final"]:
        export_ready(rid, clip)
        return
    render_clip(RUNS / rid, info, clip, cfg, final=final, layout=clip.get("layout") or "auto", run_look=run_look(rid))
    if final:
        export_ready(rid, clip)
        set_decision(rid, n, final_sig=visual_sig(clip))
    else:
        set_decision(rid, n, preview_sig=visual_sig(clip))


def worker():
    cfg = config.load()
    while True:
        rid, n, final = JOBQ.get()
        key = f"{rid}:{n}"
        jobs[key] = {"state": "rendering", "final": final, "t": time.time()}
        try:
            render_one(rid, n, final, cfg)
            jobs[key] = {"state": "done", "final": final}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            jobs[key] = {"state": "error", "error": str(e)[-600:], "final": final}


def busy(rid, n):
    return (jobs.get(f"{rid}:{n}") or {}).get("state") in ("queued", "rendering")


def enqueue(rid, n, final):
    if busy(rid, n):
        return False
    jobs[f"{rid}:{n}"] = {"state": "queued", "final": final}
    JOBQ.put((rid, n, final))
    return True


def run_payload(rid):
    info = jload(RUNS / rid / "info.json") or {}
    cl = jload(RUNS / rid / "clips.json") or {}
    sub = jload(RUNS / rid / "submission.json")
    return {"id": rid, "title": info.get("title"), "url": info.get("webpage_url") or info.get("input"),
            "channel": info.get("channel"), "duration": info.get("duration"), "look": cl.get("look"),
            "look_reason": cl.get("look_reason"), "clips": effective_clips(rid), "costs": cost_summary(RUNS / rid),
            "ready_dir": str(READY / rid), "submission": sub,
            "looks": {k: {**v} for k, v in LOOKS.items()}, "fonts": FONTS,
            "music": {m: len(music_tracks(m)) for m in MOODS}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or ("font/ttf" if path.suffix == ".ttf" else "application/octet-stream")
        rng = self.headers.get("Range")
        a, b = 0, size - 1
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                a = int(m.group(1))
                b = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
            else:
                a = size - int(m.group(2))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    return
                left -= len(chunk)

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p in ("/", "/index.html"):
            return self._file(Path(__file__).parent / "review.html")
        if p == "/api/ping":
            return self._json({"ok": True, "root": str(ROOT), "pid": os.getpid()})
        if p == "/api/runs":
            runs = []
            for d in sorted(RUNS.glob("*"), key=lambda x: -x.stat().st_mtime):
                info = jload(d / "info.json")
                if info and (d / "clips.json").exists():
                    runs.append({"id": d.name, "title": info.get("title"), "channel": info.get("channel")})
            return self._json(runs)
        if m := re.match(r"^/api/run/([\w.-]+)$", p):
            return self._json(run_payload(m.group(1)))
        if p.startswith("/fonts/"):
            f = (FONTS_DIR / p[len("/fonts/"):]).resolve()
            if FONTS_DIR.resolve() in f.parents:
                return self._file(f)
        if p.startswith("/runs/"):
            f = (ROOT / p.lstrip("/")).resolve()
            if RUNS.resolve() in f.parents:
                return self._file(f)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        p = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if m := re.match(r"^/api/run/([\w.-]+)/clip/(\d+)$", p):
            rid, n = m.group(1), m.group(2)
            for k in ("start", "end", "cx", "cy", "fw", "caption_y", "speed", "caption_size", "teaser_start", "teaser_end"):
                if k in body and body[k] is not None and not (isinstance(body[k], (int, float)) and math.isfinite(body[k])):
                    body.pop(k)
            if "start" in body or "end" in body:
                clip = next((c for c in effective_clips(rid) if str(c["n"]) == n), None)
                if clip:
                    a, b = body.get("start", clip["start"]), body.get("end", clip["end"])
                    if b - a < 3:
                        return self._json({"error": f"A clip must be at least 3 seconds long (this one is {b - a:.1f}s)."}, 400)
            set_decision(rid, n, **{k: v for k, v in body.items() if k in EDITABLE})
            return self._json({"ok": True})
        if m := re.match(r"^/api/run/([\w.-]+)/apply_all$", p):
            rid = m.group(1)
            fields = {k: v for k, v in body.items() if k in STYLE_KEYS or k in ("layout",)}
            for c in effective_clips(rid):
                set_decision(rid, c["n"], **fields)
            return self._json({"ok": True, "applied": list(fields)})
        if m := re.match(r"^/api/run/([\w.-]+)/clip/(\d+)/rerender$", p):
            return self._json({"ok": True, "queued": enqueue(m.group(1), int(m.group(2)), False)})
        if m := re.match(r"^/api/run/([\w.-]+)/rerender_stale$", p):
            rid = m.group(1)
            q = [c["n"] for c in effective_clips(rid) if not c["preview_sig_ok"] and c.get("status") != "rejected"
                 and enqueue(rid, c["n"], False)]
            return self._json({"ok": True, "queued": q})
        if m := re.match(r"^/api/run/([\w.-]+)/submit$", p):
            rid = m.group(1)
            kept = [c["n"] for c in effective_clips(rid) if c.get("status") == "approved"]
            if not kept:
                return self._json({"error": "Keep at least one clip before sending."}, 400)
            jsave(RUNS / rid / "submission.json", {"kept": kept, "note": (body.get("note") or "").strip(),
                                                    "at": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "sent"})
            for n in kept:      # start the full-quality renders now; finished ones are skipped automatically
                enqueue(rid, n, True)
            return self._json({"ok": True, "kept": kept})
        if m := re.match(r"^/api/run/([\w.-]+)/queue_finals$", p):
            rid = m.group(1)
            q = [c["n"] for c in effective_clips(rid) if c.get("status") == "approved" and not c["final"]
                 and enqueue(rid, c["n"], True)]
            busy_ = [c["n"] for c in effective_clips(rid) if busy(rid, c["n"])]
            return self._json({"ok": True, "queued": q, "busy": busy_})
        if m := re.match(r"^/api/run/([\w.-]+)/open$", p):
            d = READY / m.group(1)
            d.mkdir(parents=True, exist_ok=True)
            open_path(d)
            return self._json({"ok": True})
        if p == "/api/shutdown":
            self._json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return None
        return self._json({"error": "not found"}, 404)


def open_path(p):
    if sys.platform == "win32":
        os.startfile(str(p))
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])


def _ping(port):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=2) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def queue_finals_remote(rid, port=8765):
    """Ask a running studio to (re)queue the kept clips' finals; returns the clip numbers it is working on."""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/run/{rid}/queue_finals", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.loads(r.read())
            return sorted(set(d.get("queued", []) + d.get("busy", [])))
    except Exception:  # noqa: BLE001
        return None


def stop_studio(port=8765):
    import urllib.request
    if not _ping(port):
        return False
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/shutdown", data=b"{}",
                                                      method="POST"), timeout=3)
    except Exception:  # noqa: BLE001
        pass
    for _ in range(20):
        if not _ping(port):
            return True
        time.sleep(0.3)
    return False


def launch_studio(port=8765, open_browser=True, restart=False):
    """Start the studio as a background process and return at once (agents' terminals hang on servers)."""
    url = f"http://127.0.0.1:{port}/"
    if restart:
        stop_studio(port)
    if not _ping(port):
        RUNS.mkdir(exist_ok=True)
        logf = open(RUNS / "studio.log", "a", encoding="utf-8")
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        env = {**os.environ, "PYTHONUTF8": "1"}
        subprocess.Popen([sys.executable, "-m", "clipper", "review", "--no-open", "--port", str(port)], cwd=str(ROOT),
                         stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags,
                         env=env, start_new_session=sys.platform != "win32")
        for _ in range(60):
            if _ping(port):
                break
            time.sleep(0.5)
    ok = bool(_ping(port))
    log(f"studio {'running' if ok else 'FAILED to start (see runs/studio.log)'}: {url}")
    if ok and open_browser:
        import webbrowser
        webbrowser.open(url)
    return ok


def serve(port=8765, open_browser=True):
    threading.Thread(target=worker, daemon=True).start()
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    except OSError:
        log(f"port {port} is busy: the studio is probably already running at http://127.0.0.1:{port}/")
        return
    url = f"http://127.0.0.1:{port}/"
    log(f"review studio: {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
