"""Stage 5: cut, reframe to 9:16, styled captions + hook, grade, speed, music, encode."""
import glob
import random
import re
import statistics
import subprocess
from pathlib import Path

import numpy as np

from . import config as C
from .analyze import analyze_clip
from .transcript import whisper_words
from .util import FFMPEG, MODELS, ROOT, jload, jsave, log, probe_duration, probe_video, run, ytdlp

PAD = 2.0
W, H = 1080, 1920
SAFE_TOP, SAFE_BOTTOM = 240, 1420      # TikTok safe zone: top bar ~240px, bottom caption/UI ~500px
FONTS_DIR = ROOT / "fonts"
MUSIC_DIR = ROOT / "music"


def clip_dir(run_dir, n):
    d = Path(run_dir) / "clips" / f"{n:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_section(run_dir, info, clip, d):
    """Return (path, offset): offset = source time at t=0 of the returned file."""
    if info["source"] == "file":
        return info["local_path"], 0.0
    full = sorted(glob.glob(str(Path(run_dir) / "video_full.*")))
    full = [f for f in full if not f.endswith((".part", ".ytdl")) and ".f" not in Path(f).suffixes[0]]
    if not full and (info.get("duration") or 0) <= 5400:
        # videos up to 90 min: one full download is faster than many throttled section downloads,
        # and makes every later trim instant and frame-accurate
        log("  downloading the full video once (<=1080p)")
        run([*ytdlp(), "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
             "--merge-output-format", "mp4", "--no-warnings", "--no-playlist", "-N", "4",
             "-o", str(Path(run_dir) / "video_full.%(ext)s"), info["webpage_url"]], timeout=5400)
        full = sorted(glob.glob(str(Path(run_dir) / "video_full.mp4")))
    if full:
        return full[0], 0.0
    a, b = max(0.0, clip["start"] - PAD - 8), clip["end"] + PAD + 8   # slack for edits in review
    meta = jload(d / "src.json")
    if meta and meta["a"] <= clip["start"] - PAD and meta["b"] >= clip["end"] + PAD and Path(meta["path"]).exists():
        return meta["path"], meta["a"]
    for f in d.glob("src.*"):
        f.unlink()
    log(f"  download section {a:.0f}-{b:.0f}s")
    for attempt in range(3):   # YouTube occasionally 403s a signed URL; a fresh extraction fixes it
        try:
            run([*ytdlp(), "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
                 "--download-sections", f"*{a:.2f}-{b:.2f}", "--force-keyframes-at-cuts", "--merge-output-format",
                 "mp4", "--no-warnings", "--no-playlist", "-o", str(d / "src.%(ext)s"), info["webpage_url"]],
                timeout=1800)
            break
        except RuntimeError:
            if attempt == 2:
                raise
            log(f"  download failed, retrying ({attempt + 1}/2)")
            for f in d.glob("src.*"):
                f.unlink()
    path = str(next(f for f in d.glob("src.*") if f.suffix in (".mp4", ".mkv", ".webm")))
    jsave(d / "src.json", {"a": a, "b": b, "path": path})
    return path, a


HQ_FMT = "bv*[height>1080][height<=2160]"
SR = 4000


def _pcm(path, s, d):
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{max(0.0, s):.3f}", "-t", f"{d:.3f}", "-i", str(path), "-vn",
                        "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float64)


def _vsig(path, s, d, fps=60):
    """Motion signature: mean frame-to-frame change of a tiny grey copy, one value per frame."""
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{max(0.0, s):.3f}", "-t", f"{d:.3f}", "-i", str(path), "-an",
                        "-vf", f"fps={fps},scale=48:27:flags=area,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    f = np.frombuffer(p.stdout, np.uint8)
    f = f[: len(f) // (48 * 27) * 48 * 27].reshape(-1, 48 * 27).astype(np.float64)
    return np.abs(np.diff(f, axis=0)).mean(axis=1) if len(f) > 1 else np.zeros(0)


def _locate(hay, needle):
    """Position (samples) and normalised correlation of needle inside hay, via FFT."""
    n, m = len(hay), len(needle)
    if m == 0 or n < m:
        return None, 0.0
    needle = needle - needle.mean()
    size = 1 << int(np.ceil(np.log2(n + m)))
    c = np.fft.irfft(np.fft.rfft(hay, size) * np.conj(np.fft.rfft(needle, size)), size)[: n - m + 1]
    cs, cs2 = np.concatenate([[0.0], np.cumsum(hay)]), np.concatenate([[0.0], np.cumsum(hay * hay)])
    win, win2 = cs[m:] - cs[:-m], cs2[m:] - cs2[:-m]
    score = c / (np.sqrt(np.maximum(win2 - win * win / m, 1e-6)) * (np.linalg.norm(needle) + 1e-6))
    k = int(np.argmax(score))
    return k, float(score[k])


def hq_section(run_dir, info, clip, d, src, offset):
    """Finals only: the picture of the same stretch in the best resolution the source has (stream VODs often go to
    1440p/4K), so a small webcam crop is upscaled ~1.4x instead of ~3x. The sound keeps coming from the normal source.
    Returns (video_path, video_offset, scale) or None; video_offset maps "-ss t" on that file to source time."""
    if info.get("source") == "file" or not info.get("webpage_url"):
        return None
    meta = jload(d / "hq.json") or {}
    if meta.get("none"):
        return None
    if (meta.get("path") and meta.get("v") == 2 and meta["a"] <= clip["start"] - PAD and meta["b"] >= clip["end"] + PAD
            and Path(meta["path"]).exists()):
        return meta["path"], meta["offset"], meta["scale"]
    for f in d.glob("hq*.*"):
        f.unlink()
    a, b = max(0.0, clip["start"] - PAD - 3), clip["end"] + PAD + 2
    log(f"  fetching a sharper picture (up to 4K) for the final, {a:.0f}-{b:.0f}s")
    p, unavailable = None, False
    for attempt in range(3):
        # video only, stream-copied (re-encoding 4K at the cut runs at ~3 fps); we line it up by its frames below
        p = run([*ytdlp(), "-f", HQ_FMT, "--download-sections", f"*{a:.2f}-{b:.2f}", "--no-warnings", "--no-playlist",
                 "-o", str(d / "hq.%(ext)s"), info["webpage_url"]], timeout=1800, check=False)
        unavailable = "Requested format is not available" in (p.stderr or "") + (p.stdout or "")
        if p.returncode == 0 or unavailable:
            break
        for f in d.glob("hq.*"):
            f.unlink()
    files = [f for f in d.glob("hq.*") if f.suffix in (".mkv", ".mp4", ".webm")]
    if unavailable:
        jsave(d / "hq.json", {"none": True})
        log("  source has nothing above 1080p")
        return None
    if p is None or p.returncode != 0 or not files:
        log("  sharper picture unavailable, using the normal source")
        return None
    hq = str(files[0])
    sw, sh, _ = probe_video(src)
    hw, hh, _ = probe_video(hq)
    if not hh or hh <= sh:
        return None
    # measure exactly what the renderer will do ("-ss t -i hq"): which source moment appears at a given seek
    hay = _vsig(src, 0, probe_duration(src) or (b - a + 20))
    vdur = probe_duration(hq) or (b - a)

    def where(t):
        k, s = _locate(hay, _vsig(hq, t, 4.0))
        return (None, s) if k is None else (offset + k / 60 - t, s)

    o1, s1 = where(min(max(0.5, clip["start"] - a), max(0.5, vdur - 6)))
    o2, s2 = where(min(max(0.5, clip["end"] - a - 6), max(0.5, vdur - 6))) if o1 is not None else (None, 0)
    if o1 is None or o2 is None or min(s1, s2) < 0.6 or abs(o1 - o2) > 0.05:
        log(f"  could not line up the sharper picture ({s1:.2f}/{s2:.2f}), using the normal source")
        return None
    voff = (o1 + o2) / 2
    meta = {"v": 2, "path": hq, "offset": voff, "a": voff, "b": voff + vdur, "scale": hw / sw,
            "res": f"{hw}x{hh}", "match": [round(s1, 3), round(s2, 3)], "drift": round(o1 - o2, 3)}
    jsave(d / "hq.json", meta)
    log(f"  sharper picture {hw}x{hh} lined up (match {min(s1, s2):.2f})")
    return hq, voff, hw / sw


def clip_words(d, src, offset, clip, model):
    cached = jload(d / "words.json")
    key = [clip["start"], clip["end"]]
    if cached and cached["key"] == key:
        return cached["words"]
    log("  whisper word timings")
    t0 = clip["start"] - offset
    words = whisper_words(src, model, max(0, t0 - 0.3), clip["end"] - clip["start"] + 0.6)
    rel = [{**w, "start": round(w["start"] - t0, 2), "end": round(w["end"] - t0, 2)} for w in words]
    rel = [w for w in rel if w["end"] > 0 and w["start"] < clip["end"] - clip["start"]]
    jsave(d / "words.json", {"key": key, "words": rel})
    return rel


def detect_layout(src, t0, dur):
    import cv2
    cap = cv2.VideoCapture(src)
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    yunet_path = MODELS / "face_detection_yunet_2023mar.onnx"
    yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.75) if yunet_path.exists() else None
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    hits, all_faces, n_frames = [], [], 0
    for k in range(12):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + dur * (k + 0.5) / 12) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        n_frames += 1
        sc = 1280 / fr.shape[1]
        small = cv2.resize(fr, None, fx=sc, fy=sc)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if yunet is not None:
            yunet.setInputSize((small.shape[1], small.shape[0]))
            _, det = yunet.detect(small)
            faces = [] if det is None else [tuple(int(v) for v in f[:4]) for f in det if f[2] >= 12]
        else:
            faces = casc.detectMultiScale(g, 1.1, 6, minSize=(22, 22))
        gw, gh = g.shape[1], g.shape[0]
        norm = [((x + w / 2) / gw, (y + h / 2) / gh, w / gw, h / gh) for x, y, w, h in faces]
        all_faces.append(norm)
        if norm:
            hits.append(max(norm, key=lambda f: f[2] * f[3]))
    cap.release()
    # facecam = a small face that stays in the same spot across frames (game NPC faces move around)
    pts = [f for fr_ in all_faces for f in fr_ if f[2] < 0.12]
    best = None
    for p in pts:
        near = [q for q in pts if abs(q[0] - p[0]) < 0.04 and abs(q[1] - p[1]) < 0.06]
        frames_with = sum(1 for fr_ in all_faces if any(abs(q[0] - p[0]) < 0.04 and abs(q[1] - p[1]) < 0.06 for q in fr_))
        if frames_with >= max(4, 0.4 * n_frames) and (best is None or frames_with > best[0]):
            best = (frames_with, near)
    if best and not (vw and vh and vw / vh < 0.8):
        near = best[1]
        cx, cy = statistics.median(q[0] for q in near), statistics.median(q[1] for q in near)
        fw = statistics.median(q[2] for q in near)
        big = [h for h in hits if h[2] >= 0.12]
        if len(big) < 0.5 * n_frames and (cx < 0.38 or cx > 0.62 or cy < 0.3 or cy > 0.7):
            return {"layout": "facecam", "cx": cx, "cy": cy, "fw": fw}
    if vw and vh and vw / vh < 0.8:
        return {"layout": "native"}   # already vertical
    if len(hits) < 4:
        return {"layout": "blur"}
    cx = statistics.median(h[0] for h in hits)
    cy = statistics.median(h[1] for h in hits)
    fw = statistics.median(h[2] for h in hits)
    spread = statistics.pstdev([h[0] for h in hits])
    edge = cx < 0.34 or cx > 0.66 or cy < 0.3 or cy > 0.7
    if fw < 0.14 and edge and spread < 0.05:
        return {"layout": "facecam", "cx": cx, "cy": cy, "fw": fw}
    if fw >= 0.07:
        return {"layout": "crop", "cx": cx}
    return {"layout": "blur"}


def _even(x):
    return int(x) // 2 * 2


def consensus_cam_rect(run_dir, own, lay, vw, vh):
    """Streamers keep their webcam in one place: use the rect most clips of this run agree on
    when this clip's own detection failed or disagrees."""
    import statistics as stt
    rects = [r for r in ((jload(p) or {}).get("cam_rect") for p in Path(run_dir).glob("clips/*/detected.json")) if r]
    best = []
    for r in rects:
        grp = [q for q in rects if all(abs(q[i] - r[i]) <= 16 for i in range(4))]
        if len(grp) > len(best):
            best = grp
    if len(best) < 2:
        return own
    cons = [int(stt.median(q[i] for q in best)) for i in range(4)]
    fx, fy = lay["cx"] * vw, lay["cy"] * vh
    face_inside = cons[0] <= fx <= cons[0] + cons[2] and cons[1] <= fy <= cons[1] + cons[3]
    if own and all(abs(own[i] - cons[i]) <= 16 for i in range(4)):
        return own
    return cons if face_inside else own


def detect_cam_rect(src, seek, dur, lay):
    """Find the exact webcam overlay rectangle (full-frame pixels) from its static border edges.

    Gameplay moves, the overlay frame doesn't: edges that persist across many frames outline the cam box,
    so the crop can exclude chat, alerts and other overlay clutter around it."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(src)
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sc = 960 / vw
    acc_x = acc_y = None
    n = 0
    for k in range(20):
        cap.set(cv2.CAP_PROP_POS_MSEC, (seek + dur * (k + 0.5) / 20) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(fr, None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ex = (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) > 40).astype(np.float32)   # vertical edges
        ey = (np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) > 40).astype(np.float32)   # horizontal edges
        acc_x = ex if acc_x is None else acc_x + ex
        acc_y = ey if acc_y is None else acc_y + ey
        n += 1
    cap.release()
    if n < 6:
        return None
    px, py = acc_x / n, acc_y / n                     # edge persistence 0..1
    H, W = py.shape
    cx, cy, fw = lay["cx"] * W, lay["cy"] * H, max(8, lay["fw"] * W)
    x0, x1 = int(max(0, cx - 2.5 * fw)), int(min(W, cx + 2.5 * fw))
    rows = py[:, x0:x1].mean(1)
    up = rows[int(max(0, cy - 7 * fw)):int(max(1, cy - 1.2 * fw))]
    dn = rows[int(min(H - 1, cy + 1.2 * fw)):int(min(H, cy + 7 * fw))]
    top = int(max(0, cy - 7 * fw)) + int(np.argmax(up)) if up.size and up.max() > 0.5 else 0
    bot = int(min(H - 1, cy + 1.2 * fw)) + int(np.argmax(dn)) if dn.size and dn.max() > 0.5 else None
    if bot is None:
        return None
    cols = px[top:bot, :].mean(0)
    lft = cols[int(max(0, cx - 9 * fw)):int(max(1, cx - 1.2 * fw))]
    rgt = cols[int(min(W - 1, cx + 1.2 * fw)):int(min(W, cx + 9 * fw))]
    left = int(max(0, cx - 9 * fw)) + int(np.argmax(lft)) if lft.size and lft.max() > 0.5 else 0
    right = int(min(W - 1, cx + 1.2 * fw)) + int(np.argmax(rgt)) if rgt.size and rgt.max() > 0.5 else W
    w, h = right - left, bot - top
    if w < 3 * fw or h < 2.5 * fw or not (0.9 <= w / max(h, 1) <= 2.4):
        return None
    inset = 3   # stay inside the border line
    return [round((left + inset) / sc), round((top + inset) / sc), round((w - 2 * inset) / sc), round((h - 2 * inset) / sc)]


def detect_bars(src, seek, dur):
    """Find baked-in black bars (cinematic letterboxing) so they don't end up inside the vertical frame."""
    p = run([FFMPEG, "-v", "info", "-ss", f"{seek + dur * 0.2:.2f}", "-t", f"{min(12, dur * 0.6):.2f}", "-i", src,
             "-vf", "fps=2,cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"], check=False)
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", p.stderr or "")
    if not found:
        return None
    w, h, x, y = map(int, found[-1])
    return [w, h, x, y]


def filtergraph(lay, vw, vh, vin="[v0]"):
    """Return (graph producing [base], text anchors). Anchors: cap_y, hook_y, hook_an (ASS alignment)."""
    L = lay["layout"]
    if L == "native":
        return f"{vin}scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}[base]", {}
    if L == "crop":
        cw = _even(min(vw, vh * 9 / 16))
        x = _even(min(max(0, lay["cx"] * vw - cw / 2), vw - cw))
        return f"{vin}crop={cw}:{_even(vh)}:{x}:0,scale={W}:{H}:flags=lanczos,cas=0.35[base]", {}
    if L == "facecam":
        rect = lay.get("cam_rect")
        if rect:
            fx, fy, fw, fh = rect
            top = _even(min(820, max(520, W * fh / fw)))     # cam keeps its own aspect, butts onto the game
        else:
            top = 720
            fw = min(vw, lay["fw"] * vw * 4.5)
            fh = fw / (W / top)
            fx = min(max(0, lay["cx"] * vw - fw / 2), vw - fw)
            fy = min(max(0, lay["cy"] * vh - fh * 0.45), vh - fh)
        gh_out = H - top
        gw = min(vw, vh * W / gh_out)
        gh = gw * gh_out / W
        gx, gy = (vw - gw) / 2, (vh - gh) / 2
        g = (f"{vin}split=2[a][b];[a]crop={_even(fw)}:{_even(fh)}:{_even(fx)}:{_even(fy)},hqdn3d=1.5:1.5:3:3,"
             f"scale={W}:{top}:flags=lanczos,cas=0.45[cam];"
             f"[b]crop={_even(gw)}:{_even(gh)}:{_even(gx)}:{_even(gy)},scale={W}:{gh_out}:flags=lanczos[game];"
             f"[cam][game]vstack[base]")
        return g, {"cap_y": top + 520, "hook_y": top, "hook_an": 5}
    zoom = 1.35 if L == "letterbox" else 1.0
    fg_w = _even(W * zoom)
    fg_h = _even(fg_w * vh / vw)
    crop = f",crop={W}:{fg_h}" if zoom > 1 else ""
    if L == "letterbox":   # cinematic: black canvas, slightly zoomed 16:9 band
        bg = f"color=c=black:s={W}x{H}[bg];"
        g = f"{bg}{vin}scale={fg_w}:{fg_h}{crop}[fg];[bg][fg]overlay=0:(H-h)/2:shortest=1[base]"
    else:                  # blur: full frame centred on a blurred fill
        g = (f"{vin}split=2[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
             f"boxblur=24:2,eq=brightness=-0.10[bg];[b]scale={W}:{fg_h}[fg];[bg][fg]overlay=0:(H-h)/2[base]")
    return g, {"cap_y": min(SAFE_BOTTOM - 70, H // 2 + fg_h // 2 + 130), "hook_y": H // 2 - fg_h // 2 - 70, "hook_an": 2}


GRADES = {
    "none": "",
    "cinematic": "eq=contrast=1.10:saturation=0.80:brightness=-0.03,vignette=angle=PI/4.6,noise=alls=7:allf=t+u",
    "punchy": "eq=contrast=1.12:saturation=1.25,unsharp=5:5:0.6",
    "warm": "colorbalance=rs=0.06:gs=0.02:bs=-0.06,eq=saturation=1.08",
}

# ---------- captions ----------

def _t(s):
    s = max(0, s)
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


def _esc(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def _wrap(text, width=20):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return r"\N".join(lines)


SWEARS = re.compile(r"\b(fuck\w*|motherfuck\w*|shit\w*|bitch\w*|nigg\w*|cunt\w*|dick\w*|pussy\w*|faggot\w*|retard\w*)\b", re.I)


def censor(s):
    return SWEARS.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 2) + m.group(0)[-1], s)


def write_ass(path, words, hook, dur, st, anchors, captions_on=True, speed=1.0):
    cap_y = int(st.get("caption_y_px") or anchors.get("cap_y") or st["caption_y"] * H)
    cap_y = max(SAFE_TOP + 200, min(SAFE_BOTTOM - 60, cap_y))
    cz = censor if st.get("censor", True) else (lambda s: s)
    up = (lambda s: cz(s.upper())) if st["uppercase"] else cz
    serif = st["hook_style"] == "serif"
    hook_box = (f"Style: Hook,{st['hook_font']},{st['hook_size']},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,"
                f"-1,0,0,0,100,100,0,0,3,16,0,8,80,80,{SAFE_TOP + 20},1")
    hook_serif = (f"Style: Hook,{st['hook_font']},{st['hook_size']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,"
                  f"0,0,0,0,100,100,0,0,1,3,4,8,90,90,{SAFE_TOP + 20},1")
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['caption_size']},{st['primary']},{st['primary']},&H00000000,&H90000000,0,0,0,0,100,100,1,0,1,{st['outline']},{st['shadow']},5,70,140,0,1
{hook_serif if serif else hook_box}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if hook and st["hook_style"] != "none":
        hend = dur if not st.get("hook_seconds") else min(dur, st["hook_seconds"])
        pos = ""
        if anchors.get("hook_y"):
            hy = max(SAFE_TOP + 40, anchors["hook_y"])
            pos = f"{{\\an{anchors.get('hook_an', 2)}\\pos(540,{hy})}}"
        txt = hook if serif else hook
        ev.append(f"Dialogue: 1,{_t(0)},{_t(hend)},Hook,,0,0,0,,{pos}{{\\fad(250,0)}}{_wrap(_esc(txt), 22 if serif else 20)}")
    if captions_on:
        ws = [{**w, "start": w["start"] / speed, "end": w["end"] / speed} for w in words]
        chunks, cur = [], []
        for i, w in enumerate(ws):
            cur.append(w)
            nxt = ws[i + 1] if i + 1 < len(ws) else None
            if (nxt is None or len(cur) >= st["words_per_chunk"] or nxt["start"] - w["end"] > 0.6
                    or w["text"].endswith((".", "?", "!", ","))):
                chunks.append(cur)
                cur = []
        hl = st["highlight"]
        pop = "\\t(0,70,\\fscx116\\fscy116)\\t(70,150,\\fscx106\\fscy106)" if st.get("pop") else "\\fscx106\\fscy106"
        for ci, ch in enumerate(chunks):
            c_end = ch[-1]["end"] + 0.15
            if ci + 1 < len(chunks):
                c_end = min(c_end, chunks[ci + 1][0]["start"])
            for wi, w in enumerate(ch):
                a = w["start"]
                b = ch[wi + 1]["start"] if wi + 1 < len(ch) else c_end
                if b <= a:
                    continue
                parts = []
                for k, x in enumerate(ch):
                    t = _esc(up(x["text"]))
                    parts.append(f"{{\\c{hl}{pop}}}{t}{{\\r}}" if k == wi else t)
                ev.append(f"Dialogue: 0,{_t(a)},{_t(min(b, dur))},Cap,,0,0,0,,{{\\pos(540,{cap_y})}}" + " ".join(parts))
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


# ---------- music ----------

def music_tracks(mood):
    files = []
    for ext in ("mp3", "m4a", "wav", "ogg", "flac"):
        files += glob.glob(str(MUSIC_DIR / mood / f"*.{ext}"))
    return sorted(files)


def pick_track(mood, n, chosen=None):
    if chosen and Path(chosen).exists():
        return chosen
    tr = music_tracks(mood)
    return tr[(n - 1) % len(tr)] if tr else None


# ---------- render ----------

def auto_teaser(src, seek, dur, words):
    """Loudest ~2s moment in the clip's second part, snapped to word boundaries (relative seconds)."""
    from .analyze import _rms_track
    import numpy as np
    db = _rms_track(src, seek, dur, 0.1)
    if db.size < 60:
        return None
    k = 15
    sm = np.convolve(db, np.ones(k) / k, mode="same")
    lo, hi = int(max(3.0, dur * 0.3) / 0.1), int(max(0, dur - 0.8) / 0.1)
    if hi - lo < 10:
        return None
    c = (lo + int(np.argmax(sm[lo:hi]))) * 0.1
    a, b = c - 1.1, c + 1.1
    ws = [w for w in words if w["end"] > a and w["start"] < b]
    if ws:
        a, b = min(a, ws[0]["start"]), max(b, ws[-1]["end"])
    a, b = max(0.0, a), min(dur, b)
    if b - a > 3.2:
        b = a + 3.2
    return round(a, 2), round(b, 2)


def teaser_window(clip, src, seek, dur, words):
    if (clip.get("teaser") or "on") == "off":
        return None
    ts, te = clip.get("teaser_start"), clip.get("teaser_end")
    if ts is not None and te is not None:
        a, b = float(ts) - clip["start"], float(te) - clip["start"]
        if 0 <= a < b <= dur + 0.01 and 0.8 <= b - a <= 4.5:
            return round(a, 2), round(b, 2)
    return auto_teaser(src, seek, dur, words)


def measure_lufs(src, seek, dur):
    """Integrated loudness of the source segment (EBU R128)."""
    p = run([FFMPEG, "-v", "info", "-nostats", "-ss", f"{seek:.2f}", "-t", f"{dur:.2f}", "-i", src, "-vn",
             "-af", "ebur128=framelog=quiet", "-f", "null", "-"], check=False)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", p.stderr or "")
    return float(m[-1]) if m else None


def effective_style(clip, run_look):
    overrides = {k: clip.get(k) for k in ("font", "caption_size", "highlight", "primary", "uppercase", "words_per_chunk",
                                          "hook_style", "hook_font", "grade", "speed", "music", "music_mood",
                                          "caption_y", "pop")}
    st = C.look(clip.get("look") or run_look, overrides)
    st.setdefault("censor", True)
    for k in ("highlight", "primary"):
        v = str(st.get(k, ""))
        if v.startswith("#") and len(v) == 7:          # UI sends #RRGGBB; ASS wants &H00BBGGRR
            st[k] = f"&H00{v[5:7]}{v[3:5]}{v[1:3]}".upper()
    for k, typ in (("caption_size", int), ("words_per_chunk", int), ("speed", float), ("caption_y", float)):
        if st.get(k) is not None:
            st[k] = typ(st[k])
    if isinstance(st.get("uppercase"), str):
        st["uppercase"] = st["uppercase"] == "true"
    return st


def render_clip(run_dir, info, clip, cfg, final=False, layout="auto", run_look=None):
    d = clip_dir(run_dir, clip["n"])
    src, offset = source_section(run_dir, info, clip, d)
    dur = round(clip["end"] - clip["start"], 2)
    seek = clip["start"] - offset
    words = clip_words(d, src, offset, clip, cfg["whisper_model"])
    st = effective_style(clip, run_look)
    an = analyze_clip(d, src, seek, dur, words, cfg, run_dir)

    det = jload(d / "detected.json")
    if det is None:
        det = detect_layout(src, seek, dur)
        jsave(d / "detected.json", det)
    lay = dict(det)
    if layout and layout != "auto":
        lay["layout"] = layout
    elif lay["layout"] == "blur" and st["grade"] == "cinematic":
        lay["layout"] = "letterbox"
    lay.setdefault("cx", 0.5)
    lay.setdefault("cy", 0.2)
    lay.setdefault("fw", 0.1)
    for k in ("cx", "cy", "fw"):
        if clip.get(k) is not None:
            lay[k] = float(clip[k])
    if lay["layout"] == "facecam" and "cam_rect" not in det and clip.get("cx") is None:
        det["cam_rect"] = detect_cam_rect(src, seek, dur, lay)
        jsave(d / "detected.json", det)
    if lay["layout"] == "facecam" and clip.get("cx") is None:
        fvw, fvh, _ = probe_video(src)
        lay["cam_rect"] = consensus_cam_rect(run_dir, det.get("cam_rect"), lay, fvw, fvh)
    if clip.get("cx") is not None or clip.get("fw") is not None:
        lay.pop("cam_rect", None)                         # manual position from the studio wins
    jsave(d / "layout.json", lay)

    if "bars" not in det:
        det["bars"] = detect_bars(src, seek, dur)
        jsave(d / "detected.json", det)
    k, vsrc, vseek = 1.0, src, seek
    if final and lay["layout"] in ("facecam", "crop"):
        hq = hq_section(run_dir, info, clip, d, src, offset)
        if hq:
            vsrc, voff, k = hq
            vseek = clip["start"] - voff
            if lay.get("cam_rect"):
                lay["cam_rect"] = [v * k for v in lay["cam_rect"]]

    speed = float(st.get("speed") or 1.0)
    out_dur = dur / speed
    vw, vh, fps = probe_video(vsrc)
    if "bars" not in det:
        det["bars"] = detect_bars(src, seek, dur)
        jsave(d / "detected.json", det)
    crop = ""
    bars = det.get("bars")
    if bars and k != 1.0:
        bars = [round(v * k) for v in bars]
    if bars and (bars[1] < vh * 0.93 or bars[0] < vw * 0.93) and bars[1] > vh * 0.4 and bars[0] > vw * 0.4:
        crop = f"crop={bars[0]}:{bars[1]}:{bars[2]}:{bars[3]},"
        vw, vh = bars[0], bars[1]
        if lay.get("cam_rect"):
            r = lay["cam_rect"]
            lay["cam_rect"] = [r[0] - bars[2], max(0, r[1] - bars[3]), r[2], r[3]]
    pre = f"[0:v]{crop}setpts=PTS/{speed}[v0];" if speed != 1.0 else f"[0:v]{crop}null[v0];"
    graph, anchors = filtergraph(lay, vw, vh)
    graph = pre + graph

    # cold open: replay the peak moment first, white flash, then the clip from its start
    tw = teaser_window(clip, src, seek, dur, words)
    tdur = (tw[1] - tw[0]) if tw else 0.0
    if tw:
        # input 0 = teaser, input 1 = main (this order keeps the teaser's audio in sync; the reverse order
        # made ffmpeg feed the main clip's audio under the teaser's picture)
        tv = (f"[0:v]setpts=PTS-STARTPTS,fade=t=out:st={max(0, tdur - 0.12):.2f}:d=0.12:color=white[tv];"
              f"[0:a]asetpts=PTS-STARTPTS,afade=t=out:st={max(0, tdur - 0.08):.2f}:d=0.08[ta];"
              "[1:v]setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.18:color=white[mv];"
              "[1:a]asetpts=PTS-STARTPTS,afade=t=in:d=0.08[ma];"
              "[tv][ta][mv][ma]concat=n=2:v=1:a=1[vc][ac];")
        graph = tv + graph.replace("[0:v]", "[vc]", 1)
        twords = [{**w, "start": w["start"] - tw[0], "end": w["end"] - tw[0]} for w in words
                  if w["start"] >= tw[0] - 0.05 and w["end"] <= tw[1] + 0.05]
        words_all = twords + [{**w, "start": w["start"] + tdur, "end": w["end"] + tdur} for w in words]
    else:
        words_all = words
    out_dur = (dur + tdur) / speed

    cap_mode = clip.get("captions") or "auto"
    # game dialogue subtitles inside gameplay are not the streamer's words, so keep our captions for webcam layouts
    captions_on = cap_mode == "on" or (cap_mode == "auto" and (not an.get("source_subs") or lay["layout"] == "facecam"))
    write_ass(d / "captions.ass", words_all, clip.get("hook", ""), out_dur, st, anchors, captions_on, speed)
    grade = GRADES.get(st["grade"], "")
    if st["grade"] == "cinematic" and (an.get("luma") or 128) < 75:
        grade = "noise=alls=6:allf=t+u"          # footage is already dark/graded: add grain only
    tail = (grade + "," if grade else "") + "ass=captions.ass:fontsdir=fonts"
    if final:
        graph += f";[base]{tail}[v]"
        fpsmax = "60" if (clip.get("look") or run_look) in ("streamer", "hype") else "30"
        venc = ["-c:v", "libx264", "-preset", "faster", "-crf", "21", "-fpsmax", fpsmax]
        abr, out = "160k", d / "final.mp4"
    else:
        graph += f";[base]{tail},scale=540:960[v]"
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-r", "30"]
        abr, out = "96k", d / "preview.mp4"

    # audio: speed -> optional music bed with ducking -> loudness normalise for TikTok (-14 LUFS)
    music_mode = st.get("music") or "off"
    track = None
    if music_mode == "on" or (music_mode == "auto" and an.get("has_bed") is False):
        track = pick_track(st["music_mood"], clip["n"], clip.get("music_track"))
    ain = "[ac]" if tw else "[0:a]"
    a = f"{ain}atempo={speed}[sp0];" if speed != 1.0 else f"{ain}anull[sp0];"
    # static gain to -14 LUFS + limiter (single-pass dynamic loudnorm garbled the first ~3s of audio)
    lufs = measure_lufs(src, seek, dur)
    gain = max(-12.0, min(12.0, -14.0 - lufs)) if lufs is not None and lufs > -70 else 0.0
    norm = f"volume={gain:.1f}dB,alimiter=limit=0.89:level=0"
    extra = (2 if tw else 1) if vsrc != src else 0      # separate sound inputs when the picture is the 4K copy
    mi = (2 if tw else 1) + extra
    if track:
        a += (f"[sp0]asplit=2[sp][key];[{mi}:a]volume={clip.get('music_volume') or 0.22},"
              f"afade=t=in:d=1.2,afade=t=out:st={max(0, out_dur - 1.8):.2f}:d=1.8[m];"
              "[m][key]sidechaincompress=threshold=0.05:ratio=8:attack=10:release=400[md];"
              f"[sp][md]amix=inputs=2:duration=first:normalize=0[mix];[mix]{norm}[a]")
    else:
        a += f"[sp0]{norm}[a]"
    graph += ";" + a

    # fonts: copy the project fonts next to the clip so libass finds them without Windows path escaping
    fdst = d / "fonts"
    if not fdst.exists():
        fdst.mkdir()
        for f in FONTS_DIR.glob("*.ttf"):
            if f.name.startswith("ui-"):
                continue
            (fdst / f.name).write_bytes(f.read_bytes())

    log(f"  render {'FINAL' if final else 'preview'} #{clip['n']} {lay['layout']} look={clip.get('look') or run_look} "
        f"{out_dur:.0f}s teaser={f'{tdur:.1f}s' if tw else 'off'} speed={speed} captions={'on' if captions_on else 'off'} music={Path(track).stem if track else 'none'}")
    cmd = [FFMPEG, "-y", "-v", "error"]
    if tw:
        cmd += ["-ss", f"{vseek + tw[0]:.3f}", "-t", f"{tdur:.3f}", "-i", vsrc]
    cmd += ["-ss", f"{vseek:.3f}", "-t", f"{dur:.3f}", "-i", vsrc]
    if extra:    # picture from the sharper copy, sound from the normal source (both seeked to the same moment)
        if tw:
            cmd += ["-ss", f"{seek + tw[0]:.3f}", "-t", f"{tdur:.3f}", "-i", src]
        cmd += ["-ss", f"{seek:.3f}", "-t", f"{dur:.3f}", "-i", src]
        graph = (graph.replace("[0:a]", "[2:a]").replace("[1:a]", "[3:a]") if tw else graph.replace("[0:a]", "[1:a]"))
    if track:
        cmd += ["-stream_loop", "-1", "-i", track]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", *venc, "-c:a", "aac", "-b:a", abr, "-ar", "48000",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-t", f"{out_dur:.3f}", out.name]
    run(cmd, cwd=d, timeout=3600)
    if not final:
        run([FFMPEG, "-y", "-v", "error", "-ss", f"{min(2.0, out_dur / 3):.2f}", "-i", out.name, "-frames:v", "1",
             "-vf", "scale=270:480", "poster.jpg"], cwd=d)
    meta = {"layout": lay["layout"], "look": clip.get("look") or run_look, "speed": speed, "captions": captions_on,
            "teaser": [round(clip["start"] + tw[0], 2), round(clip["start"] + tw[1], 2)] if tw else None,
            "music": Path(track).name if track else None, "analysis": an}
    jsave(d / ("final_meta.json" if final else "preview_meta.json"), meta)
    return str(out), lay["layout"]
