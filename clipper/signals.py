"""Stage 3: cheap 'no-watching' signals on a 5-second grid.

heat     YouTube "most replayed" heatmap
chat     live-chat replay messages per bin (+ laugh / hype / shock keyword counts)
comments timestamp mentions in top comments (like-weighted)
loud     audio loudness relative to its local 2-minute median (screams, laughter, hype)
"""
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from .util import FFMPEG, jload, jsave, log

BIN = 5.0
MOODS = {
    "funny": re.compile(r"(lmao|lmfao|lol|haha|kekw|omegalul|icant|i can'?t|dead|crying|💀|😂|🤣|😭)", re.I),
    "hype": re.compile(r"(\bw+\b|lets go+|let'?s go+|goat|insane|clutch|🔥|pog|\bfire\b|cinema)", re.I),
    "shock": re.compile(r"(wtf|omg|what+\b|no way|bro what|😳|😱|nah+)", re.I),
    "drama": re.compile(r"(\bl+\b|ratio|cooked|fell off|exposed|beef|caught|😬)", re.I),
}
TS = re.compile(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)")


def _rank(x):
    """Percentile-rank normalisation to 0..1 (robust to outliers)."""
    x = np.asarray(x, float)
    if x.size == 0 or np.allclose(x, x[0]):
        return np.zeros_like(x)
    order = x.argsort().argsort()
    return order / (len(x) - 1)


def heatmap(info, n):
    s = np.zeros(n)
    for h in info.get("heatmap") or []:
        a, b = int(h["start_time"] // BIN), int(math.ceil(h["end_time"] / BIN))
        s[a:min(b, n)] = np.maximum(s[a:min(b, n)], h["value"])
    return s if s.any() else None


def live_chat(path, n):
    if not Path(path).exists():
        return None, None
    counts = np.zeros(n)
    moods = {k: np.zeros(n) for k in MOODS}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
                r = d["replayChatItemAction"]
                t = int(r.get("videoOffsetTimeMsec", 0)) / 1000
            except Exception:  # noqa: BLE001
                continue
            i = int(t // BIN)
            if not 0 <= i < n:
                continue
            for a in r.get("actions", []):
                item = a.get("addChatItemAction", {}).get("item", {})
                m = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
                if not m:
                    continue
                runs = m.get("message", {}).get("runs", [])
                text = "".join(x.get("text") or "".join(x.get("emoji", {}).get("shortcuts", [])[:1]) for x in runs)
                counts[i] += 1
                for k, rx in MOODS.items():
                    if rx.search(text):
                        moods[k][i] += 1
    return (counts if counts.any() else None), moods


def comment_marks(comments, n):
    s = np.zeros(n)
    for c in comments or []:
        w = 1 + math.log1p(c.get("likes", 0))
        for h, m, sec in TS.findall(c.get("text", "")):
            t = int(h or 0) * 3600 + int(m) * 60 + int(sec)
            i = int(t // BIN)
            if 0 <= i < n:
                s[max(0, i - 2):i + 3] += w   # people timestamp slightly before the moment
    return s if s.any() else None


def loudness(audio, n):
    if not audio:
        return None
    sr = 4000
    per = int(sr * BIN)
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", audio, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         stdout=subprocess.PIPE)
    vals = []
    while True:
        buf = p.stdout.read(per * 2)
        if not buf:
            break
        a = np.frombuffer(buf, np.int16).astype(np.float32)
        vals.append(20 * np.log10(np.sqrt(np.mean(a * a)) + 1e-6))
    p.wait()
    v = np.array(vals[:n] + [vals[-1] if vals else 0] * max(0, n - len(vals)))
    # excitement = loudness above the local 2-minute median
    k = int(120 / BIN)
    pad = np.pad(v, k, mode="edge")
    med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(n)])
    return np.clip(v - med, 0, None)


def build(run_dir, info, audio):
    run_dir = Path(run_dir)
    out = run_dir / "signals.json"
    sig = jload(out)
    if sig is not None:
        return sig
    n = int(math.ceil((info.get("duration") or 0) / BIN)) or 1
    series, extra = {}, {}
    h = heatmap(info, n)
    if h is not None:
        series["heat"] = h
    c, moods = live_chat(run_dir / "src.live_chat.json", n)
    if c is not None:
        series["chat"] = c
        extra["moods"] = {k: v.round(1).tolist() for k, v in moods.items()}
    cm = comment_marks(jload(run_dir / "comments.json", []), n)
    if cm is not None:
        series["comments"] = cm
    log("signals: audio loudness")
    ld = loudness(audio, n)
    if ld is not None:
        series["loud"] = ld
    weights = {"heat": 1.0, "chat": 1.2, "comments": 0.8, "loud": 0.6}
    comb = np.zeros(n)
    tot = 0
    for k, v in series.items():
        # smooth over ~15s so peaks become regions
        vs = np.convolve(v, np.ones(3) / 3, mode="same")
        comb += weights[k] * _rank(vs)
        tot += weights[k]
    comb = comb / tot if tot else comb
    sig = {"bin": BIN, "available": list(series), "combined": comb.round(3).tolist(),
           "series": {k: np.asarray(v).round(3).tolist() for k, v in series.items()}, **extra}
    log(f"signals: available = {sig['available'] or 'none'}")
    jsave(out, sig)
    return sig


def describe_range(sig, a, b):
    """Short human/LLM-readable signal summary for [a, b) seconds."""
    i, j = int(a // BIN), max(int(a // BIN) + 1, int(b // BIN))
    parts = []
    s = sig["series"]
    if "heat" in s:
        parts.append(f"replay {max(s['heat'][i:j] or [0]):.2f}")
    if "chat" in s:
        base = np.median(s["chat"]) or 1
        peak = max(s["chat"][i:j] or [0])
        parts.append(f"chat {peak / base:.1f}x")
        moods = sig.get("moods", {})
        if moods:
            tot = {k: sum(v[i:j]) for k, v in moods.items()}
            top = max(tot, key=tot.get)
            if tot[top] >= 3:
                parts.append(f"chat mood {top}")
    if "comments" in s and sum(s["comments"][i:j]) > 0:
        parts.append("timestamped in comments")
    if "loud" in s and max(s["loud"][i:j] or [0]) > 6:
        parts.append("loud spike")
    comb = sig["combined"][i:j]
    score = max(comb) if comb else 0
    return score, ", ".join(parts)


def hotspots(sig, length, k):
    """Top-k non-overlapping regions (start, end) by combined signal."""
    comb = np.array(sig["combined"])
    w = max(1, int(length // BIN))
    if comb.size < w:
        return [(0.0, comb.size * BIN)]
    roll = np.convolve(comb, np.ones(w) / w, mode="valid")
    taken, out = np.zeros_like(roll, bool), []
    for i in roll.argsort()[::-1]:
        if len(out) >= k:
            break
        if taken[max(0, i - w):i + w].any():
            continue
        taken[i] = True
        out.append((float(i * BIN), float((i + w) * BIN)))
    return sorted(out)
