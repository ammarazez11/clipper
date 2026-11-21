"""Stage 4: Haiku scans the whole video in windows -> candidates; Opus picks finals + writes copy."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import signals as S
from .llm import AgentPending, ask
from .transcript import whisper_words, words_to_segments
from .util import hms, jload, jsave, log

SCAN_SCHEMA = {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
    "start": {"type": "number"}, "end": {"type": "number"}, "title": {"type": "string"},
    "hook": {"type": "string"}, "reason": {"type": "string"},
    "score": {"type": "integer", "minimum": 1, "maximum": 10},
    "mood": {"type": "string", "enum": ["funny", "hype", "shocking", "drama", "wholesome", "informative", "other"]}},
    "required": ["start", "end", "title", "hook", "reason", "score", "mood"]}}}, "required": ["candidates"]}

FINAL_SCHEMA = {"type": "object", "properties": {
    "look": {"type": "string", "enum": ["streamer", "motivation", "podcast", "comedy", "education", "hype"]},
    "look_reason": {"type": "string"},
    "clips": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "integer"}, "start": {"type": "number"}, "end": {"type": "number"},
    "hook": {"type": "string"}, "title": {"type": "string"}, "caption": {"type": "string"},
    "hashtags": {"type": "array", "items": {"type": "string"}}, "why": {"type": "string"},
    "teaser_start": {"type": "number"}, "teaser_end": {"type": "number"}},
    "required": ["id", "start", "end", "hook", "title", "caption", "hashtags", "why", "teaser_start", "teaser_end"]}}},
    "required": ["look", "clips"]}


def _lines(segs, sig, a, b):
    out, last_sig = [], -999
    for s in segs:
        if s["end"] < a or s["start"] > b:
            continue
        if sig["available"] and s["start"] - last_sig >= 30:
            score, desc = S.describe_range(sig, s["start"], s["start"] + 30)
            if score >= 0.75 and desc:
                out.append(f"[{s['start']:.0f}] >>> SIGNAL {score:.2f}: {desc}")
                last_sig = s["start"]
        out.append(f"[{s['start']:.1f}] {s['text']}")
    return "\n".join(out)


def _blocks(tr, extra, info, cfg):
    segs = sorted(tr["segments"] + extra, key=lambda s: s["start"])
    if tr["complete"] or extra:
        dur = info["duration"] or (segs[-1]["end"] if segs else 0)
        win, ov = cfg["window_sec"], cfg["window_overlap"]
        a, out = 0.0, []
        while a < dur:
            out.append((a, min(dur, a + win)))
            a += win - ov
        return segs, out
    # partial transcript (signal-first): one block per transcribed region, grouped
    regions, out, cur = tr.get("regions", []), [], None
    for r in regions:
        if cur and r[1] - cur[0] <= cfg["window_sec"] * 1.5:
            cur = (cur[0], r[1])
        else:
            if cur:
                out.append(cur)
            cur = tuple(r)
    if cur:
        out.append(cur)
    return segs, out


def signal_first(run_dir, tr, sig, audio, cfg, preset, count):
    """Long video without captions: transcribe only the hottest regions."""
    if tr.get("regions") or not audio:
        return tr
    k = min(30, max(8, count * 4))
    regs = S.hotspots(sig, preset["max"] + 30, k)
    log(f"signal-first: whisper on {len(regs)} hot regions instead of the full {len(sig['combined']) * S.BIN / 3600:.1f} h")
    words = []
    for i, (a, b) in enumerate(regs, 1):
        log(f"  region {i}/{len(regs)} {hms(a)}-{hms(b)}")
        words += whisper_words(audio, cfg["whisper_model"], a, b - a)
    words.sort(key=lambda w: w["start"])
    tr = {**tr, "source": "whisper-regions", "segments": words_to_segments(words), "words": words,
          "regions": [list(r) for r in regs]}
    jsave(Path(run_dir) / "transcript.json", tr)
    return tr


def _snap(c, segs, preset):
    starts = [s for s in segs if s["start"] <= c["start"] + 0.5]
    a = starts[-1]["start"] if starts else c["start"]
    ends = [s for s in segs if s["end"] >= c["end"] - 0.5 and s["start"] >= a]
    b = ends[0]["end"] if ends else c["end"]
    following = [s for s in segs if s["start"] >= b - 0.01]
    while b - a < preset["min"] and following:      # too short: extend into next lines
        b = following.pop(0)["end"]
    if b - a > preset["max"]:                        # too long: cut at last line that fits
        fit = [s for s in segs if a < s["end"] <= a + preset["max"]]
        b = fit[-1]["end"] if fit else a + preset["max"]
    return round(a, 2), round(b + 0.25, 2)


def _iou(x, y):
    inter = max(0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    return inter / (max(x["end"], y["end"]) - min(x["start"], y["start"]) or 1)


def _excerpt(segs, a, b, limit=330):
    words = " ".join(f"[{s['start']:.1f}] {s['text']}" for s in segs if s["end"] > a and s["start"] < b).split()
    return " ".join(words[:limit]) + (" ..." if len(words) > limit else "")


def scan(run_dir, info, tr, sig, extra, cfg, preset, intent, fresh=False):
    run_dir = Path(run_dir)
    key = {"preset": preset, "intent": intent or ""}
    cached = jload(run_dir / "candidates.json")
    if cached and cached.get("key") == key and not fresh:
        return cached["candidates"]
    segs, blocks = _blocks(tr, extra, info, cfg)
    if not segs:
        raise RuntimeError("nothing to analyse: no transcript, no speech and vision disabled")
    want = (f"\nThe user specifically wants: {intent}. Prefer those moments strongly." if intent else "")
    base = (f"Video: \"{info['title']}\" by {info.get('channel') or info.get('uploader') or 'unknown'} "
            f"(duration {hms(info['duration'] or 0)}).\n"
            f"Find moments that would be strong STANDALONE TikTok clips of {preset['min']}-{preset['max']} seconds "
            f"(ideal ~{preset['target']}s) for an English-speaking general audience that has never seen this creator."
            f"{want}\nRules:\n"
            "- Must make sense with zero context and have a clear payoff: punchline, reveal, big reaction, fail, win, "
            "argument, hot take, shocking statement, satisfying moment.\n"
            "- start = just before the setup (no dead air); end = right after the payoff/reaction lands.\n"
            "- SIGNAL lines mean real viewers replayed it / chat exploded / loud reaction: strong evidence, "
            "but judge the content yourself. (visual) lines describe what is on screen.\n"
            "- Skip intros, outros, sponsor reads, waiting/loading screens, inside jokes needing context, "
            "and anything hateful, sexual or gratuitously graphic.\n"
            "- Use the [seconds] timestamps; start/end are seconds from the video start and must lie inside this part.\n"
            "- Return 0-4 candidates. Return none if nothing here is genuinely good. Score 8+ only for moments you "
            "would bet on.\n")
    log(f"scan: {len(blocks)} windows with {cfg['models']['scan']} (x{cfg['scan_concurrency']} parallel)")

    waiting = []

    def one(blk):
        a, b = blk
        text = _lines(segs, sig, a, b)
        if len(text) < 200:
            return []
        try:
            res = ask(base + f"\nPart {hms(a)}-{hms(b)} ({a:.0f}s-{b:.0f}s):\n{text}", SCAN_SCHEMA,
                      cfg["models"]["scan"], run_dir, "scan")
        except AgentPending as e:      # agent mode: write every window's question before stopping
            waiting.extend(e.files)
            return []
        return [c for c in res["candidates"] if a - 5 <= c["start"] < c["end"] <= b + 30]

    raw = []
    with ThreadPoolExecutor(cfg["scan_concurrency"]) as ex:
        for r in ex.map(one, blocks):
            raw += r
    if waiting:
        raise AgentPending(waiting)
    speech = [s for s in segs if not s.get("visual")] or segs
    cands = []
    for c in raw:
        c["start"], c["end"] = _snap(c, speech, preset)
        sscore, sdesc = S.describe_range(sig, c["start"], c["end"]) if sig["available"] else (0, "")
        c["signal"], c["signal_desc"] = round(sscore, 2), sdesc
        c["rank"] = round(0.7 * c["score"] / 10 + 0.3 * sscore if sig["available"] else c["score"] / 10, 3)
        cands.append(c)
    cands.sort(key=lambda c: -c["rank"])
    keep = []
    for c in cands:
        if all(_iou(c, k) < 0.35 for k in keep):
            keep.append(c)
    for i, c in enumerate(keep):
        c["id"] = i
    log(f"scan: {len(raw)} raw -> {len(keep)} unique candidates")
    jsave(run_dir / "candidates.json", {"key": key, "candidates": keep})
    return keep


def finalize(run_dir, info, tr, extra, cands, cfg, preset, count, intent, fresh=False):
    run_dir = Path(run_dir)
    cached = jload(run_dir / "clips.json")
    key = {"preset": preset, "intent": intent or "", "count": count, "n_cands": len(cands)}
    if cached and cached.get("key") == key and not fresh:
        return cached["clips"]
    segs = sorted(tr["segments"] + extra, key=lambda s: s["start"])
    top = cands[:min(len(cands), max(count * 2 + 6, 12), 30)]
    if not top:
        raise RuntimeError("scan found no candidates")
    blocks = []
    for c in top:
        blocks.append(f"### id {c['id']} | {c['start']:.1f}-{c['end']:.1f}s ({c['end'] - c['start']:.0f}s) | "
                      f"scanner score {c['score']}/10, mood {c['mood']}"
                      + (f" | signals: {c['signal_desc']}" if c.get("signal_desc") else "")
                      + f"\nscanner note: {c['title']} - {c['reason']}\n{_excerpt(segs, c['start'], c['end'])}")
    who = info.get("channel") or info.get("uploader") or ""
    prompt = (f"Source video: \"{info['title']}\" by {who}.\n"
              + (f"User intent: {intent}\n" if intent else "")
              + f"Below are {len(top)} candidate clips found by a cheaper scanner model. Pick the best {count} "
              "(fewer if the rest are weak) for a TikTok clip account posting in English to a general audience. "
              "Rank best first.\n"
              f"For each pick:\n- start/end: you may tighten or shift by up to 8s using the [timestamps], "
              f"keep the length within {preset['min']}-{preset['max']}s.\n"
              "- hook: on-screen text shown at the top, max 8 words, makes a stranger NEED to see what happens; "
              "no lies, no spoilers of the payoff.\n"
              "- title: short internal name.\n"
              "- caption: TikTok description <=150 chars, natural, includes the creator's name and the searchable topic "
              "(game, show, event).\n"
              "- hashtags: 4-7 lowercase tags with #, mix one broad trend tag, the creator, the topic, one niche; "
              "at most one generic tag like #fyp.\n"
              "- why: one sentence on why this will perform.\n"
              "- teaser_start/teaser_end: the 1.5-3s peak moment INSIDE the clip (the scream, the punchline, the "
              "shocking line), shown as a cold open in the first second before the clip plays from its start. "
              "Use the [timestamps]; pick it from the second half if possible, and never the clip's first 3 seconds.\n"
              "Also choose the edit look for this video: streamer (gaming/stream reactions), motivation (speeches, "
              "self-improvement, gym), podcast (interviews/talk shows), comedy, education (explainers), hype (sports). "
              "Clip lengths may vary inside the allowed range: short punchy moments can stay short.\n\n"
              + "\n\n".join(blocks))
    # new selection: old per-clip folders / review decisions belong to different clips
    import shutil
    shutil.rmtree(run_dir / "clips", ignore_errors=True)
    (run_dir / "decisions.json").unlink(missing_ok=True)
    log(f"final: {cfg['models']['final']} choosing {count} of {len(top)}")
    res = ask(prompt, FINAL_SCHEMA, cfg["models"]["final"], run_dir, "final")
    by_id = {c["id"]: c for c in top}
    speech = [s for s in segs if not s.get("visual")] or segs
    clips = []
    for n, p in enumerate(res["clips"][:count], 1):
        c = by_id.get(p["id"], {})
        a, b = float(p["start"]), float(p["end"])
        if not (preset["min"] <= b - a <= preset["max"] + 5):
            a, b = _snap({"start": a, "end": b}, speech, preset)   # extend/trim on sentence boundaries
        if b - a < preset["min"]:                                   # hard floor (e.g. Creator Rewards > 60s)
            b = a + preset["min"]
        clips.append({"n": n, "cand_id": p["id"], "start": round(a, 2), "end": round(b, 2),
                      "hook": p["hook"].strip(), "title": p["title"], "caption": p["caption"].strip(),
                      "hashtags": [h if h.startswith("#") else "#" + h for h in p["hashtags"]],
                      "why": p["why"], "mood": c.get("mood"),
                      "teaser_start": p.get("teaser_start"), "teaser_end": p.get("teaser_end"), "scanner_score": c.get("score"),
                      "signal": c.get("signal_desc", "")})
    jsave(run_dir / "clips.json", {"key": key, "look": res.get("look", "streamer"),
                                   "look_reason": res.get("look_reason", ""), "clips": clips})
    return clips
