"""Stage 2: transcript (YouTube json3 captions when available, else local faster-whisper)."""
import glob
import json
import tempfile
from pathlib import Path

from .util import FFMPEG, MODELS, jload, jsave, log, run

_model = {}


def whisper_model(name):
    if name not in _model:
        from faster_whisper import WhisperModel
        log(f"whisper: loading {name} (first time downloads it)")
        _model[name] = WhisperModel(name, device="cpu", compute_type="int8", download_root=str(MODELS))
    return _model[name]


def words_to_segments(words, max_words=14, gap=0.7):
    segs, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        end_punct = w["text"].rstrip().endswith((".", "?", "!"))
        if nxt is None or end_punct or len(cur) >= max_words or nxt["start"] - w["end"] > gap:
            segs.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                         "text": " ".join(x["text"].strip() for x in cur).strip()})
            cur = []
    return segs


def parse_json3(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    events = [e for e in d.get("events", []) if e.get("segs")]
    words, lines = [], []
    per_word = any(len(e["segs"]) > 1 for e in events)
    for i, e in enumerate(events):
        t0 = e["tStartMs"] / 1000
        t1 = t0 + e.get("dDurationMs", 0) / 1000
        text = "".join(s.get("utf8", "") for s in e["segs"]).replace("\n", " ").strip()
        if not text:
            continue
        if per_word:
            ws = [s for s in e["segs"] if s.get("utf8", "").strip()]
            for j, s in enumerate(ws):
                st = t0 + s.get("tOffsetMs", 0) / 1000
                en = t0 + ws[j + 1].get("tOffsetMs", 0) / 1000 if j + 1 < len(ws) else t1
                words.append({"start": round(st, 2), "end": round(max(en, st + 0.05), 2), "text": s["utf8"].strip()})
        else:
            lines.append({"start": round(t0, 2), "end": round(t1, 2), "text": text})
    if per_word:
        words.sort(key=lambda w: w["start"])
        for a, b in zip(words, words[1:]):   # auto-caption events overlap; clamp
            a["end"] = min(a["end"], max(b["start"], a["start"] + 0.05))
        return words_to_segments(words), words
    return lines, []


def extract_wav(src, out_wav, start=None, dur=None):
    cmd = [FFMPEG, "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0, start):.2f}"]
    cmd += ["-i", src]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", out_wav]
    run(cmd, timeout=3600)


def whisper_words(src, model_name, start=0.0, dur=None):
    """Word-level timestamps for src[start:start+dur] (absolute times)."""
    with tempfile.TemporaryDirectory() as td:
        wav = str(Path(td) / "a.wav")
        extract_wav(src, wav, start if start else None, dur)
        segs, _ = whisper_model(model_name).transcribe(wav, word_timestamps=True, vad_filter=True,
                                                       beam_size=1, condition_on_previous_text=False)
        words = []
        for s in segs:
            for w in s.words or []:
                if w.word.strip():
                    words.append({"start": round(w.start + start, 2), "end": round(w.end + start, 2),
                                  "text": w.word.strip()})
    return words


def build(run_dir, info, audio, cfg):
    run_dir = Path(run_dir)
    out = run_dir / "transcript.json"
    tr = jload(out)
    if tr is not None:
        return tr
    subs = sorted(glob.glob(str(run_dir / "src.*.json3")), key=lambda p: ("orig" not in p, p))
    if subs:
        log(f"transcript: from captions {Path(subs[0]).name}")
        segs, words = parse_json3(subs[0])
        tr = {"source": "captions", "segments": segs, "words": words, "complete": True}
    elif audio and info["duration"] / 60 <= cfg["full_whisper_max_min"]:
        log(f"transcript: whisper {cfg['whisper_model']} on full audio ({info['duration'] / 60:.0f} min, CPU)")
        words = whisper_words(audio, cfg["whisper_model"])
        tr = {"source": "whisper", "segments": words_to_segments(words), "words": words, "complete": True}
    else:
        log("transcript: none yet (long video without captions -> signal-first mode)")
        tr = {"source": "none", "segments": [], "words": [], "complete": False}
    jsave(out, tr)
    return tr
