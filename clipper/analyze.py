"""Per-clip source analysis so the edit adapts to the footage.

source_subs  burned-in subtitles already on screen? (Haiku looks at 3 frames) -> skip our captions
has_bed      music / ambience under the voice? (audio: loudness in speech gaps vs speech) -> skip added music
"""
import subprocess

import numpy as np

from .llm import AgentPending, ask
from .util import FFMPEG, jload, jsave, run

SCHEMA = {"type": "object", "properties": {"frames": {"type": "array", "items": {"type": "object", "properties": {
    "subtitles": {"type": "boolean"},
    "region": {"type": "string", "enum": ["top", "middle", "bottom", "none"]},
    "scene": {"type": "string"}}, "required": ["subtitles", "region", "scene"]}}}, "required": ["frames"]}


def _rms_track(src, seek, dur, hop=0.1, sr=8000):
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{seek:.2f}", "-t", f"{dur:.2f}", "-i", src, "-vn", "-ac", "1",
                        "-ar", str(sr), "-f", "s16le", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.int16).astype(np.float32)
    n = int(sr * hop)
    k = len(a) // n
    if k == 0:
        return np.array([])
    a = a[:k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt((a * a).mean(1)) + 1e-6)


def music_bed(src, seek, dur, words, hop=0.1):
    db = _rms_track(src, seek, dur, hop)
    if db.size == 0:
        return None, None
    speech = np.zeros(db.size, bool)
    for w in words:
        a, b = int(max(0, w["start"] - 0.12) / hop), int((w["end"] + 0.12) / hop) + 1
        speech[a:b] = True
    gaps, talk = db[~speech], db[speech]
    if gaps.size < 8 or talk.size < 8:
        return None, None                      # can't tell (no pauses or no speech)
    rel = float(np.median(gaps) - np.median(talk))
    # dry voice: pauses are >22 dB quieter than speech; music/ambience keeps pauses within ~18 dB
    return (rel > -18.0), round(rel, 1)


def analyze_clip(d, src, seek, dur, words, cfg, run_dir):
    key = [round(seek, 2), round(dur, 2)]
    cached = jload(d / "analysis.json")
    if cached and cached.get("key") == key and "luma" in cached:
        return cached
    has_bed, rel = music_bed(src, seek, dur, words)
    frames = []
    for i, f in enumerate((0.2, 0.5, 0.8)):
        out = d / f"probe_{i}.jpg"
        run([FFMPEG, "-y", "-v", "error", "-ss", f"{seek + dur * f:.2f}", "-i", src, "-frames:v", "1",
             "-vf", "scale=640:-2", str(out)], check=False)
        if out.exists():
            frames.append(out)
    subs, region, scenes = False, "none", []
    if cached and "source_subs" in cached:     # same clip, maybe re-trimmed: burned-in subs don't change
        subs, region, scenes = cached["source_subs"], cached["subs_region"], cached.get("scenes", [])
        frames_for_llm = []
    else:
        frames_for_llm = frames
    if frames_for_llm:
        try:
            listing = "\n".join(str(p.resolve()) for p in frames)
            res = ask("Use the Read tool to open each image (frames from one video clip). For each frame say whether it "
                      "shows burned-in SUBTITLES (spoken words rendered as text, usually bottom or middle; NOT logos, "
                      "watermarks, UI or game HUD), their region, and a <=10-word scene description.\n" + listing,
                      SCHEMA, cfg["models"]["vision"], run_dir, "analyze", tools=["Read"], add_dirs=[d])
            fr = res["frames"]
            hits = [x for x in fr if x["subtitles"]]
            subs = len(hits) >= 2
            region = hits[0]["region"] if hits else "none"
            scenes = [x["scene"] for x in fr]
        except AgentPending:
            raise
        except Exception:  # noqa: BLE001
            pass
    luma = None
    try:
        import cv2
        vals = [float(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean()) for p in frames]
        luma = round(sum(vals) / len(vals), 1) if vals else None
    except Exception:  # noqa: BLE001
        pass
    an = {"key": key, "luma": luma, "source_subs": subs, "subs_region": region, "has_bed": has_bed, "gap_vs_speech_db": rel,
          "scenes": scenes}
    jsave(d / "analysis.json", an)
    return an
