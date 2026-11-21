"""Fallback 'watching': sample frames and let Haiku describe them in batches.

Only used when there is little or no speech (gameplay without commentary, music, sports...),
so the expensive path is cheap-model-only and frame-capped.
"""
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import AgentPending, ask
from .util import FFMPEG, jload, jsave, log, run, ytdlp

SCHEMA = {"type": "object", "properties": {"frames": {"type": "array", "items": {"type": "object", "properties": {
    "t": {"type": "number"}, "desc": {"type": "string"}, "interest": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["t", "desc", "interest"]}}}, "required": ["frames"]}


def low_res_video(run_dir, info):
    if info["source"] == "file":
        return info["local_path"]
    hits = glob.glob(str(Path(run_dir) / "video_low.*"))
    if hits:
        return hits[0]
    log("vision: downloading 360p copy for frame sampling")
    run([*ytdlp(), "-f", "bv*[height<=360]/b[height<=360]/wv*/b", "--no-warnings", "-o",
         str(Path(run_dir) / "video_low.%(ext)s"), info["webpage_url"]], timeout=7200)
    return glob.glob(str(Path(run_dir) / "video_low.*"))[0]


def visual_lines(run_dir, info, cfg):
    run_dir = Path(run_dir)
    cached = jload(run_dir / "vision.json")
    if cached is not None:
        return cached
    vc = cfg["vision"]
    every = max(vc["every_sec"], (info["duration"] or 0) / vc["max_frames"])
    src = low_res_video(run_dir, info)
    fdir = run_dir / "frames"
    fdir.mkdir(exist_ok=True)
    if not list(fdir.glob("*.jpg")):
        log(f"vision: extracting a frame every {every:.0f}s")
        run([FFMPEG, "-v", "error", "-i", src, "-vf", f"fps=1/{every:.3f},scale=384:-2", "-q:v", "5",
             str(fdir / "%05d.jpg")], timeout=7200)
    frames = sorted(fdir.glob("*.jpg"))
    items = [(round((i + 0.5) * every, 1), f) for i, f in enumerate(frames)]
    batches = [items[i:i + vc["batch"]] for i in range(0, len(items), vc["batch"])]
    log(f"vision: {len(frames)} frames in {len(batches)} {cfg['models']['vision']} calls")

    def one(b):
        listing = "\n".join(f"t={t}s: {f.resolve()}" for t, f in b)
        prompt = (f"Video: \"{info['title']}\". Open and look at EVERY image file below (frames of the video at the "
                  "given time). For each frame return t, a <=18-word description of what is happening (people, action, "
                  "on-screen text, emotion), and interest 1-5 = how likely this moment is part of something funny, "
                  "shocking, impressive or dramatic that strangers would watch.\n" + listing)
        return ask(prompt, SCHEMA, cfg["models"]["vision"], run_dir, "vision", tools=["Read"], add_dirs=[fdir])["frames"]

    waiting = []

    def safe(b):
        try:
            return one(b)
        except AgentPending as e:      # agent mode: write every batch's question before stopping
            waiting.extend(e.files)
            return []

    lines = []
    with ThreadPoolExecutor(3) as ex:
        for res in ex.map(safe, batches):
            for fr in res:
                lines.append({"start": fr["t"], "end": fr["t"] + every,
                              "text": f"(visual, interest {fr['interest']}/5) {fr['desc']}", "visual": True})
    if waiting:
        raise AgentPending(waiting)
    lines.sort(key=lambda x: x["start"])
    jsave(run_dir / "vision.json", lines)
    return lines
