"""clip <command> ...   (Windows: clip.bat, macOS/Linux: ./clip.sh)

  run URL [options]   analyse a video, pick clips, render previews, open the studio (--review)
  resume RUN          re-run a run with its saved options (after answering agent questions, or after a crash)
  studio              open the review studio (starts it in the background if needed; --restart after updates)
  status RUN          where a run is: stages done, previews, questions waiting, finals
  pending RUN         list questions waiting for the agent (agent mode)
  finalize RUN        render kept clips in full quality into ready/RUN/ with TikTok/Instagram/YouTube text
  doctor              check that everything is installed

Pipeline (each stage cached in runs/<id>/, safe to re-run):
  1 fetch      metadata, captions, live-chat replay, top comments, audio      (no AI)
  2 transcript YouTube captions, else local whisper (full, or hot-spots only)  (no AI)
  3 signals    replay heatmap, chat bursts + mood, comment timestamps, loudness (no AI)
  4 vision     only if there's little speech: frame descriptions               (AI, cheap tier)
  5 scan       whole video in windows -> candidate moments                     (AI, cheap tier)
  6 final      pick best N, hook / caption / hashtags / look                   (AI, best tier)
  7 render     540p previews -> review studio -> 1080p finals into ready/
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from . import config, fetch, review, select, signals, transcript, vision
from .llm import AgentPending, backend, cost_summary, pending
from .render import render_clip
from .util import READY, ROOT, RUNS, is_url, jload, jsave, log, set_logfile

# PowerShell never runs files from the current folder by bare name ("clip" is Windows' clipboard tool)
CLIP = r".\clip.bat" if sys.platform == "win32" else "./clip.sh"


def run_id(source, name=None):
    if name:
        return re.sub(r"[^\w.-]+", "-", name)
    m = re.search(r"(?:v=|youtu\.be/|/videos/|/live/|shorts/)([\w-]{6,})", source)
    base = m.group(1) if m else re.sub(r"[^\w]+", "-", Path(source).stem if not is_url(source) else source)[-30:]
    return f"{base}-{hashlib.sha1(source.encode()).hexdigest()[:6]}"


def cost_line(rd):
    return ", ".join(f"{m}: {v['calls']} calls, USD {v['cost_usd']}" for m, v in cost_summary(rd).items())


def cmd_run(a):
    cfg = config.load()
    if a.whisper:
        cfg["whisper_model"] = a.whisper
    if backend() == "agent":       # fewer, larger questions: the operating agent has a big context window
        cfg["window_sec"] = cfg.get("agent_window_sec", 1800)
    preset = dict(cfg["presets"][a.preset])
    if a.min:
        preset["min"] = a.min
    if a.max:
        preset["max"] = a.max
    preset["target"] = min(max(preset["target"], preset["min"]), preset["max"])
    rid = run_id(a.source, a.name)
    rd = RUNS / rid
    rd.mkdir(parents=True, exist_ok=True)
    set_logfile(rd / "log.txt")
    a.run = rid
    if a.fresh:
        for f in ("candidates.json", "clips.json"):
            (rd / f).unlink(missing_ok=True)
    jsave(rd / "params.json", {k: v for k, v in vars(a).items() if k not in ("fresh", "cmd")})
    log(f"run {rid}  ->  {rd}   (AI: {backend()})")

    info = fetch.fetch(a.source, rd, cfg, want_chat=cfg["chat"] and not a.no_chat)
    audio = fetch.audio_path(rd, info)
    log(f"video: {info['title']} ({(info['duration'] or 0) / 60:.0f} min)")
    tr = transcript.build(rd, info, audio, cfg)
    sig = signals.build(rd, info, audio)
    if not tr["complete"]:
        tr = select.signal_first(rd, tr, sig, audio, cfg, preset, a.count)

    extra = []
    n_words = len(tr["words"]) or len(" ".join(s["text"] for s in tr["segments"]).split())
    wpm = n_words / max(1, (info["duration"] or 60) / 60)
    use_vision = a.vision == "on" or (a.vision == "auto" and tr["complete"] and wpm < 25
                                       and (info["duration"] or 0) <= 3 * 3600)
    if use_vision:
        log(f"vision: on (speech density {wpm:.0f} words/min)")
        extra = vision.visual_lines(rd, info, cfg)

    cands = select.scan(rd, info, tr, sig, extra, cfg, preset, a.intent, fresh=a.fresh)
    clips = select.finalize(rd, info, tr, extra, cands, cfg, preset, a.count, a.intent, fresh=a.fresh)
    look = (jload(rd / "clips.json") or {}).get("look")
    log(f"{len(clips)} clips chosen (look: {look}); rendering previews")
    waiting = []
    for c in clips:
        try:
            render_clip(rd, info, c, cfg, final=False, layout=a.layout, run_look=look)
        except AgentPending as e:      # agent mode: collect every clip's question, then stop once
            waiting += e.files
        except Exception as e:  # noqa: BLE001
            log(f"  clip {c['n']} failed: {str(e)[:400]}")
    if waiting:
        raise AgentPending(waiting)
    if backend() == "claude":
        log("AI cost: " + cost_line(rd))
    for c in clips:
        log(f"  #{c['n']} [{c['start']:.0f}-{c['end']:.0f}s] {c['hook']}")
    log(f"PREVIEWS READY: {len(clips)} clips for run {rid}")
    if a.review:
        review.launch_studio(open_browser=True)
    else:
        log(f"open the studio with:  {CLIP} studio")


def cmd_resume(rid, review_after=True):
    p = jload(RUNS / rid / "params.json")
    if not p:
        raise SystemExit(f"no saved options for run {rid}")
    ns = argparse.Namespace(**{"fresh": False, "whisper": None, "no_chat": False, "name": None, "layout": "auto",
                               "vision": "auto", "min": None, "max": None, "intent": None, "count": 6,
                               "preset": "earn", "review": review_after, **p})
    ns.name = rid
    ns.fresh = False
    cmd_run(ns)


def cmd_finalize(rid):
    from .publish import post_copy
    from .review import effective_clips, render_one
    rd = RUNS / rid
    set_logfile(rd / "log.txt")
    kept = [c for c in effective_clips(rid) if c.get("status") == "approved"]
    log(f"finalize {rid}: {len(kept)} kept clips")
    post_copy(rid, kept, strict=True)   # one batched question for every clip's TikTok / Instagram / YouTube text
    for c in kept:
        try:
            render_one(rid, c["n"], final=True)   # already-rendered finals are skipped, their text is refreshed
        except Exception as e:  # noqa: BLE001
            log(f"  clip {c['n']} failed: {str(e)[:400]}")
    s = jload(rd / "submission.json")
    if s:
        s["state"] = "done"
        jsave(rd / "submission.json", s)
    n = len(list((READY / rid).glob("*.mp4"))) if (READY / rid).exists() else 0
    log(f"FINALS READY: {n} videos with their post text in {READY / rid}")


def cmd_status(rid):
    rd = RUNS / rid
    if not rd.exists():
        runs = sorted((d.name for d in RUNS.glob("*") if d.is_dir()), key=lambda x: -(RUNS / x).stat().st_mtime)
        print(f"no run '{rid}'. Runs: " + (", ".join(runs[:10]) or "none"))
        return
    info = jload(rd / "info.json") or {}
    cl = jload(rd / "clips.json") or {}
    clips = cl.get("clips", [])
    prev = sum((rd / "clips" / f"{c['n']:02d}" / "preview.mp4").exists() for c in clips)
    sub = jload(rd / "submission.json")
    fin = len(list((READY / rid).glob("*.mp4"))) if (READY / rid).exists() else 0
    stages = [("download + captions", (rd / "info.json").exists()), ("transcript", (rd / "transcript.json").exists()),
              ("signals", (rd / "signals.json").exists()), ("scan", (rd / "candidates.json").exists()),
              ("clip choice", bool(clips)), (f"previews {prev}/{len(clips)}", bool(clips) and prev == len(clips))]
    print(f"run {rid}: {info.get('title', '?')}")
    for name, ok in stages:
        print(f"  [{'x' if ok else ' '}] {name}")
    q = pending(rd)
    print(f"  questions waiting for the agent: {len(q)}")
    if sub:
        print(f"  studio: {len(sub['kept'])} clips sent at {sub['at']} (state {sub['state']}); note: {sub.get('note') or '-'}")
    else:
        print("  studio: nothing sent yet")
    print(f"  finals in ready/{rid}: {fin}")
    lf = rd / "log.txt"
    if lf.exists():
        print("  last log lines:")
        for line in lf.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]:
            print("    " + line)


def print_pending(rid, files, again):
    print()
    print("=" * 72)
    print(f"AGENT INPUT NEEDED: {len(files)} question(s) for run {rid}")
    print("For EACH file below: read it, then write your JSON answer to the .answer.json path it names.")
    for f in files:
        try:
            print("  " + str(Path(f).resolve().relative_to(ROOT)))
        except ValueError:
            print("  " + str(f))
    print(f"When all are answered, run:  {again}")
    print("=" * 72)


def main(argv=None):
    p = argparse.ArgumentParser(prog="clip", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse a video and render preview clips")
    r.add_argument("source", help="YouTube/Twitch/Kick URL (anything yt-dlp supports) or a local video file")
    r.add_argument("--preset", choices=["growth", "mix", "earn"], default="earn",
                   help="growth = 18-40s, mix = 15-60s, earn = 62-95s (TikTok Creator Rewards needs >60s)")
    r.add_argument("--count", type=int, default=6)
    r.add_argument("--intent", help='e.g. "only funny moments", "rage moments", "GTA 6 reactions"')
    r.add_argument("--min", type=float)
    r.add_argument("--max", type=float)
    r.add_argument("--layout", default="auto", choices=["auto", "blur", "crop", "facecam", "native"])
    r.add_argument("--vision", default="auto", choices=["auto", "on", "off"])
    r.add_argument("--whisper", help="whisper model override, e.g. base.en (faster) or medium.en")
    r.add_argument("--no-chat", action="store_true")
    r.add_argument("--fresh", action="store_true", help="redo the AI stages (keeps downloads)")
    r.add_argument("--name", help="custom run folder name")
    r.add_argument("--review", action="store_true", help="open the review studio when done")
    rs = sub.add_parser("resume", help="continue a run with its saved options")
    rs.add_argument("run")
    st = sub.add_parser("studio", help="open the review studio (background process)")
    st.add_argument("--restart", action="store_true", help="restart it (after code updates)")
    st.add_argument("--stop", action="store_true")
    st.add_argument("--port", type=int, default=8765)
    v = sub.add_parser("review", help="run the studio server in this terminal (blocking)")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--no-open", action="store_true", help="don't open the browser")
    s = sub.add_parser("status", help="progress of a run")
    s.add_argument("run")
    q = sub.add_parser("pending", help="questions waiting for the agent")
    q.add_argument("run")
    c = sub.add_parser("costs", help="AI cost of a run (claude backend)")
    c.add_argument("run")
    f = sub.add_parser("finalize", help="full-quality kept clips + post text into ready/")
    f.add_argument("run")
    w = sub.add_parser("wait", help="block until the studio's Send button is pressed; print the submission")
    w.add_argument("run")
    w.add_argument("--timeout", type=int, default=7000)
    sub.add_parser("doctor", help="check the installation")
    a = p.parse_args(argv)
    try:
        if a.cmd == "run":
            cmd_run(a)
        elif a.cmd == "resume":
            cmd_resume(a.run)
        elif a.cmd == "studio":
            if a.stop:
                review.stop_studio(a.port)
            else:
                review.launch_studio(a.port, restart=a.restart)
        elif a.cmd == "review":
            review.serve(a.port, open_browser=not a.no_open)
        elif a.cmd == "status":
            cmd_status(a.run)
        elif a.cmd == "pending":
            fl = pending(RUNS / a.run)
            if fl:
                print_pending(a.run, fl, f"{CLIP} resume {a.run}   (or {CLIP} finalize {a.run} if the clips were already sent)")
            else:
                print("no questions waiting")
        elif a.cmd == "costs":
            print(cost_line(RUNS / a.run))
        elif a.cmd == "finalize":
            cmd_finalize(a.run)
        elif a.cmd == "doctor":
            from .doctor import doctor
            return doctor()
        elif a.cmd == "wait":
            t0 = time.time()
            while time.time() - t0 < a.timeout:
                s = jload(RUNS / a.run / "submission.json")
                if s and s.get("state") == "sent":
                    print(json.dumps(s, ensure_ascii=False))
                    return 0
                time.sleep(3)
            print("timeout")
            return 1
    except AgentPending as e:
        rid = getattr(a, "run", None) or "?"
        again = f"{CLIP} finalize {rid}" if a.cmd == "finalize" else f"{CLIP} resume {rid}"
        print_pending(rid, e.files, again)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
