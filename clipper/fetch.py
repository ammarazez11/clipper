"""Stage 1: metadata, subtitles, live chat, comments and low-bitrate audio."""
import glob
import re
from pathlib import Path

from .util import jload, jsave, log, probe_duration, run, is_url, ytdlp

KEEP = ["id", "title", "duration", "channel", "uploader", "heatmap", "chapters", "was_live", "live_status",
        "webpage_url", "view_count", "like_count", "extractor", "language", "description", "tags"]


def fetch(source, run_dir, cfg, want_chat=True, want_audio=True):
    run_dir = Path(run_dir)
    info_p = run_dir / "info.json"
    info = jload(info_p)
    if info is None:
        if is_url(source):
            log("fetch: metadata")
            p = run([*ytdlp(), "--dump-single-json", "--skip-download", "--no-warnings", source], timeout=300)
            full = __import__("json").loads(p.stdout)
            if full.get("live_status") in ("is_live", "is_upcoming") or not full.get("duration"):
                raise SystemExit("This stream is still live (or hasn't started). Wait until the VOD is published, "
                                 "or clip a finished part by downloading it first.")
            info = {k: full.get(k) for k in KEEP}
            info["source"] = "url"
            info["has_auto_subs"] = bool(full.get("automatic_captions"))
            info["has_subs"] = bool(full.get("subtitles"))
        else:
            src = str(Path(source).resolve())
            info = {"title": Path(src).stem, "duration": probe_duration(src), "local_path": src, "source": "file"}
        info["input"] = source
        jsave(info_p, info)
    if info["source"] != "url":
        return info

    url = info["webpage_url"] or source
    # subtitles (json3 keeps per-word timing for auto captions)
    if not glob.glob(str(run_dir / "src.*.json3")) and not (run_dir / "nosubs").exists():
        log("fetch: subtitles")
        lang = (info.get("language") or "en").split("-")[0]
        langs = f"{lang}-orig,{lang},en-orig,en" if lang != "en" else "en-orig,en,en-US,en-GB"
        run([*ytdlp(), "--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", langs,
             "--sub-format", "json3", "--no-warnings", "-o", str(run_dir / "src.%(ext)s"), url],
            timeout=300, check=False)
        if not glob.glob(str(run_dir / "src.*.json3")):
            (run_dir / "nosubs").write_text("")
    # live chat replay for streamed VODs (YouTube)
    if want_chat and info.get("live_status") == "was_live" and not (run_dir / "src.live_chat.json").exists() \
            and not (run_dir / "nochat").exists():
        log("fetch: live chat replay (can take a few minutes on long streams)")
        run([*ytdlp(), "--skip-download", "--write-subs", "--sub-langs", "live_chat", "--no-warnings",
             "-o", str(run_dir / "src.%(ext)s"), url], timeout=3600, check=False)
        part = run_dir / "src.live_chat.json.part"
        if not (run_dir / "src.live_chat.json").exists() and part.exists() and part.stat().st_size > 0:
            part.replace(run_dir / "src.live_chat.json")   # partial chat is still useful
        if not (run_dir / "src.live_chat.json").exists():
            (run_dir / "nochat").write_text("")
    # top comments (timestamp mentions = free "this moment is good" votes)
    if not (run_dir / "comments.json").exists():
        log("fetch: top comments")
        n = cfg["comments_max"]
        p = run([*ytdlp(), "--skip-download", "--write-comments", "--dump-single-json", "--no-warnings",
                 "--extractor-args", f"youtube:max_comments={n},{n},0,0;comment_sort=top", url],
                timeout=300, check=False)
        comments = []
        try:
            d = __import__("json").loads(p.stdout)
            comments = [{"text": c.get("text", ""), "likes": c.get("like_count") or 0}
                        for c in (d.get("comments") or [])]
        except Exception:  # noqa: BLE001
            pass
        jsave(run_dir / "comments.json", comments)
    # low-bitrate audio: loudness signal + whisper fallback
    if want_audio and not glob.glob(str(run_dir / "audio.*")):
        log("fetch: audio track")
        run([*ytdlp(), "-f", "ba[abr<=96]/ba/b", "--no-warnings", "--no-playlist", "-o",
             str(run_dir / "audio.%(ext)s"), url], timeout=7200)
    return info


def audio_path(run_dir, info):
    if info["source"] == "file":
        return info["local_path"]
    hits = [h for h in glob.glob(str(Path(run_dir) / "audio.*")) if not re.search(r"\.(part|ytdl)$", h)]
    return hits[0] if hits else None
