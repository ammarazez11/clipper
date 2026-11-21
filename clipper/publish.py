"""Per-platform post copy for each finished clip: TikTok caption, Instagram caption, YouTube Shorts title + description.

One Opus call per batch (the copy is judgment work, the input is tiny). Cached in runs/<id>/post_copy.json and
regenerated only when a clip's hook/caption/hashtags/trim change.
"""
from . import config
from .llm import AgentPending, ask
from .util import RUNS, jload, jsave, log

SCHEMA = {
    "type": "object",
    "properties": {"posts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "tiktok": {"type": "string"},
            "instagram": {"type": "string"},
            "youtube_title": {"type": "string"},
            "youtube_description": {"type": "string"},
        },
        "required": ["n", "tiktok", "instagram", "youtube_title", "youtube_description"]}}},
    "required": ["posts"],
}

RULES = """You write the post copy for short vertical clips on a GENERAL internet-clips channel: the best moments from
whatever the internet is watching (streams, podcasts, viral videos). It is not a single-game or single-creator channel,
so write each post for a stranger scrolling past, in natural English. Every clip goes to TikTok, Instagram Reels and
YouTube Shorts. Ground every claim in the clip info below: never invent things that don't happen in the clip.

For every clip return:
- tiktok: one line, max 100 characters, that makes people stay for the payoff without spoiling it. TikTok is a search
  engine: name the creator and the topic in plain words people would type. At most one emoji. Then a blank line, then
  3-5 lowercase hashtags: the creator, the topic, one niche tag, one broad tag (#gaming, #podcast, #funny...).
  Never use #fyp, #foryou, #foryoupage, #viral or #trending.
- instagram: line 1 is a hook of max 80 characters (it is all people see before "more"); line 2 is one short line of
  context or a question that invites comments; line 3 is the credit, exactly "🎥 {credit}"; then a blank line and
  5-8 lowercase hashtags (same rules, a little broader).
- youtube_title: max 70 characters, searchable: creator name + what happens, curiosity without lying. No hashtags,
  at most one fully capitalised word, no clickbait that the clip doesn't deliver.
- youtube_description: 1-2 short sentences about the moment, then a blank line, then "Full stream: {url}" and
  "Credit: {credit}" on their own lines, then a blank line and 3 hashtags on one line, the first being #shorts.
"""


def _key(c):
    return [c.get("hook"), c.get("caption"), list(c.get("hashtags") or []), round(c["start"], 1), round(c["end"], 1)]


def _transcript(rid, n):
    w = jload(RUNS / rid / "clips" / f"{n:02d}" / "words.json") or {}
    text = " ".join((x.get("text") or x.get("word") or "").strip() for x in w.get("words", []))
    return text[:700]


def fallback(c):
    tags = " ".join(c.get("hashtags") or [])
    cap = (c.get("caption") or c.get("hook") or "").strip()
    return {"tiktok": f"{cap}\n\n{tags}", "instagram": f"{cap}\n\n{tags}",
            "youtube_title": (c.get("title") or c.get("hook") or "")[:70],
            "youtube_description": f"{cap}\n\n#shorts {' '.join((c.get('hashtags') or [])[:2])}"}


def post_copy(rid, clips, cfg=None, allow_llm=True, strict=False):
    """Return {n: copy} for the given clips, generating whatever is missing or stale in one call.
    allow_llm=False only reads the cache (draft text otherwise); strict re-raises agent-mode questions."""
    cfg = cfg or config.load()
    rd = RUNS / rid
    cache = jload(rd / "post_copy.json", {})
    todo = [c for c in clips if (cache.get(str(c["n"])) or {}).get("key") != _key(c)]
    if todo and allow_llm:
        info = jload(rd / "info.json", {})
        credit = info.get("channel") or info.get("uploader") or "the original creator"
        url = info.get("webpage_url") or info.get("input") or ""
        lines = []
        for c in todo:
            lines.append(f"--- clip {c['n']} ({c['end'] - c['start']:.0f}s, mood {c.get('mood', '?')})\n"
                         f"hook on screen: {c.get('hook', '')}\ntitle: {c.get('title', '')}\n"
                         f"draft caption: {c.get('caption', '')}\ndraft hashtags: {' '.join(c.get('hashtags') or [])}\n"
                         f"why it works: {c.get('why', '')}\nwhat is said: {_transcript(rid, c['n'])}")
        prompt = (RULES.replace("{credit}", credit).replace("{url}", url)
                  + f"\nSource video: \"{info.get('title', '')}\" by {credit} ({url})\n\n" + "\n\n".join(lines))
        log(f"post copy: writing TikTok / Instagram / YouTube text for {len(todo)} clip(s)")
        try:
            res = ask(prompt, SCHEMA, cfg["models"]["final"], rd, "copy", timeout=600)
            by_n = {p["n"]: p for p in res.get("posts", [])}
            for c in todo:
                p = by_n.get(c["n"])
                if p:
                    cache[str(c["n"])] = {"key": _key(c), **{k: p[k].strip() for k in SCHEMA["properties"]["posts"]["items"]["required"] if k != "n"}}
            jsave(rd / "post_copy.json", cache)
        except AgentPending:
            if strict:
                raise
        except Exception as e:  # noqa: BLE001
            log(f"  post copy failed, using the draft caption: {str(e)[:200]}")
    out = {}
    for c in clips:
        got = cache.get(str(c["n"]))
        out[c["n"]] = got if got and got.get("key") == _key(c) else fallback(c)
    return out


def post_text(copy):
    return ("TIKTOK\n------\n" + copy["tiktok"].strip()
            + "\n\n\nINSTAGRAM\n---------\n" + copy["instagram"].strip()
            + "\n\n\nYOUTUBE SHORTS\n--------------\nTitle:\n" + copy["youtube_title"].strip()
            + "\n\nDescription:\n" + copy["youtube_description"].strip() + "\n")
