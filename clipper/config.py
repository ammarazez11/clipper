"""Defaults. Override per run with CLI flags or a config.json in the project root."""
from .util import ROOT, jload

DEFAULTS = {
    # models: bulk scanning is cheap (Haiku); final judgment + copywriting on Opus
    "models": {"scan": "haiku", "vision": "haiku", "final": "opus"},
    "scan_concurrency": 4,
    "window_sec": 600,          # transcript window per Haiku call
    "agent_window_sec": 1800,   # agent mode: fewer, larger questions for the operating agent
    "window_overlap": 60,
    "whisper_model": "small.en",  # CPU int8; "base.en" is ~2x faster, less accurate
    "full_whisper_max_min": 45,   # longer videos without subtitles: transcribe signal hot-spots only
    "presets": {
        "growth": {"min": 18, "max": 40, "target": 28},
        "mix": {"min": 15, "max": 60, "target": 35},
        "earn": {"min": 62, "max": 95, "target": 75},   # TikTok Creator Rewards needs > 60s
    },
    "style": {
        "font": "Arial Black",
        "caption_size": 80,
        "hook_size": 62,
        "highlight": "&H0000F0FF",   # ASS BGR: yellow
        "uppercase": True,
        "words_per_chunk": 3,
        "hook_seconds": 0,           # 0 = hook stays on screen the whole clip
        "censor": True,              # mask swear words in burned-in captions (audio untouched)
    },
    "vision": {"every_sec": 10, "batch": 16, "max_frames": 360},
    "comments_max": 400,
    "chat": True,
}


# Genre "looks". The final model picks one per video; every field can be overridden per clip in the studio.
LOOKS = {
    "streamer": {
        "label": "Streamer / gaming reaction", "font": "Montserrat Black", "caption_size": 84, "uppercase": True,
        "words_per_chunk": 3, "primary": "&H00FFFFFF", "highlight": "&H0000E5FF", "outline": 7, "shadow": 3,
        "caption_y": 0.66, "pop": True, "hook_style": "box", "hook_font": "Montserrat Black", "hook_size": 60,
        "grade": "none", "speed": 1.0, "music": "off", "music_mood": "hype",
    },
    "motivation": {
        "label": "Motivation / self-improvement", "font": "Anton", "caption_size": 104, "uppercase": True,
        "words_per_chunk": 2, "primary": "&H00FFFFFF", "highlight": "&H0037AFD4", "outline": 4, "shadow": 5,
        "caption_y": 0.60, "pop": True, "hook_style": "serif", "hook_font": "DM Serif Display", "hook_size": 70,
        "grade": "cinematic", "speed": 1.0, "music": "auto", "music_mood": "epic",
    },
    "podcast": {
        "label": "Podcast / interview", "font": "Montserrat Black", "caption_size": 82, "uppercase": True,
        "words_per_chunk": 3, "primary": "&H00FFFFFF", "highlight": "&H0055FF4D", "outline": 7, "shadow": 2,
        "caption_y": 0.64, "pop": True, "hook_style": "box", "hook_font": "Montserrat Black", "hook_size": 60,
        "grade": "none", "speed": 1.08, "music": "off", "music_mood": "lofi",
    },
    "comedy": {
        "label": "Comedy / funny", "font": "Luckiest Guy", "caption_size": 96, "uppercase": True,
        "words_per_chunk": 2, "primary": "&H00FFFFFF", "highlight": "&H0000D7FF", "outline": 8, "shadow": 3,
        "caption_y": 0.65, "pop": True, "hook_style": "box", "hook_font": "Luckiest Guy", "hook_size": 64,
        "grade": "none", "speed": 1.0, "music": "off", "music_mood": "comedy",
    },
    "education": {
        "label": "Educational / explainer", "font": "Poppins ExtraBold", "caption_size": 74, "uppercase": False,
        "words_per_chunk": 4, "primary": "&H00FFFFFF", "highlight": "&H00F5C24F", "outline": 6, "shadow": 2,
        "caption_y": 0.66, "pop": False, "hook_style": "box", "hook_font": "Poppins ExtraBold", "hook_size": 58,
        "grade": "none", "speed": 1.05, "music": "auto", "music_mood": "chill",
    },
    "hype": {
        "label": "Sports / hype", "font": "Anton", "caption_size": 110, "uppercase": True,
        "words_per_chunk": 1, "primary": "&H00FFFFFF", "highlight": "&H003C3CFF", "outline": 6, "shadow": 4,
        "caption_y": 0.62, "pop": True, "hook_style": "box", "hook_font": "Anton", "hook_size": 66,
        "grade": "punchy", "speed": 1.0, "music": "auto", "music_mood": "hype",
    },
}
FONTS = ["Montserrat Black", "Montserrat ExtraBold", "Anton", "Bebas Neue", "Archivo Black", "Luckiest Guy",
         "Bangers", "Poppins ExtraBold", "Poppins Black", "DM Serif Display", "Arial Black", "Impact"]


def look(name, overrides=None):
    base = dict(LOOKS.get(name or "streamer", LOOKS["streamer"]))
    for k, v in (overrides or {}).items():
        if v is not None and v != "":
            base[k] = v
    return base


def load():
    cfg = jload(ROOT / "config.json", {}) or {}
    out = {**DEFAULTS}
    for k, v in cfg.items():
        out[k] = {**DEFAULTS[k], **v} if isinstance(v, dict) and isinstance(DEFAULTS.get(k), dict) else v
    return out
