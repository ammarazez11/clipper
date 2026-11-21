# Clipper

Turn a long video (stream VOD, podcast, any YouTube/Twitch link) into reviewed, ready-to-post vertical clips for TikTok, Instagram Reels and YouTube Shorts.

You give an AI agent a link and say what you want. The agent runs the pipeline. You pick and tweak clips in a local studio. The finished clips land in `ready/<run>/` with post text for every platform.

## How it works

1. **Find moments**
   - Captions or local Whisper speech-to-text.
   - Viewer signals: replay heatmap, live-chat bursts, comment timestamps and loudness.
   - An AI scan of the whole video, then a final pick with hooks, captions and hashtags.
2. **Render previews**
   - 9:16 reframing: webcam stacked over gameplay, face-following crop, or blurred fill.
   - Word-by-word captions in the TikTok safe zone, in a look chosen per video (streamer, motivation, podcast, comedy, education, hype).
   - A cold-open teaser that replays the peak moment first, then the clip from its start.
   - Loudness set to -14 LUFS.
3. **Review in the studio** (http://127.0.0.1:8765)
   - **Keyboard:** Y keep, N cut, arrow keys to move between clips.
   - **Editing:** trim, change the hook, captions, look or framing.
   - **Send kept clips:** renders the full-quality finals.
4. **Finals**
   - 1080x1920, using the source's best resolution (up to 4K) so small webcams stay sharp.
   - A `.txt` next to each video with **TikTok**, **Instagram** and **YouTube Shorts** (title + description) text.

It adapts to the source automatically:
- If the source already has burned-in subtitles, its own captions are turned off.
- If the source already has background music, no music is added.
- Black bars are removed.
- Footage that is already dark is not darkened further.

## Setup (Windows 10/11)

Run `setup.bat`. It installs Python 3.11, FFmpeg and Deno with winget, creates `.venv`, installs the packages, downloads the models, and finishes with `clip.bat doctor`.

On macOS/Linux: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`, install `ffmpeg` and `deno`, then use `./clip.sh`.

## AI: your agent is the model

- **Claude Code installed:** the pipeline calls it headlessly. A cheap model does the bulk scanning and a strong model makes the final picks.
- **Any other agent** (Antigravity, Cursor, Codex...): "agent mode". Questions are written to `runs/<run>/agent/*.request.md`, the agent answers them as JSON, and the run resumes. No API key needed.

`AGENTS.md` is the full operating manual the agent follows. Force a backend with `{"llm": "agent"}` or `{"llm": "claude"}` in `config.json`.

## Commands

```
clip.bat run URL [--preset growth|mix|earn] [--count N] [--min S --max S] [--intent "..."] [--layout auto|facecam|blur|crop] --review
clip.bat resume RUN      clip.bat status RUN      clip.bat pending RUN
clip.bat studio [--restart]                       clip.bat finalize RUN      clip.bat doctor
```

Presets: growth 18-40 s, mix 15-60 s, earn 62-95 s (TikTok Creator Rewards needs clips over a minute).

## Notes

- **Music:** put royalty-free tracks in `music/<mood>/`. Without tracks, nothing is added.
- **Brand images:** `tools/brand.py` renders a YouTube banner and a profile picture.
- **Posting is manual.** Respect the original creators: the generated descriptions credit and link the source.
- **Fonts** are from Google Fonts; see `fonts/licenses/`.
