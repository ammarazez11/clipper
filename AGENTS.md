# Operating manual for the AI agent

You operate this project for its owner. Read this whole file at the start of every session, and follow it.

The project turns a long video (YouTube, Twitch and more) into short vertical clips for TikTok, Instagram Reels and YouTube Shorts:
1. It finds the best moments.
2. It writes the hooks and captions.
3. It renders previews that the user reviews in a local "studio" web page.
4. It renders the finished clips into `ready/<run>/`, each with ready-to-paste post text for all three platforms.

## 1. Who you are working for

- **The user is not technical.** Never ask them to type commands, edit files, install anything or read logs. You do all of that yourself in the terminal. Ask them only for things only they can decide, such as which video, how many clips, or what kind of moments.
- **Speak the user's language.** Many users write in Arabic. Answer in the same language and dialect, in short, simple sentences. Avoid technical words; when you must use one, explain it in a few words.
- **All content is in English.** Hooks, captions, hashtags and post texts are for a global English-speaking audience, whatever language the user speaks to you. When you pass the user's wishes to the tool (`--intent`), translate them into clear English and keep every detail: counts, lengths, people, topics, moods.
- **Keep them informed.** Before a long step, say what will happen and roughly how long it takes. Afterwards, say what is ready and what they should do next.

## 2. Terminal rules

- **The command.**
  - Windows: always run `.\clip.bat <command>` from the project folder.
  - Never type a bare `clip`: on Windows that is the clipboard tool.
  - macOS/Linux: `./clip.sh <command>`.
- **Long commands.** Downloading and transcribing a long stream can take 5–40 minutes. Every stage is cached in `runs/<run>/`.
  - If the terminal seems stuck or gets cut off, don't start a second copy. Check progress with `.\clip.bat status <run>`. It also shows the last lines of `runs/<run>/log.txt`.
  - `.\clip.bat resume <run>` continues exactly where a run stopped.
- **The studio.** It is a small local web server that runs in the background. `.\clip.bat studio` starts it if needed and opens http://127.0.0.1:8765 in the browser. The command returns immediately; don't wait on it. After a code update, use `.\clip.bat studio --restart`.

## 3. First session: setup

1. Run `.\clip.bat doctor`.
2. If it fails, or `.venv` doesn't exist yet, run `.\setup.bat`.
   - It installs Python 3.11, FFmpeg and Deno with winget, then the Python packages and the models.
   - It downloads about 1.5 GB and takes 5–20 minutes. Windows may show install prompts the user has to accept; tell them so first.
   - Then run `doctor` again. Every line must say OK (WARN is fine).
3. Tell the user, in their language, that everything is ready and they can send a video link.

## 4. The daily flow

### Step 1: the user sends a link and what they want

Turn the request into options:

| The user says | Option |
| --- | --- |
| how many clips ("give me 5") | `--count 5` (default 6) |
| short clips / for growth (18–40 s) | `--preset growth` |
| medium (15–60 s) | `--preset mix` |
| over a minute / for TikTok Creator Rewards (62–95 s) | `--preset earn` |
| exact lengths ("between 20 and 45 seconds") | `--min 20 --max 45`, together with the closest preset |
| what kind of moments, who, what topic | `--intent "..."`, written in English and specific |
| webcam on top of gameplay / full frame / follow the face | `--layout facecam` / `blur` / `crop` (default `auto` decides by itself) |

- If the user didn't say a length, use `--preset growth`.
- If something important is unclear, ask **one** short question; otherwise just start.
- Live streams can only be clipped after they end.

### Step 2: run it

```
.\clip.bat run "<URL>" --preset growth --count 6 --intent "<English description>" --review
```

If the output ends with **AGENT INPUT NEEDED**, answer the questions (section 5), then run the command it prints (`.\clip.bat resume <run>`). Repeat until it prints **PREVIEWS READY**. The studio then opens by itself.

### Step 3: the user reviews

Tell the user, in their language:
1. The studio is open in the browser (http://127.0.0.1:8765).
2. Press **Y** to keep a clip and **N** to cut it. Trim, change the hook, captions, look or framing if they want.
3. Press **Send kept clips**, optionally with a note.
4. Come back to you and say they are done.

Pressing Send already starts the full-quality renders in the studio (1–3 minutes per clip).

### Step 4: finish

When the user says they're done:
1. Run `.\clip.bat status <run>`. It shows how many clips were sent and the user's note.
2. If there is a note, or the user asked for changes in the chat, apply them (section 6).
3. Run `.\clip.bat finalize <run>`.
   - In agent mode it first asks you to write the post texts; answer them as in section 5.
   - Then run `.\clip.bat finalize <run>` again, until it prints **FINALS READY**.
   - Clips that are already rendered are not rendered again; only their text files are updated.
4. Tell the user where the videos are, `ready\<run>\`, and open the folder for them (`explorer ready\<run>`).
5. Explain that each video has a `.txt` file next to it with three parts:
   - **TIKTOK:** caption plus hashtags.
   - **INSTAGRAM:** caption, a question for comments, a credit line and hashtags.
   - **YOUTUBE SHORTS:** a title and a description.

   Tell them to post the same video on all three platforms: TikTok first, the other two an hour or two later.

## 5. Answering questions (agent mode)

When the Claude Code CLI isn't installed, this project uses **you** as its AI ("agent mode"; `doctor` shows the backend). The pipeline writes each question to a file and stops with **AGENT INPUT NEEDED** and a list of files.

**For every listed file:**
1. Open `runs/<run>/agent/<stage>-<id>.request.md`. It contains:
   - the role to play;
   - the full prompt;
   - the JSON schema;
   - an example of the JSON shape;
   - the exact path of the answer file.
2. Do the task with your best judgment, as the expert described.
3. Write **only** the JSON object, as UTF-8 with no code fences and no comments, to the `.answer.json` path given in the file.

When every file is answered, run the command printed at the end (`resume` or `finalize`). `.\clip.bat pending <run>` lists the questions still open.

**If an answer is rejected:** the request file comes back with a "rejected" section listing exactly what was wrong. Fix it and write the file again.

**The stages you'll see:**

| Stage | What you do | How many |
| --- | --- | --- |
| `scan` | Find clip-worthy moments in one ~30-minute part of the transcript. | One per part |
| `final` | Choose the best clips from the candidates and write the hook, caption, hashtags, look and cold-open moment. | One |
| `analyze` | Open 3 frames from one clip and say whether the video already has subtitles burned into the picture. | One per clip |
| `vision` | Describe video frames. Only for videos with very little talking. | Several |
| `copy` | Write the TikTok, Instagram and YouTube text for each kept clip. | One, at finalize |

**Rules:**
- **Timestamps.** Use the `[seconds]` numbers from the transcript lines. Never invent times outside the part you were given.
- **Quality.** `scan`, `final` and `copy` decide how good the channel is, so take your time on them. A great clip:
  - makes sense to a stranger with no context;
  - has its hook in the first second;
  - ends right after the payoff.
- **Images.** When a prompt lists image files, open and look at each one. If you really cannot open images, give `analyze` answers with `"subtitles": false`, `"region": "none"` and a short `"scene"` guess, and say so to the user.
- **Correctness.** Don't change numbers or enums into other types. Follow the schema exactly.

## 6. Changing clips yourself

The studio is the normal way. If the user asks you in the chat ("make clip 3 start 2 seconds later", "remove the music", "change the hook of clip 2"), edit `runs/<run>/decisions.json`. Keys are clip numbers as strings, for example `{"3": {"start": 1234.5}}`. Useful fields:

| Field | Meaning |
| --- | --- |
| `status` | `"approved"` or `"rejected"` |
| `start`, `end` | Seconds in the source video |
| `hook` | Text shown on screen at the start |
| `caption`, `hashtags` | Draft post text; the final text is written at finalize |
| `look` | `streamer`, `motivation`, `podcast`, `comedy`, `education` or `hype` |
| `captions` | `"auto"`, `"on"` or `"off"` |
| `teaser` | Cold open, on or off |
| `speed` | e.g. `1.1` |
| `music` | `"auto"`, `"on"` or `"off"` |
| `layout` | `facecam`, `blur`, `crop` or `native` |

Then run `.\clip.bat finalize <run>`. Anything that changes the picture is re-rendered automatically.

## 7. Never

- Never post to TikTok, Instagram or YouTube, and never log in to the user's accounts. The user posts by hand.
- Never delete `runs/` or `ready/` folders without asking. Old `runs/<run>` folders can be large (several GB), so offer to remove old ones when disk space gets low (`doctor` shows it).
- Never change the code unless something is clearly broken. Keep any fix small, and restart the studio afterwards.
- Never commit or push to git.

## 8. Updating

- Run `git pull` in the project folder. If `requirements.txt` changed, run `.\setup.bat` again.
- Then run `.\clip.bat studio --restart`.

## 9. Troubleshooting

| Symptom | What to do |
| --- | --- |
| "Sign in to confirm you're not a bot", HTTP 403, age-restricted video | The download needs the user's YouTube login cookies. If the user has Firefox signed in to YouTube, create `config.json` containing `{"cookies_from_browser": "firefox"}`. Otherwise ask them to install the "Get cookies.txt LOCALLY" browser extension, export youtube.com cookies, and save the file as `cookies.txt` in the project folder. Never share that file. |
| Downloads fail after working before | YouTube changed something. Run `.venv\Scripts\python -m pip install -U "yt-dlp[default]"`. |
| Studio page doesn't load | `.\clip.bat studio --restart`; the log is `runs/studio.log`. |
| A step crashed | `.\clip.bat status <run>`, read the log lines, fix the cause, then `.\clip.bat resume <run>`. |
| `doctor` shows FAIL | Run `.\setup.bat` again. If FFmpeg or Deno is still missing, close and reopen the terminal, since PATH updates need a new terminal, and run doctor again. |

## 10. How it works (for you, not for the user)

`fetch` (metadata, captions, live-chat replay, comments, audio) → `transcript` (YouTube captions or local Whisper) → `signals` (replay heatmap, chat bursts, comment timestamps, loudness) → `scan` (AI) → `final` (AI) → `render` (previews).

Rendering does a lot by itself:
- burned-in subtitle check;
- music-bed check;
- webcam detection and stacking;
- black-bar removal;
- a cold-open teaser that replays the peak moment first;
- word-by-word captions inside TikTok's safe zone;
- loudness set to -14 LUFS.

Finals use the source's best resolution, up to 4K, so webcams stay sharp.

Code is in `clipper/`; the studio is `clipper/review.py` plus `review.html`; settings are `clipper/config.py`, which a `config.json` in the project root can override (for example `{"llm": "agent"}`). Royalty-free music can go in `music/<mood>/`. Without tracks no music is added; trending sounds are better added inside the TikTok app.
