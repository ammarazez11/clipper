"""LLM calls, with two interchangeable backends.

claude  headless Claude Code (uses the logged-in Claude Code subscription, no API key). Prompts go over stdin,
        the default system prompt and MCP servers are stripped to keep per-call overhead small, costs are logged.
agent   no API at all: the AI agent that operates this project (Antigravity, Cursor, Codex, Claude Code, ...) is the
        model. Each question is written to runs/<id>/agent/<stage>-<hash>.request.md; the pipeline stops with
        "AGENT INPUT NEEDED"; the agent writes the matching .answer.json and re-runs the same command, which resumes
        from its caches. See AGENTS.md.

Pick one with "llm": "auto" | "claude" | "agent" in config.json (or env CLIPPER_LLM). auto = claude if the
Claude Code CLI is installed, otherwise agent.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .util import ROOT, jload, log

SYSTEM = ("You are a senior short-form video editor and viral clip strategist. "
          "Answer only through the requested JSON structure. Be decisive and concrete.")
FENCE = chr(96) * 3


class AgentPending(Exception):
    """Raised in agent mode when one or more questions are waiting for the operating agent."""

    def __init__(self, files):
        self.files = list(files)
        super().__init__(f"{len(self.files)} question(s) for the agent")


def _claude_bin():
    cands = [Path(os.environ.get("APPDATA", "")) / "npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe"]
    for c in cands:
        if c.exists():
            return str(c)
    return shutil.which("claude")


_backend = None


def backend():
    global _backend
    if _backend is None:
        want = (os.environ.get("CLIPPER_LLM") or (jload(ROOT / "config.json", {}) or {}).get("llm") or "auto").lower()
        _backend = ("claude" if _claude_bin() else "agent") if want == "auto" else want
    return _backend


def _record(run_dir, rec):
    with open(Path(run_dir) / "costs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def ask(prompt, schema, model, run_dir, stage, tools=None, add_dirs=(), timeout=900, retries=2):
    if backend() == "agent":
        return _agent_ask(prompt, schema, run_dir, stage)
    return _claude_ask(prompt, schema, model, run_dir, stage, tools, add_dirs, timeout, retries)


def _claude_ask(prompt, schema, model, run_dir, stage, tools, add_dirs, timeout, retries):
    claude = _claude_bin()
    if not claude:
        raise RuntimeError('Claude Code CLI not found; set "llm": "agent" in config.json')
    cmd = [claude, "-p", "--model", model, "--output-format", "json",
           "--system-prompt", SYSTEM, "--strict-mcp-config", "--no-session-persistence",
           "--json-schema", json.dumps(schema)]
    if tools:
        cmd += ["--tools", ",".join(tools), "--allowedTools", ",".join(tools)]
        for d in add_dirs:
            cmd += ["--add-dir", str(d)]
    else:
        cmd += ["--tools", ""]
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout, cwd=tempfile.gettempdir())
            out = p.stdout
            data = json.loads(out[out.index("{"):])
            if data.get("is_error") or data.get("structured_output") is None:
                raise RuntimeError(f"claude error: {data.get('subtype')} {str(data.get('result'))[:300]}")
            u = data.get("usage", {})
            _record(run_dir, {"stage": stage, "model": model, "cost_usd": data.get("total_cost_usd", 0),
                              "in": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                              + u.get("cache_read_input_tokens", 0),
                              "out": u.get("output_tokens", 0), "sec": round(time.time() - t0, 1)})
            return data["structured_output"]
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"  llm {stage} attempt {attempt + 1} failed: {str(e)[:200]}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"LLM stage {stage} failed: {last}")


# ---------------------------------------------------------------- agent mode

def validate(obj, schema, path="answer"):
    """Small JSON-schema check (type / required / properties / items / enum / min / max). Returns error strings."""
    errs = []
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}
    if t in types:
        ok = isinstance(obj, types[t]) and not (t in ("integer", "number") and isinstance(obj, bool))
        if t == "integer" and isinstance(obj, float) and obj.is_integer():
            ok = True
        if not ok:
            return [f"{path} must be {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path} must be one of {schema['enum']}, got {obj!r}")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path} must be <= {schema['maximum']}")
    if t == "object":
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path} is missing \"{k}\"")
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errs += validate(obj[k], sub, f"{path}.{k}")
    if t == "array" and "items" in schema:
        for i, x in enumerate(obj):
            errs += validate(x, schema["items"], f"{path}[{i}]")
    return errs


def _skeleton(schema):
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        return {k: _skeleton(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_skeleton(schema.get("items", {}))]
    return {"string": "...", "integer": 0, "number": 0.0, "boolean": False}.get(t)


def _agent_ask(prompt, schema, run_dir, stage):
    qdir = Path(run_dir) / "agent"
    qdir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1((stage + "\n" + prompt + json.dumps(schema, sort_keys=True)).encode("utf-8")).hexdigest()[:10]
    req, ans = qdir / f"{stage}-{h}.request.md", qdir / f"{stage}-{h}.answer.json"
    problem = ""
    if ans.exists():
        try:
            data = json.loads(ans.read_text(encoding="utf-8-sig"))
            errs = validate(data, schema)
        except Exception as e:  # noqa: BLE001
            data, errs = None, [f"not valid JSON: {e}"]
        if not errs:
            _record(run_dir, {"stage": stage, "model": "agent", "cost_usd": 0, "in": 0, "out": 0, "sec": 0})
            req.unlink(missing_ok=True)
            return data
        problem = ("\n## Your previous answer was rejected\n\n" + "\n".join(f"- {e}" for e in errs[:20])
                   + "\n\nFix it and write the corrected JSON to the same file.\n")
        ans.replace(ans.with_suffix(".rejected.json"))
    req.write_text(
        f"# Question for the agent ({stage})\n\n"
        "Answer as the expert described below. Write ONLY a JSON object to this file (UTF-8, no markdown fences):\n\n"
        f"    {ans.resolve()}\n\n"
        "It must match the JSON schema at the end. When every pending question is answered, re-run the same "
        "clip command; it continues where it stopped.\n"
        "If the prompt lists image files, open and look at each of them before answering.\n"
        f"{problem}\n## Role\n\n{SYSTEM}\n\n## Prompt\n\n{prompt}\n\n## JSON schema\n\n{FENCE}json\n"
        f"{json.dumps(schema, indent=1)}\n{FENCE}\n\n## Shape example (values are placeholders)\n\n{FENCE}json\n"
        f"{json.dumps(_skeleton(schema), indent=1)}\n{FENCE}\n", encoding="utf-8")
    raise AgentPending([req])


def pending(run_dir):
    d = Path(run_dir) / "agent"
    return sorted(d.glob("*.request.md")) if d.exists() else []


def cost_summary(run_dir):
    f = Path(run_dir) / "costs.jsonl"
    if not f.exists():
        return {}
    agg = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        a = agg.setdefault(r["model"], {"calls": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0})
        a["calls"] += 1
        a["cost_usd"] += r["cost_usd"]
        a["tokens_in"] += r["in"]
        a["tokens_out"] += r["out"]
    for a in agg.values():
        a["cost_usd"] = round(a["cost_usd"], 4)
    return agg
