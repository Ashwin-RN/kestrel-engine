"""
LLM adapter: one thin interface over every model provider the system uses.

Provider paths:
- `AnthropicLLM`: the Anthropic API directly (headless / batch work).
- The Claude Code CLI as a subprocess, in three shapes: a project agent run by
  name (`--agent`, its definition synced into an isolated directory first);
  prompt mode, with an explicit model and a system prompt passed by FILE (the
  council's Claude seat); and `ClaudeCodeCLI`, the generic adapter. Every
  spawn runs from an isolated working directory outside the repo.
- `CodexCLI`: the OpenAI Codex CLI in exec mode over ChatGPT OAuth, in a
  read-only sandbox. A council critic seat, and the twin every agent falls
  back to on a Claude rate limit.
- OpenRouter: the bulk graders' primary tier (an OpenAI-compatible HTTP API).
- `GeminiLLM`: Google Gemini over REST with one key, a bulk fallback tier.
- `OllamaLLM`: local models, the offline fallback.

The bulk ladder in one sentence: a bulk agent (Stage 2 to 4 and the sector
outlook) walks the primary model line (a GLM model on OpenRouter), then the
fallback line (Gemini pinned to one model), then Claude; any agent falls from
Claude to its Codex twin on a rate limit, except that a weekly cap on the
protected portfolio synthesizer raises unless the operator has armed an escape
hatch to its Codex twin (armed in the live system);
every served row is stamped with the provider and model that served it.

Isolation: the CLIs load instruction files from the working directory and its
ancestors, so every spawn runs from `claude_cli_cwd()`, outside the repo. It
is NOT a full sandbox: the user-scope instruction file still loads from any
working directory, and the `--agent` path does not pass a strict-MCP flag, so
account-level connectors still attach to it. Only prompt mode passes it.

Surface: `generate(prompt, system=...)` returns text; `generate_json(...)`
returns parsed JSON (retry once on parse failure); Anthropic supports prompt
caching via `cached_system=True`.

Public edition: an excerpt. The ladder's model lines, the Codex model and
effort, the rate-limit park slice and cap, the breaker counts and the debug
switch are withheld (module-level None constants, each behind a guard).
Agent prompts are not published (agents are named only), nor are
`data.openai_compat` (the OpenAI-compatible client) and `data.bulk_provider`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from core import config

log = logging.getLogger(__name__)


# --- Withheld selections and knobs ---
# The bulk ladder's two model lines (live: a GLM slug on OpenRouter, then a
# Gemini model id; Claude is tier 3 in code). Both None: straight to Claude.
BULK_PRIMARY_MODEL: Optional[str] = None  # withheld from the public edition
BULK_FALLBACK_MODEL: Optional[str] = None  # withheld from the public edition
# Codex model and reasoning effort. None: the CLI's own default, no override.
CODEX_MODEL: Optional[str] = None  # withheld from the public edition
CODEX_REASONING_EFFORT: Optional[str] = None  # withheld from the public edition
# Claude session-limit parking: one park's length and the total-wait ceiling.
# Either None: a session limit raises RateLimitError at once (Codex fallback).
SUBAGENT_RL_SLEEP_S: Optional[int] = None  # withheld from the public edition
SUBAGENT_RL_MAX_TOTAL_S: Optional[int] = None  # withheld from the public edition
# Escape hatch letting the protected synthesizer fall to Codex on a weekly
# cap. None: refuse the silent swap.
SYNTH_ALLOW_CODEX: Optional[bool] = None  # withheld from the public edition
# Directory for full failing-payload dumps. None: no dumps.
SUBAGENT_DEBUG_DIR: Optional[str] = None  # withheld from the public edition
# OpenRouter breaker: lost calls before a timed bench, and the doubling bench
# steps (seconds). Either None: never benches (the hard-refusal latch works).
BULK_BENCH_AFTER_LOST: Optional[int] = None  # withheld from the public edition
BULK_BENCH_STEPS_S: Optional[tuple] = None  # withheld from the public edition

_JSON_INSTRUCTION = "\n\nRespond with a single JSON object. No prose, no markdown fences."


def _err_excerpt(text: str, head: int = 400, tail: int = 400) -> str:
    """Excerpt an error body keeping BOTH ends, never just the head.

    Provider error bodies often put boilerplate first and the discriminating
    fields (a quota id, a status token) last. Text that fits is returned
    unchanged; otherwise head and tail are joined by a classifier-NEUTRAL
    marker (if the dropped count would render a status code, one more head
    character shifts it). Never longer than the input; never raises.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    head, tail = max(int(head), 0), max(int(tail), 0)
    if len(text) <= head + tail:
        return text
    for h in (head, head + 1):
        marker = f"\n...[{len(text) - h - tail} chars elided]...\n"
        if "429" not in marker:
            break
    candidate = text[:h] + marker + (text[len(text) - tail:] if tail else "")
    return candidate if len(candidate) < len(text) else text


def _json_retry(generate, prompt: str, system: Optional[str], label: str) -> dict:
    """The shared prompt-engineered JSON path: ask for one object, retry once."""
    raw = ""
    for attempt in range(2):
        raw = generate(prompt + _JSON_INSTRUCTION, system=system)
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed
        log.warning("%s returned unparseable JSON (attempt %d): %r", label, attempt + 1, raw[:200])
    raise ValueError(f"{label} failed to produce valid JSON after 2 attempts: {raw[:300]}")


# --- Anthropic (direct API) ---

class AnthropicLLM:
    def __init__(self, model: str, api_key: str):
        # Lazy import so Ollama-only users don't need the anthropic package.
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024,
                 temperature: float = 0.0, cached_system: bool = False) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            if cached_system:
                kwargs["system"] = [{"type": "text", "text": system,
                                     "cache_control": {"type": "ephemeral"}}]
            else:
                kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        # Concatenate all text blocks (Claude may split responses)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def generate_json(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048,
                      temperature: float = 0.0, cached_system: bool = False) -> dict:
        def _gen(p, system=None):
            return self.generate(p, system=system, max_tokens=max_tokens,
                                 temperature=temperature, cached_system=cached_system)
        return _json_retry(_gen, prompt, system, "anthropic")


# --- Claude CLI rate limits ---
# One detector shared by ClaudeCodeCLI and the subagent paths. STRICT patterns
# are machine-shaped (CLI wrapper JSON), safe to scan anywhere. LOOSE patterns
# are human phrasings: they count only in stderr, or in stdout when the call
# visibly failed, so a dossier that merely MENTIONS "rate limit" cannot park.
_RL_STRICT_PATTERNS = ('api_error_status":429', '"status":"limit_exceeded"')
_RL_LOOSE_PATTERNS = ("you've hit your limit", "5-hour limit", "usage limit",
                      "rate limit", "weekly limit")
_RATE_LIMIT_PATTERNS = _RL_STRICT_PATTERNS + _RL_LOOSE_PATTERNS
# "week" within this many chars of the hit makes it WEEKLY (scoped, so a
# "52-week low" elsewhere in a large stdout cannot flip the verdict).
_RL_WEEKLY_WINDOW = 150
# Sanity ceiling on a parsed reset hint: the session window plus a margin.
_RL_HINT_CAP_S = 5 * 3600 + 1800
# One log line per slice while parked.
_PARK_HEARTBEAT_S = 300.0


@dataclass(frozen=True)
class RateLimitSignal:
    """A classified Claude CLI rate-limit hit. kind: "session" (the rolling
    window: parking works) or "weekly" (resets days away: fail fast or fall
    back). reset_hint_s: the parsed hint, if any. matched: the pattern."""
    kind: str
    reset_hint_s: Optional[int]
    matched: str


def classify_rate_limit(stdout_text: str, stderr_text: str, *,
                        returncode: int = 1) -> Optional[RateLimitSignal]:
    """Single source of truth for Claude CLI rate-limit detection."""
    stdout_lc = (stdout_text or "").lower()
    stderr_lc = (stderr_text or "").lower()
    stdout_failed = (returncode != 0
                     or '"is_error":true' in re.sub(r"\s+", "", stdout_lc))

    def _find(patterns: tuple, text: str):
        for p in patterns:
            pos = text.find(p)
            if pos >= 0:
                return p, pos, text
        return None

    hit = (_find(_RL_STRICT_PATTERNS, stderr_lc)
           or _find(_RL_STRICT_PATTERNS, stdout_lc)
           or _find(_RL_LOOSE_PATTERNS, stderr_lc)
           or (_find(_RL_LOOSE_PATTERNS, stdout_lc) if stdout_failed else None))
    if hit is None:
        return None
    matched, pos, text = hit
    lo = max(0, pos - _RL_WEEKLY_WINDOW)
    kind = "weekly" if "week" in text[lo:pos + len(matched) + _RL_WEEKLY_WINDOW] else "session"
    # The hint is parsed from the region around the hit, not the whole blob.
    hint = _parse_claude_rate_limit_sleep(text[lo:pos + len(matched) + 400])
    return RateLimitSignal(kind=kind, reset_hint_s=hint, matched=matched)


def _parse_claude_rate_limit_sleep(stderr_text: str) -> Optional[int]:
    """The reset hint (seconds) a rate-limit message carries ("try again in 3
    hours and 30 minutes", "resets in 1h 45m"), clamped to a sane range; None
    when unparseable, so the caller falls back to its park slice."""
    text_lc = (stderr_text or "").lower()
    if not any(p in text_lc for p in _RATE_LIMIT_PATTERNS):
        return None
    # Alternations longest-first, or "hours" would parse as a bare "h".
    m = re.search(r"(\d+)\s*(?:hours?|hr|h)\s*(?:and|,)?\s*(\d+)?\s*(?:minutes?|min|m)?",
                  text_lc)
    if m:
        guess = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60
        if guess > 0:
            return min(max(60, guess), _RL_HINT_CAP_S)
    m = re.search(r"(\d+)\s*(?:m|min|minutes?)", text_lc)
    if m:
        return min(max(60, int(m.group(1)) * 60), _RL_HINT_CAP_S)
    return None


def _audit_rate_limit_pause(source: str, attempt: int, sleep_s: int,
                            stderr_excerpt: str) -> None:
    """Append one JSON line to the state directory's rate-limit pause log.
    Audit-only; never raises."""
    try:
        path = config.STATE_DIR / "rate_limit_pauses.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
                 "attempt": attempt, "sleep_s": sleep_s,
                 "stderr_excerpt": (stderr_excerpt or "")[:400]}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# The private edition also persists the observed rate-limit window to the
# state directory, so a relaunched process skips the call that would
# rediscover it; omitted here.


def _rl_jitter(sleep_s: float) -> int:
    """Small multiplicative jitter so parallel workers don't retry in lockstep."""
    import random as _random
    return int(sleep_s * _random.uniform(0.90, 1.15))


def _park_with_countdown(name: str, until_epoch: float) -> int:
    """Sleep until ``until_epoch`` in heartbeat slices, one loud line per
    slice, so a long park is visible in the log tail. Returns seconds slept.
    The park LENGTH is the caller's (SUBAGENT_RL_SLEEP_S, withheld)."""
    total = 0.0
    while True:
        left = until_epoch - time.time()
        if left <= 0:
            return int(total)
        resume = time.strftime("%H:%M:%S", time.localtime(until_epoch))
        log.warning("[rate-limit] %s: parked, resume ~%s (%ds left)", name, resume, int(left))
        chunk = min(_PARK_HEARTBEAT_S, left)
        time.sleep(chunk)
        total += chunk


# --- Isolation: CLI subprocesses run OUTSIDE the repo ---
# The Claude Code CLI loads instruction files from the cwd and every ancestor
# (plus a per-directory session memory); the Codex CLI loads the cwd's agent
# instructions the same way. Run from the repo root, every agent would read
# the operator's working notes beside its system prompt, so every spawn runs
# from `claude_cli_cwd()`. `--agent <name>` still resolves by name: the
# definition is SYNCED byte for byte into the isolated directory before each
# spawn. These constants are what the run metadata stamps.
# Limits: the user-scope instruction file still loads from ANY cwd, and the
# `--agent` path passes no strict-MCP flag, so account connectors attach there.
CLAUDE_CLI_CONTEXT = {"cwd": "isolated", "agents": "synced"}
CODEX_CLI_CONTEXT = {"cwd": "isolated"}
# The CLI's project-instruction file names, checked on the cwd and ancestors.
_CLAUDE_INSTRUCTION_FILES = ("CLAUDE.md", "CLAUDE.local.md")


def claude_cli_cwd() -> Path:
    """The directory every Claude and Codex CLI subprocess runs in: a stable
    `Kestrel/claude_cwd` folder under the local application-data directory
    (the temp dir when unset), created on demand. REFUSES a path inside the
    repo, or one where it or any ancestor carries a project instruction file.
    It holds nothing but the synced agent definitions."""
    import tempfile
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    p = (Path(base) / "Kestrel" / "claude_cwd").resolve()
    root = Path(config.PROJECT_ROOT).resolve()
    if p == root or root in p.parents:
        raise RuntimeError(f"claude CLI cwd {p} is inside the repo ({root}); the "
                           f"repo's instruction file would load into the model")
    for anc in (p, *p.parents):
        for f in _CLAUDE_INSTRUCTION_FILES:
            if (anc / f).exists():
                raise RuntimeError(f"claude CLI cwd {p}: {anc / f} would load into "
                                   f"the model; pick a directory with no such ancestor")
    p.mkdir(parents=True, exist_ok=True)
    return p


def sync_agent_md(name: str, md_dir=None) -> Path:
    """Copy the agent definition `<name>.md` into the isolated directory's
    `.claude/agents/`, byte for byte, when absent or different, and return the
    copy's path. The CLI then finds `--agent <name>` as a PROJECT agent there
    and applies its own frontmatter rules to the same file. A missing source
    is refused: never an agent run with no prompt."""
    if md_dir is None:
        from tools.sync_agent_prompts import MD_DIR as md_dir  # not published in the public edition
    src = Path(md_dir) / f"{name}.md"
    if not src.exists():
        raise RuntimeError(f"subagent {name!r}: agent definition not found at {src}")
    dest = claude_cli_cwd() / ".claude" / "agents" / f"{name}.md"
    data = src.read_bytes()
    if not dest.exists() or dest.read_bytes() != data:
        # Atomic temp + os.replace: another worker's CLI may be reading it.
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    return dest


# The Claude agents Phase B spawns: the synthesizer (R1, the coverage retries,
# R3) and the council's arbiter. The agent path passes no --model, so each
# definition's frontmatter `model:` line IS the pin; the record stamps it and
# it enters the Stage-5 cache key.
PHASE_B_CLAUDE_AGENTS = ("cortex-portfolio-synthesizer", "cortex-council-arbiter")


def claude_agent_model(name: str, md_dir=None) -> Optional[str]:
    """The `model:` value in the agent definition's YAML frontmatter, or None
    when the file, the frontmatter or the line is missing. Never raises."""
    if md_dir is None:
        from tools.sync_agent_prompts import MD_DIR as md_dir  # not published in the public edition
    try:
        lines = (Path(md_dir) / f"{name}.md").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        key, sep, value = line.partition(":")
        if sep and key.strip() == "model":
            return value.strip().strip("\"'") or None
    return None


# --- Claude Code CLI (generic adapter; bills against the plan, not the API) ---

class ClaudeCodeCLI:
    """Runs the Claude Code CLI in print mode as a subprocess, from the
    isolated `claude_cli_cwd()`, with no session persistence and no tools.
    `max_tokens`, `temperature` and `cached_system` are ignored (no such CLI
    knobs). On a session limit it sleeps for the parsed reset hint (or the
    park slice) and retries up to MAX_RATE_LIMIT_ATTEMPTS times; a weekly cap
    fails at once. Each pause goes to the audit log.
    """

    DEFAULT_TIMEOUT_S = 300
    MAX_RATE_LIMIT_ATTEMPTS = 3
    MAX_TOTAL_WAIT_S = 6 * 3600  # safety ceiling across all attempts

    def __init__(self, model: str, claude_path: Optional[str] = None):
        self.model = model
        resolved = claude_path or shutil.which("claude")
        if not resolved:
            raise RuntimeError("`claude` CLI not found on PATH. Install Claude Code "
                               "or set claude_path explicitly.")
        self.claude_path = resolved

    def _build_cmd(self, system: Optional[str]) -> list[str]:
        cmd = [self.claude_path, "-p", "--model", self.model,
               "--no-session-persistence", "--tools", ""]
        if system:
            cmd.extend(["--system-prompt", system])
        return cmd

    def _run_with_rate_limit_retry(self, cmd: list[str], prompt: str):
        """Run the CLI; on a session limit, sleep + retry within
        MAX_RATE_LIMIT_ATTEMPTS / MAX_TOTAL_WAIT_S. Raises RuntimeError on any
        other failure, a timeout, a weekly cap, or exhausted retries."""
        total_waited = 0
        last_stderr = ""
        for attempt in range(self.MAX_RATE_LIMIT_ATTEMPTS):
            try:
                result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                        encoding="utf-8", timeout=self.DEFAULT_TIMEOUT_S,
                                        cwd=str(claude_cli_cwd()))   # never the repo root
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"claude CLI timed out after {self.DEFAULT_TIMEOUT_S}s") from e
            if result.returncode == 0:
                return result

            last_stderr = result.stderr or result.stdout or ""
            signal = classify_rate_limit(result.stdout, result.stderr,
                                         returncode=result.returncode)
            if signal is None:
                raise RuntimeError(f"claude CLI failed (exit {result.returncode}): "
                                   f"{_err_excerpt(last_stderr)}")
            if signal.kind == "weekly":
                raise RuntimeError(f"claude CLI hit the WEEKLY cap; not retrying "
                                   f"(matched {signal.matched!r}): {last_stderr[:300]}")
            base_s = signal.reset_hint_s or SUBAGENT_RL_SLEEP_S
            if base_s is None:
                raise RuntimeError("claude CLI rate-limited with no reset hint; the park "
                                   f"slice is withheld from the public edition: {last_stderr[:300]}")
            sleep_s = max(60, _rl_jitter(base_s))
            if (attempt == self.MAX_RATE_LIMIT_ATTEMPTS - 1
                    or total_waited + sleep_s > self.MAX_TOTAL_WAIT_S):
                raise RuntimeError(f"claude CLI rate-limited; retries exhausted "
                                   f"({attempt + 1}/{self.MAX_RATE_LIMIT_ATTEMPTS}, "
                                   f"total_waited={total_waited}s): {last_stderr[:300]}")
            log.warning("claude CLI rate-limited; sleeping %ds before retry %d/%d",
                        sleep_s, attempt + 2, self.MAX_RATE_LIMIT_ATTEMPTS)
            _audit_rate_limit_pause("ClaudeCodeCLI", attempt + 1, sleep_s, last_stderr)
            time.sleep(sleep_s)
            total_waited += sleep_s
        raise RuntimeError(f"claude CLI retry loop exited unexpectedly: {last_stderr[:300]}")

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024,
                 temperature: float = 0.0, cached_system: bool = False) -> str:
        return self._run_with_rate_limit_retry(self._build_cmd(system), prompt).stdout.strip()

    def generate_json(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048,
                      temperature: float = 0.0, cached_system: bool = False) -> dict:
        # The private edition also accepts a JSON schema here and has the CLI
        # enforce it (`--json-schema`); omitted here.
        return _json_retry(self.generate, prompt, system, "claude CLI")


# --- OpenAI Codex CLI (council seat and fallback twin, via ChatGPT OAuth) ---

# The Codex seat's persisted voice label, derived from the model so a model
# swap can never leave records naming a model that did not run.
CODEX_VOICE_LABEL = ("codex_" + re.sub(r"[^a-z0-9]+", "_", CODEX_MODEL.lower()).strip("_")
                     if CODEX_MODEL else "codex")


class CodexCLI:
    """Runs the Codex CLI in exec mode, billed to the ChatGPT plan via OAuth
    (the CLI on PATH plus a signed-in session). `max_tokens`, `temperature`
    and `cached_system` are ignored; reasoning effort defaults to
    CODEX_REASONING_EFFORT (omitted when None). The system prompt is embedded
    in the user prompt (no system-prompt flag). The sandbox is read-only and
    user config is ignored, so a call cannot touch the filesystem or inherit
    local settings.
    """

    DEFAULT_TIMEOUT_S = 300

    def __init__(self, model: Optional[str] = CODEX_MODEL, codex_path: Optional[str] = None,
                 reasoning_effort: Optional[str] = CODEX_REASONING_EFFORT):
        self.model = model
        self.reasoning_effort = reasoning_effort
        # which() finds the Windows launcher shims via PATHEXT; an explicit
        # codex_path overrides; then the npm global launcher under the roaming
        # application-data folder (which() can miss it in some shells).
        resolved = codex_path or shutil.which("codex")
        if not resolved:
            npm_prefix = os.environ.get("APPDATA")
            if npm_prefix:
                candidate = os.path.join(npm_prefix, "npm", "codex.cmd")
                if os.path.exists(candidate):
                    resolved = candidate
        if not resolved:
            raise RuntimeError("`codex` CLI not found on PATH. Install the Codex CLI "
                               "and sign in with a ChatGPT account.")
        self.codex_path = resolved

    def _build_cmd(self, output_file: str) -> list[str]:
        cmd = [self.codex_path, "exec"]
        if self.model:
            cmd += ["--model", self.model]
        if self.reasoning_effort:
            # TOML-typed config value; the quotes make it an explicit string.
            cmd += ["--config", f'model_reasoning_effort="{self.reasoning_effort}"']
        cmd += ["--skip-git-repo-check", "--sandbox", "read-only", "--ignore-user-config",
                "--ephemeral", "--output-last-message", output_file]
        return cmd

    def _embed_system(self, prompt: str, system: Optional[str]) -> str:
        """Embed the system prompt as a marked block in the user prompt."""
        if not system:
            return prompt
        return f"## SYSTEM INSTRUCTIONS\n{system}\n\n## USER MESSAGE\n{prompt}"

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024,
                 temperature: float = 0.0, cached_system: bool = False) -> str:
        import tempfile
        # The final message goes to a temp file (cleaner than parsing stdout).
        fd, out_path = tempfile.mkstemp(suffix=".txt", prefix="codex-out-")
        os.close(fd)
        try:
            # The prompt goes through stdin (the Windows command line is far
            # shorter than a critique prompt). Never the repo root as cwd.
            try:
                result = subprocess.run(self._build_cmd(out_path),
                                        input=self._embed_system(prompt, system),
                                        capture_output=True, text=True, encoding="utf-8",
                                        timeout=self.DEFAULT_TIMEOUT_S, cwd=str(claude_cli_cwd()))
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"codex CLI timed out after {self.DEFAULT_TIMEOUT_S}s") from e
            if result.returncode != 0:
                raise RuntimeError(f"codex CLI failed (exit {result.returncode}): "
                                   f"{_err_excerpt(result.stderr or result.stdout)}")
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except OSError as e:
                raise RuntimeError(f"codex output file unreadable: {e}") from e
            if not text:
                tail = result.stdout[-300:] if result.stdout else "empty"
                raise RuntimeError(f"codex returned empty final message (stdout tail: {tail})")
            return text
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    def generate_json(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048,
                      temperature: float = 0.0, cached_system: bool = False) -> dict:
        return _json_retry(self.generate, prompt, system, "codex")


# --- Google Gemini (REST, one key) ---

class GeminiRateLimitError(RuntimeError):
    """Gemini returned HTTP 429. The bulk ladder moves to the next tier."""


class GeminiLLM:
    """Google Gemini over the REST `generateContent` endpoint. One key, read
    from `GOOGLE_API_KEY` (core.config) unless passed in, sent in the
    `x-goog-api-key` header and never in the URL, so it cannot leak into a log
    line or a proxy's request record. `cached_system` is ignored.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    DEFAULT_TIMEOUT_S = 120

    def __init__(self, model: str, api_key: Optional[str] = None):
        key = (api_key or config.GOOGLE_API_KEY or "").strip()
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set; cannot use Gemini")
        self.model = model
        self._api_key = key
        self.last_usage: Optional[dict] = None   # usageMetadata of the last call

    def _invoke(self, prompt: str, system: Optional[str], max_tokens: int,
                temperature: float, json_mode: bool) -> str:
        import urllib.error as _urlerr
        import urllib.request as _urlreq

        gen_cfg: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": gen_cfg}
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        req = _urlreq.Request(self.API_URL.format(model=self.model),
                              data=json.dumps(body).encode("utf-8"),
                              headers={"Content-Type": "application/json",
                                       "x-goog-api-key": self._api_key})
        try:
            with _urlreq.urlopen(req, timeout=self.DEFAULT_TIMEOUT_S) as resp:
                data = json.loads(resp.read())
        except _urlerr.HTTPError as e:
            if e.code == 429:
                raise GeminiRateLimitError(f"Gemini 429 on model {self.model}") from e
            msg = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {e.code}: {_err_excerpt(msg, 200, 200)}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {type(e).__name__}: {e}") from e

        usage = data.get("usageMetadata") if isinstance(data, dict) else None
        self.last_usage = usage if isinstance(usage, dict) else None
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Gemini response shape unexpected: {str(data)[:300]}") from e

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048,
                 temperature: float = 0.0, cached_system: bool = False) -> str:
        return self._invoke(prompt, system, max_tokens, temperature, json_mode=False)

    def generate_json(self, prompt: str, system: Optional[str] = None, max_tokens: int = 4096,
                      temperature: float = 0.0, cached_system: bool = False) -> dict:
        raw = ""
        for attempt in range(2):
            raw = self._invoke(prompt, system, max_tokens, temperature, json_mode=True)
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return parsed
            log.warning("gemini returned unparseable JSON (attempt %d): %r", attempt + 1, raw[:200])
        raise ValueError(f"Gemini failed to produce valid JSON after 2 attempts: {raw[:300]}")


# --- Ollama (offline fallback) ---

class OllamaLLM:
    def __init__(self, model: str, host: Optional[str] = None):
        import ollama
        self._ollama = ollama
        self.client = ollama.Client(host=host) if host else ollama
        self.model = model

    def _call(self, prompt: str, system: Optional[str], max_tokens: int,
              temperature: float, json_mode: bool) -> str:
        kwargs = {"model": self.model, "prompt": prompt,
                  "options": {"temperature": temperature, "num_predict": max_tokens}}
        if json_mode:
            kwargs["format"] = "json"
        if system:
            kwargs["system"] = system
        return self.client.generate(**kwargs)["response"]

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024,
                 temperature: float = 0.0, cached_system: bool = False) -> str:
        return self._call(prompt, system, max_tokens, temperature, json_mode=False)

    def generate_json(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048,
                      temperature: float = 0.0, cached_system: bool = False) -> dict:
        raw = self._call(prompt, system, max_tokens, temperature, json_mode=True)
        parsed = _try_parse_json(raw)
        if parsed is None:
            raise ValueError(f"Ollama returned unparseable JSON: {raw[:300]}")
        return parsed


# --- Factory ---

def get_llm(model: Optional[str] = None) -> "AnthropicLLM | ClaudeCodeCLI | OllamaLLM":
    """Return an LLM instance for the configured provider.
    Override `model` to use a non-default (e.g. a larger model for escalated cases)."""
    provider = config.LLM_PROVIDER
    if provider == "claude_code_cli":
        return ClaudeCodeCLI(model or config.LLM_MODEL_PRIMARY)
    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set; cannot use Anthropic provider")
        return AnthropicLLM(model or config.LLM_MODEL_PRIMARY, config.ANTHROPIC_API_KEY)
    if provider == "ollama":
        return OllamaLLM(model or config.OLLAMA_MODEL, host=config.OLLAMA_HOST)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} "
                     "(expected 'claude_code_cli', 'anthropic', or 'ollama')")


# --- Subagent invocation: errors, routing sets, telemetry ---

class RateLimitError(RuntimeError):
    """A Claude CLI rate limit that will not clear by parking. invoke_subagent
    routes it to the Codex twin."""


class WeeklyCapError(RateLimitError):
    """Weekly plan cap: resets days away, so parking is pointless. Bulk agents
    fall through to Codex; the protected synthesizer surfaces to the operator
    instead (see SYNTH_ALLOW_CODEX)."""


class SubagentTimeoutError(RuntimeError):
    """The subagent subprocess timed out twice in a row. Fallback-eligible,
    like a rate limit."""


# Agents whose output quality is load-bearing enough that a silent model swap
# is unacceptable: on a WEEKLY cap these raise to the operator instead of
# falling back to Codex. Session limits park-and-retry regardless.
_PROTECTED_AGENTS: frozenset[str] = frozenset({"cortex-portfolio-synthesizer"})


def _synth_codex_allowed() -> bool:
    """Whether the protected synthesizer may fall to Codex on a weekly cap.
    The switch is withheld; None refuses the swap."""
    return bool(SYNTH_ALLOW_CODEX) if SYNTH_ALLOW_CODEX is not None else False


# Telemetry: callers read `subagent_telemetry` after a run for the fallback
# rate. (The private edition also renders a one-line run-tail summary.)
subagent_telemetry: dict[str, int] = {
    "claude_ok": 0, "claude_rate_limited": 0, "claude_other_error": 0,
    "claude_timeout": 0, "claude_malformed": 0,
    "gemini_ok": 0, "gemini_rate_limited": 0, "gemini_error": 0,
    "openrouter_ok": 0, "openrouter_retry": 0, "openrouter_timeout": 0,
    "openrouter_truncated": 0, "openrouter_call_lost": 0,
    "openrouter_host_latched": 0, "openrouter_host_skipped": 0,
    "openrouter_host_benched": 0, "openrouter_host_benched_skip": 0,
    "codex_ok": 0, "codex_error": 0, "no_fallback_available": 0,
}

# The BULK agents walk the ladder before Claude; every other agent tries
# Claude first. (Named for the provider that first served them; it is the
# routed set, whichever provider serves it.)
_GEMINI_AGENTS: frozenset[str] = frozenset({
    "nucleus-peer-context",
    "pathology-news-diagnosis",
    "pathology-news-diagnosis-v3",
    "fulcrum-falling-knife",
    "pulse-sector-outlook",
})


# --- Provider provenance ---
# A caller that must know WHICH provider served its call (graders read
# confidence on different scales, so a silent swap mixes calibrations) passes
# meta={} to invoke_subagent and reads it back after the call.

_served_tls = threading.local()


def _fill_meta(meta: Optional[dict], *, provider: str, model: Optional[str],
               fallback: bool, **extra) -> None:
    """Fill the caller's provenance out-param in place (no-op when None).
    fallback=True means the serving MODEL is not the agent's declared first
    choice (the primary line for a bulk agent, Claude otherwise). ``extra``
    rides only when the branch has more to add (OpenRouter: `upstream`,
    `reasoning_effort`, `profile`). The per-thread record is written on EVERY
    serve, so a caller with no out-param can still stamp its row."""
    _served_tls.record = {"provider": provider, "model": model,
                          "fallback": fallback, **extra}
    if meta is not None:
        meta.update(provider=provider, model=model, fallback=fallback, **extra)


def reset_last_served() -> None:
    """Clear this thread's record BEFORE a call whose server will be read, so a
    faked or failed call reads back as None, never as the previous server."""
    _served_tls.record = None


def last_served() -> Optional[dict]:
    """{"provider", "model", "fallback", ...} for this thread's most recent
    invoke_subagent / invoke_subagent_pinned serve, else None."""
    rec = getattr(_served_tls, "record", None)
    return dict(rec) if rec else None


def served_provenance() -> Optional[dict]:
    """The provenance stamp of this thread's last serve, or None. A
    fallback-served row is routed to manual review downstream."""
    rec = last_served()
    if not rec:
        return None
    out = {"provider": rec.get("provider"), "model": rec.get("model"),
           "provider_fallback": bool(rec.get("fallback"))}
    out.update({k: rec[k] for k in ("upstream", "reasoning_effort", "profile") if k in rec})
    return out


# --- The Gemini tier ---

def _invoke_gemini_subagent(name: str, prompt: str, *, timeout_s: int, model: str) -> dict:
    """One bulk call to Gemini on `model`, with the system prompt from the
    agent's Codex twin (same JSON contract). Raises RuntimeError,
    GeminiRateLimitError (429) or ValueError (answer not valid JSON)."""
    system = _load_codex_system_prompt(name)
    if system is None:
        raise RuntimeError(f"subagent {name!r}: no Codex twin definition, needed "
                           f"for the Gemini system prompt")
    llm = GeminiLLM(model)
    llm.DEFAULT_TIMEOUT_S = timeout_s
    raw = llm._invoke(prompt, system, max_tokens=8192, temperature=0.0, json_mode=True)
    parsed = _try_parse_json(raw)
    if parsed is None:
        raise ValueError(f"subagent {name!r} (Gemini) output not valid JSON: {raw[:300]}")
    return parsed


# --- The OpenAI-compatible bulk tier (OpenRouter) ---
# The client is `data.openai_compat` (not published). Here: the per-run HOST
# latch, a timed breaker, the telemetry and the call record.

class _BulkHostUnavailable(RuntimeError):
    """This host cannot serve this call (latched, benched, or lost after its
    retries). The ladder moves to the next tier."""


# The latch: a HOST-level hard refusal (no key, auth, credits exhausted, a
# request larger than the budget) latches the host for the process, so later
# calls skip it without a request. Keyed (host, model-or-None): None binds
# every model; a model key binds ONE slug, so a dead slug or a routing
# mismatch cannot kill a healthy primary on the same account.
_bulk_host_latch: dict[tuple, dict] = {}
_bulk_host_latch_lock = threading.Lock()
_MODEL_SCOPED_LATCH_KINDS = frozenset({"not_found", "no_route"})


def bulk_host_latched(host_name: str, model: Optional[str] = None) -> Optional[dict]:
    """The latch record that stops a call to `host_name`, else None. Host scope
    answers first; `model` then asks about that slug; no `model` asks "is
    anything latched on this host"."""
    with _bulk_host_latch_lock:
        rec = _bulk_host_latch.get((host_name, None))
        if rec is None and model is not None:
            rec = _bulk_host_latch.get((host_name, model))
        if rec is None and model is None:
            rec = next((r for (h, m), r in _bulk_host_latch.items()
                        if h == host_name and m is not None), None)
        return dict(rec) if rec else None


def reset_bulk_host_latch() -> None:
    """Clear the process latch AND the timed breaker (tests; a manual retry)."""
    with _bulk_host_latch_lock:
        _bulk_host_latch.clear()
    with _bulk_breaker_lock:
        _bulk_breaker.clear()


def _latch_bulk_host(host_name: str, kind: str, detail: str, *, model: str) -> None:
    """Latch `host_name` (or just `model` on it) off for the run, ONE error
    line. (The private edition also writes a marker file per event to the
    state directory for the post-sweep checks.)"""
    scoped = kind in _MODEL_SCOPED_LATCH_KINDS and bool(model)
    key = (host_name, model if scoped else None)
    with _bulk_host_latch_lock:
        if key in _bulk_host_latch:
            return
        rec = {"host": host_name, "kind": kind, "detail": (detail or "")[:300],
               "model": model, "scope": "model" if scoped else "host",
               "at": datetime.now().isoformat(timespec="seconds"), "pid": os.getpid()}
        _bulk_host_latch[key] = rec
    subagent_telemetry["openrouter_host_latched"] += 1
    log.error("bulk host %s LATCHED OFF for this run (%s: %s); every call now goes "
              "to the next tier, stamped fallback",
              f"{host_name} model {model}" if scoped else host_name, kind, rec["detail"])


# The timed breaker, for a host that is slow or erroring rather than refusing:
# after BULK_BENCH_AFTER_LOST lost calls the host is benched for the next step
# of BULK_BENCH_STEPS_S (doubling, capped); when a bench expires ONE caller
# takes the probe (half-open) while the rest keep skipping. Only a verdict
# saying the HOST is unwell counts as lost; a bad answer never benches. (The
# private edition also writes a marker file per bench; omitted here.)
_BULK_HEALTH_LOSS_KINDS = frozenset({"timeout", "transport", "upstream_error",
                                     "rate_limited", "billing_backoff"})
_bulk_breaker: dict[str, dict] = {}   # host -> {"lost", "level", "until", "probing"}
_bulk_breaker_lock = threading.Lock()
_bulk_now = time.monotonic            # injectable; monotonic, so a clock step is harmless


def _breaker_state(host_name: str) -> dict:
    return _bulk_breaker.setdefault(
        host_name, {"lost": 0, "level": 0, "until": None, "probing": False})


def _bulk_breaker_admit(host_name: str) -> str:
    """"closed" (call normally), "open" (benched or probing: skip) or "probe"
    (the bench expired and THIS caller takes the one probe)."""
    with _bulk_breaker_lock:
        st = _breaker_state(host_name)
        if st["until"] is None:
            return "closed"
        if st["probing"] or _bulk_now() < st["until"]:
            return "open"
        st["probing"] = True
        return "probe"


def _bulk_breaker_record(host_name: str, lost: Optional[bool], *,
                         probing: bool) -> Optional[int]:
    """Feed ONE finished call to the breaker (lost: False served, True lost,
    None neutral) and hand back any probe. Returns the bench seconds when this
    call benched the host. With the bench counts withheld it never benches."""
    with _bulk_breaker_lock:
        st = _breaker_state(host_name)
        if probing:
            st["probing"] = False
        if BULK_BENCH_AFTER_LOST is None or not BULK_BENCH_STEPS_S or lost is None:
            return None
        if lost is False:
            st.update(lost=0, level=0, until=None)
            return None
        if probing:   # the probe was lost: the next step up, capped
            level = min(st["level"] + 1, len(BULK_BENCH_STEPS_S) - 1)
        else:
            st["lost"] += 1
            if st["until"] is not None or st["lost"] < BULK_BENCH_AFTER_LOST:
                return None
            level = 0
        st.update(lost=0, level=level, until=_bulk_now() + BULK_BENCH_STEPS_S[level])
        return BULK_BENCH_STEPS_S[level]


_bulk_call_tls = threading.local()
_BULK_CALL_KEYS = ("upstream", "model_served", "finish_reason", "usage", "attempts",
                   "reasoning_effort", "profile")


def last_bulk_call() -> dict:
    """What served this thread's most recent OpenAI-compatible bulk call
    (every value None before any call)."""
    rec = getattr(_bulk_call_tls, "record", None) or {}
    return {k: rec.get(k) for k in _BULK_CALL_KEYS}


def _load_md_system_prompt(name: str) -> Optional[str]:
    """The agent definition's BODY. The prompt file follows the PROVIDER:
    OpenRouter reads the definition; Gemini and Codex read its twin."""
    from tools.sync_agent_prompts import MD_DIR, md_body  # not published in the public edition
    p = MD_DIR / f"{name}.md"
    return md_body(p) if p.exists() else None


def _count_openrouter_kind(kind: str) -> None:
    """Count ONE attempt's verdict by kind, retried attempts included."""
    from data import openai_compat as oc  # not published in the public edition
    if kind == oc.KIND_TIMEOUT:
        subagent_telemetry["openrouter_timeout"] += 1
    elif kind == oc.KIND_TRUNCATED:
        subagent_telemetry["openrouter_truncated"] += 1


def _invoke_openrouter_subagent(name: str, prompt: str, *, model: str,
                                symbol: Optional[str] = None) -> dict:
    """One bulk call through OpenRouter on `model`; returns parsed JSON. The
    timeout is the model PROFILE's, not the caller's (that one is sized for
    the Claude path). Raises _BulkHostUnavailable when the host is latched or
    benched, latches on this call, or loses it after bounded retries."""
    from data import openai_compat as oc  # not published in the public edition

    host = oc.OPENROUTER
    latched = bulk_host_latched(host.name, model)   # FIRST: never probe a latched host
    if latched:
        subagent_telemetry["openrouter_host_skipped"] += 1
        raise _BulkHostUnavailable(f"{host.name} latched off earlier this run ({latched['kind']})")
    admit = _bulk_breaker_admit(host.name)
    if admit == "open":
        subagent_telemetry["openrouter_host_benched_skip"] += 1
        raise _BulkHostUnavailable(f"{host.name} benched after repeated lost calls")
    probing = admit == "probe"
    try:
        return _openrouter_call(host, name, prompt, model=model, symbol=symbol, probing=probing)
    finally:
        if probing:   # a probe that RAISED must still hand the probe back
            with _bulk_breaker_lock:
                _breaker_state(host.name)["probing"] = False


def _openrouter_call(host, name: str, prompt: str, *, model: str,
                     symbol: Optional[str], probing: bool) -> dict:
    """The request itself, once latch and breaker let it through. A probe gets
    ONE attempt; its outcome closes the breaker or re-benches a step longer."""
    from data import openai_compat as oc  # not published in the public edition
    from data.bulk_provider import profile_for  # not published in the public edition

    key = host.key()
    if not key:
        _latch_bulk_host(host.name, "key_missing", f"{host.key_env} is not set or blank",
                         model=model)
        raise _BulkHostUnavailable(f"{host.key_env} is not set")
    system = _load_md_system_prompt(name)
    if system is None:
        raise _BulkHostUnavailable(f"subagent {name!r}: no agent definition for the system prompt")
    profile = profile_for(model)
    body = oc.build_body(host, model, system, prompt, profile)

    def _on_retry(r) -> None:
        subagent_telemetry["openrouter_retry"] += 1
        _count_openrouter_kind(r.kind)
        log.warning("openrouter %s on %r%s (HTTP %s), retrying: %s", r.kind, name,
                    f" [{symbol}]" if symbol else "", r.status, r.detail[:160])

    _bulk_call_tls.record = None
    result = oc.call_with_retries(host, key, body, profile.timeout_s, on_retry=_on_retry,
                                  max_attempts=1 if probing else None)
    _count_openrouter_kind(result.kind)

    if result.ok:
        lost: Optional[bool] = False
    elif result.action == oc.NEXT_TIER and result.kind in _BULK_HEALTH_LOSS_KINDS:
        lost = True
    else:
        lost = None   # a latch (terminal on its own), a bad answer, a rejected request
    bench_s = _bulk_breaker_record(host.name, lost, probing=probing)
    if bench_s:
        subagent_telemetry["openrouter_host_benched"] += 1
        log.error("bulk host %s BENCHED for %d min (%s: HTTP %s); bulk calls serve from "
                  "the next tier, stamped fallback, until one probe gets through",
                  host.name, int(round(bench_s / 60)), result.kind, result.status)

    if result.ok:
        subagent_telemetry["openrouter_ok"] += 1
        _bulk_call_tls.record = {"upstream": result.upstream,
                                 "model_served": result.model_served,
                                 "finish_reason": result.finish_reason, "usage": result.usage,
                                 "attempts": result.attempts, **profile.stamp()}
        return result.parsed
    if result.action == oc.LATCH:
        _latch_bulk_host(host.name, result.kind, f"HTTP {result.status}: {result.detail}",
                         model=model)
    else:
        subagent_telemetry["openrouter_call_lost"] += 1
    raise _BulkHostUnavailable(f"{host.name} {result.kind} on {name!r} after "
                               f"{result.attempts} attempt(s) (HTTP {result.status}): "
                               f"{result.detail[:160]}")


# --- The Claude tier: agent path and prompt mode ---

# Agents whose output is large enough to hit the json output format's result
# truncation; these capture stream-json and reassemble the full text.
_STREAM_CAPTURE_AGENTS = {"cortex-portfolio-synthesizer"}


def _dump_subagent_failure(name: str, reason: str, *, stdout: Optional[str],
                           wrapper: Optional[dict], raw_out: Optional[str],
                           returncode: int, stderr: Optional[str]) -> None:
    """Diagnostic only: when SUBAGENT_DEBUG_DIR is set, write the full failing
    payload (raw model output + CLI wrapper metadata) there so a parse failure
    can be root-caused. A no-op while withheld (None); never raises."""
    if not SUBAGENT_DEBUG_DIR:
        return
    try:
        d = Path(SUBAGENT_DEBUG_DIR)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}_{reason}_{os.getpid()}.json"
        payload = {
            "agent": name, "reason": reason, "returncode": returncode,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
            "stderr_head": (stderr or "")[:4000],
            # wrapper metadata minus the (huge) result string, which is raw_out.
            "wrapper_meta": ({k: v for k, v in wrapper.items() if k != "result"}
                             if isinstance(wrapper, dict) else None),
            "raw_output": raw_out,
            "stdout_raw": stdout if wrapper is None else None,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.warning("subagent %r %s: dumped failing payload to %s", name, reason, path)
    except Exception as e:
        log.warning("subagent %r failure-dump errored (%s): %s", name, type(e).__name__, e)


def _invoke_claude_subagent(name: str, prompt: str, *, timeout_s: int,
                            claude_path: Optional[str]) -> dict:
    """The project-agent path: `--agent <name>` from the isolated directory,
    with the definition synced in first. It passes NO strict-MCP flag, so
    account-level connectors attach, and the user-scope instruction file
    loads: isolation here keeps the REPO's instruction files and session
    memory out, nothing more. Limits, timeouts and malformed output are
    `_run_claude_cli`'s."""
    resolved = claude_path or shutil.which("claude")
    if not resolved:
        raise RuntimeError("`claude` CLI not found on PATH")
    stream = name in _STREAM_CAPTURE_AGENTS
    out_fmt = (["--output-format", "stream-json", "--verbose"] if stream
               else ["--output-format", "json"])
    sync_agent_md(name)
    cmd = [resolved, "-p", "--agent", name, "--no-session-persistence", "--tools", ""] + out_fmt
    payload, _run = _run_claude_cli(name, cmd, prompt, timeout_s=timeout_s,
                                    cwd=str(claude_cli_cwd()), stream=stream,
                                    malformed_reinvoke=not stream)
    return payload


def _run_claude_cli(name: str, cmd: list, prompt: str, *, timeout_s: int, cwd: str,
                    stream: bool, malformed_reinvoke: bool) -> tuple:
    """The one Claude CLI loop, shared by the agent path and prompt mode: run
    `cmd` with `prompt` on stdin from `cwd`, park on a session limit, raise on
    a weekly cap or a second timeout, parse the payload (stream-json
    reassembly when `stream`), and re-invoke once on a malformed answer only
    when `malformed_reinvoke`. Returns (payload, {"wrapper", "attempts",
    "waited_s"}).

    Public edition: with the park slice or cap withheld, a session limit
    raises RateLimitError at once. (The private edition also honours a window
    persisted by an earlier process before the first call.)
    """
    total_waited = attempt = timeouts = malformed_attempts = 0
    stream_full_text = stream_result_event = None
    parking_enabled = SUBAGENT_RL_SLEEP_S is not None and SUBAGENT_RL_MAX_TOTAL_S is not None

    while True:
        attempt += 1
        try:
            result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                    encoding="utf-8", timeout=timeout_s, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            timeouts += 1   # one retry, then a fallback-eligible error
            if timeouts >= 2:
                raise SubagentTimeoutError(
                    f"subagent {name!r} timed out twice ({timeout_s}s each)") from e
            log.warning("subagent %r timed out after %ds; one retry", name, timeout_s)
            continue

        # A limit arrives as exit 0 with an error wrapper or as exit 1: check both.
        rl_signal = classify_rate_limit(result.stdout, result.stderr,
                                        returncode=result.returncode)
        rate_limited = rl_signal is not None
        if stream:
            stream_full_text, stream_result_event, saw_rl = _reassemble_stream_json(result.stdout)
            rate_limited = rate_limited or saw_rl
            # A cleanly completed result trumps ANY limit signal: it is paid for.
            if (stream_result_event is not None and not stream_result_event.get("is_error")
                    and stream_full_text):
                rate_limited = False
        if rate_limited:
            excerpt = result.stderr or result.stdout or ""
            if rl_signal is not None and rl_signal.kind == "weekly":
                # Resets days away: raise at once; invoke_subagent decides
                # Codex vs operator.
                _audit_rate_limit_pause(f"subagent[{name}]", attempt, 0, excerpt)
                raise WeeklyCapError(f"subagent {name!r} hit the WEEKLY cap "
                                     f"(matched {rl_signal.matched!r}): {excerpt[:200]}")
            if not parking_enabled:
                raise RateLimitError(f"subagent {name!r} rate-limited; parking is withheld "
                                     f"from the public edition: {excerpt[:200]}")
            hint = rl_signal.reset_hint_s if rl_signal else None
            # The SHORTER of hint and park slice, above a floor, jittered.
            sleep_s = max(300, _rl_jitter(min(SUBAGENT_RL_SLEEP_S, hint or SUBAGENT_RL_SLEEP_S)))
            if total_waited + sleep_s > SUBAGENT_RL_MAX_TOTAL_S:
                raise RateLimitError(f"subagent {name!r} rate-limited; ceiling reached "
                                     f"(attempt {attempt}, total_waited={total_waited}s): "
                                     f"{excerpt[:200]}")
            log.warning("subagent %r rate-limited; parking %ds before retry %d "
                        "(total_waited=%ds)", name, sleep_s, attempt + 1, total_waited)
            _audit_rate_limit_pause(f"subagent[{name}]", attempt, sleep_s, excerpt)
            _park_with_countdown(name, time.time() + sleep_s)
            total_waited += sleep_s
            continue

        if result.returncode != 0:
            raise RuntimeError(f"subagent {name!r} failed (exit {result.returncode}): "
                               f"{_err_excerpt(result.stderr or result.stdout)}")
        try:
            payload = _parse_subagent_payload(name, result, stream=stream,
                                              stream_full_text=stream_full_text,
                                              stream_result_event=stream_result_event)
        except ValueError:
            # One re-invoke on malformed output, for cheap non-stream agents only.
            if not malformed_reinvoke or malformed_attempts >= 1:
                raise
            malformed_attempts += 1
            subagent_telemetry["claude_malformed"] += 1
            log.warning("subagent %r returned malformed output; one re-invoke", name)
            continue
        if stream:
            wrapper = stream_result_event
        else:
            try:
                wrapper = json.loads(result.stdout)
            except (TypeError, ValueError):
                wrapper = None
        return payload, {"wrapper": wrapper if isinstance(wrapper, dict) else None,
                         "attempts": attempt, "waited_s": total_waited}


def claude_usage_from_result(wrapper) -> dict:
    """The CLI result object's token accounting as one flat dict: the
    per-model row first (served id under `canonicalModel`), top-level `usage`
    as fallback; ints, 0 when absent; `prompt_tokens` = input + cache read +
    cache creation."""
    w = wrapper if isinstance(wrapper, dict) else {}
    mu = w.get("modelUsage") if isinstance(w.get("modelUsage"), dict) else {}
    served = next(iter(mu), None)
    row = mu.get(served) if isinstance(mu.get(served), dict) else {}
    u = w.get("usage") if isinstance(w.get("usage"), dict) else {}

    def _i(*vals) -> int:
        for v in vals:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return 0

    inp = _i(row.get("inputTokens"), u.get("input_tokens"))
    cr = _i(row.get("cacheReadInputTokens"), u.get("cache_read_input_tokens"))
    cc = _i(row.get("cacheCreationInputTokens"), u.get("cache_creation_input_tokens"))
    return {"plan_served": True, "model_served": row.get("canonicalModel") or served,
            "prompt_tokens": inp + cr + cc, "cache_read_tokens": cr,
            "cache_creation_tokens": cc,
            "completion_tokens": _i(row.get("outputTokens"), u.get("output_tokens")),
            "duration_api_ms": w.get("duration_api_ms"), "num_turns": w.get("num_turns")}


def invoke_claude_prompt(system: str, user: str, *, model: str, timeout_s: int,
                         effort: Optional[str] = None, name: str = "claude-prompt",
                         claude_path: Optional[str] = None) -> tuple:
    """Prompt mode: a Claude CLI call with an EXPLICIT system prompt and model,
    for a caller that assembles its prompt per call (the council's Claude
    seat). Returns (payload, info).

    `--model <id>` overrides any definition's pin; `--system-prompt-file` is a
    private file in the isolated directory, removed after the call (the
    Windows command line is too short for the prompt); `--effort` when given;
    no session persistence; no tools; `--strict-mcp-config`, so account-level
    connectors do NOT attach on this path (the agent path does not pass it);
    stream-json capture through `_run_claude_cli`. Never a Codex fallback: a
    seat that fell to Codex would be a second Codex seat stamped as the first.

    `info`: {"usage": claude_usage_from_result(...) plus `effort`, "attempts",
    "waited_s"}. (The private edition also stamps the CLI version.)"""
    import tempfile
    resolved = claude_path or shutil.which("claude")
    if not resolved:
        raise RuntimeError("`claude` CLI not found on PATH")
    cwd = claude_cli_cwd()
    fd, path = tempfile.mkstemp(prefix=".system_", suffix=".txt", dir=str(cwd))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(system or "")
        cmd = [resolved, "-p", "--model", model, "--system-prompt-file", path]
        if effort:
            cmd += ["--effort", str(effort)]
        cmd += ["--no-session-persistence", "--tools", "", "--strict-mcp-config",
                "--output-format", "stream-json", "--verbose"]
        payload, run = _run_claude_cli(name, cmd, user, timeout_s=timeout_s, cwd=str(cwd),
                                       stream=True, malformed_reinvoke=False)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    usage = claude_usage_from_result(run.get("wrapper"))
    usage["effort"] = effort
    return payload, {"usage": usage, "attempts": run.get("attempts"),
                     "waited_s": run.get("waited_s")}


def _parse_subagent_payload(name: str, result, *, stream: bool,
                            stream_full_text: Optional[str],
                            stream_result_event: Optional[dict]) -> dict:
    """Parse a completed (exit 0, not limited) subagent run into its JSON
    payload. Raises ValueError on empty / malformed output, dumping the
    payload first when the debug directory is set."""
    def _fail(reason: str, wrapper, raw_out, msg: str):
        _dump_subagent_failure(name, reason, stdout=result.stdout, wrapper=wrapper,
                               raw_out=raw_out, returncode=result.returncode,
                               stderr=result.stderr)
        raise ValueError(msg)

    if stream:
        # The text reassembled from streamed blocks, not the truncation-prone
        # final `result` field.
        ev = stream_result_event
        if ev is not None and ev.get("is_error"):
            _fail("stream_error", ev, stream_full_text,
                  f"subagent {name!r} stream reported error (subtype={ev.get('subtype')})")
        if not stream_full_text or not stream_full_text.strip():
            _fail("empty_result", ev, stream_full_text,
                  f"subagent {name!r} returned empty streamed output")
        parsed = _try_parse_json(stream_full_text)
        if parsed is None:
            _fail("invalid_json", ev, stream_full_text,
                  f"subagent {name!r} output is not valid JSON: {stream_full_text[:300]}")
        return parsed

    try:
        wrapper = json.loads(result.stdout)
    except json.JSONDecodeError:
        _fail("wrapper_unparseable", None, None,
              f"subagent {name!r} wrapper JSON unparseable: {result.stdout[:300]}")
    raw = wrapper.get("result", "")
    if not isinstance(raw, str) or not raw.strip():
        _fail("empty_result", wrapper, raw if isinstance(raw, str) else None,
              f"subagent {name!r} returned empty result (is_error={wrapper.get('is_error')})")
    parsed = _try_parse_json(raw)
    if parsed is None:
        _fail("invalid_json", wrapper, raw,
              f"subagent {name!r} output is not valid JSON: {raw[:300]}")
    return parsed


# --- The Codex twin ---

def _load_codex_system_prompt(name: str) -> Optional[str]:
    """`developer_instructions` from the agent's Codex twin
    (`.codex/agents/<name>.toml` in the repo), or None."""
    import tomllib as _tomllib
    p = Path(__file__).resolve().parent.parent / ".codex" / "agents" / f"{name}.toml"
    if not p.exists():
        return None
    try:
        return _tomllib.loads(p.read_text(encoding="utf-8")).get("developer_instructions")
    except Exception:
        return None


def _codex_fallback(name: str, prompt: str, timeout_s: int) -> dict:
    """The Codex fallback for the rate-limit and timeout paths: the system
    prompt from the agent's Codex twin, the telemetry in one place."""
    try:
        system = _load_codex_system_prompt(name)
        if system is None:
            raise RuntimeError(f"subagent {name!r} has no Codex twin; cannot fall back.")
        codex = CodexCLI()
        codex.DEFAULT_TIMEOUT_S = max(timeout_s, codex.DEFAULT_TIMEOUT_S)
        raw = codex.generate(prompt, system=system)
        parsed = _try_parse_json(raw)
        if parsed is None:
            raise ValueError(f"subagent {name!r} (Codex fallback) output not valid JSON: {raw[:300]}")
        subagent_telemetry["codex_ok"] += 1
        return parsed
    except RuntimeError as e:
        key = "no_fallback_available" if "no Codex twin" in str(e) else "codex_error"
        subagent_telemetry[key] += 1
        raise


# --- The bulk ladder ---

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GEMINI = "gemini"
_announced_ladder_problems: set = set()   # each announced ONCE per process


def _provider_of(identifier: Optional[str]) -> Optional[str]:
    """The provider serving `identifier`, or None for a refused shape. Never
    guesses: "vendor/model" is OpenRouter, "gemini-..." is Gemini."""
    ident = (identifier or "").strip()
    if not ident or any(ch.isspace() for ch in ident):
        return None
    if "/" in ident:
        vendor, _, model = ident.partition("/")
        return PROVIDER_OPENROUTER if vendor and model else None
    if ident.startswith("gemini-") and len(ident) > len("gemini-"):
        return PROVIDER_GEMINI
    return None


def _ladder_tiers() -> tuple[list, list]:
    """([(slot, identifier, provider), ...], problems) for the configured
    lines in ladder order. Claude (tier 3) is fixed in code, not listed."""
    tiers: list = []
    problems: list = []
    if BULK_PRIMARY_MODEL is None and BULK_FALLBACK_MODEL is None:
        problems.append("bulk ladder model lines are withheld from the public "
                        "edition; bulk agents go straight to Claude")
        return tiers, problems
    for slot, ident in (("tier_1", BULK_PRIMARY_MODEL), ("tier_2", BULK_FALLBACK_MODEL)):
        if ident is None:
            continue
        provider = _provider_of(ident)
        if provider is None:
            problems.append(f"{slot} identifier {ident!r} names no known provider; skipped")
        else:
            tiers.append((slot, ident, provider))
    return tiers, problems


def _run_bulk_ladder(name: str, prompt: str, *, timeout_s: int,
                     symbol: Optional[str], meta: Optional[dict]) -> tuple:
    """Walk the CONFIGURED tiers for one call. Returns (True, parsed) from the
    first tier that serves, with ``meta`` filled, or (False, None) and the
    caller falls through to Claude. ``fallback`` is computed against the
    DECLARED primary, so any other tier's serve is flagged for manual review
    (the skip-trades direction). A host-level failure skips every tier on that
    host; a model-level one skips only the tier naming it."""
    tiers, problems = _ladder_tiers()
    for p in problems:
        if p not in _announced_ladder_problems:
            _announced_ladder_problems.add(p)
            log.error("bulk ladder: %s", p)
    for slot, ident, provider in tiers:
        is_fallback = ident != BULK_PRIMARY_MODEL
        if provider == PROVIDER_OPENROUTER:
            try:
                out = _invoke_openrouter_subagent(name, prompt, model=ident, symbol=symbol)
            except _BulkHostUnavailable as e:
                log.warning("bulk tier %s (%s) could not serve %r (%s); next tier",
                            slot, ident, name, str(e)[:160])
                continue
            rec = last_bulk_call()
            _fill_meta(meta, provider=PROVIDER_OPENROUTER, model=ident, fallback=is_fallback,
                       upstream=rec["upstream"], reasoning_effort=rec["reasoning_effort"],
                       profile=rec["profile"])
            return True, out
        if provider == PROVIDER_GEMINI:
            try:
                out = _invoke_gemini_subagent(name, prompt, timeout_s=timeout_s, model=ident)
            except (GeminiRateLimitError, RuntimeError, ValueError) as e:
                limited = isinstance(e, GeminiRateLimitError)
                subagent_telemetry["gemini_rate_limited" if limited else "gemini_error"] += 1
                log.warning("bulk tier %s (%s) %s on %r (%s); next tier", slot, ident,
                            "rate-limited" if limited else "failed", name, str(e)[:120])
                continue
            subagent_telemetry["gemini_ok"] += 1
            _fill_meta(meta, provider=PROVIDER_GEMINI, model=ident, fallback=is_fallback)
            return True, out
    return False, None


def invoke_subagent(name: str, prompt: str, *, timeout_s: int = 300,
                    claude_path: Optional[str] = None, allow_codex_fallback: bool = True,
                    symbol: Optional[str] = None, meta: Optional[dict] = None) -> dict:
    """Invoke a project subagent. A BULK agent walks the bulk ladder first and
    reaches Claude only as the third step; every other agent tries Claude
    first. On a Claude rate limit or a double timeout either kind falls back
    to its Codex twin. ``symbol`` tags the call for the logs; ``meta`` is a
    provenance out-param (see _fill_meta) and never changes routing.

    Raises RuntimeError (a non-limit Claude failure without fallback, or Codex
    failed), ValueError (unparseable output after fallback), RateLimitError
    (only with allow_codex_fallback=False), WeeklyCapError (weekly cap on the
    protected synthesizer without the escape hatch: no silent swap on the
    portfolio author), SubagentTimeoutError (two timeouts, no fallback).
    """
    if name in _GEMINI_AGENTS:
        served, out = _run_bulk_ladder(name, prompt, timeout_s=timeout_s,
                                       symbol=symbol, meta=meta)
        if served:
            return out

    try:
        out = _invoke_claude_subagent(name, prompt, timeout_s=timeout_s, claude_path=claude_path)
        subagent_telemetry["claude_ok"] += 1
        # Claude is the FALLBACK tier for bulk agents, the first choice otherwise.
        _fill_meta(meta, provider="claude", model="claude-cli", fallback=name in _GEMINI_AGENTS)
        return out
    except RateLimitError as e:
        subagent_telemetry["claude_rate_limited"] += 1
        if (isinstance(e, WeeklyCapError) and name in _PROTECTED_AGENTS
                and not _synth_codex_allowed()):
            log.error("WEEKLY cap on protected agent %r: refusing a silent Codex "
                      "fallback; the operator decides", name)
            raise
        if not allow_codex_fallback:
            raise
        log.warning("Claude rate-limited on %r; falling back to Codex", name)
    except SubagentTimeoutError:
        subagent_telemetry["claude_timeout"] += 1
        if not allow_codex_fallback:
            raise
        log.warning("Claude timed out twice on %r; falling back to Codex", name)
    except (RuntimeError, ValueError):
        subagent_telemetry["claude_other_error"] += 1
        raise
    out = _codex_fallback(name, prompt, timeout_s)
    _fill_meta(meta, provider="codex", model="codex-cli", fallback=True)
    return out


def invoke_subagent_pinned(name: str, prompt: str, *, provider: str,
                           model: Optional[str] = None, timeout_s: int = 300,
                           claude_path: Optional[str] = None, symbol: Optional[str] = None,
                           meta: Optional[dict] = None) -> dict:
    """No-fallback dispatch to exactly ONE provider ('openrouter', 'gemini' or
    'claude') and, for the first two, exactly ONE model. Raises instead of
    swapping: replay and self-consistency callers need a fixed calibration.
    Not for the sweep hot path. ``model`` pins the grader of the record being
    replayed; left None it resolves to the ladder's model for that provider,
    and a provider the ladder does not name raises (a guessed model is a
    calibration nobody chose)."""
    if provider in (PROVIDER_OPENROUTER, PROVIDER_GEMINI) and not model:
        on_ladder = [ident for _, ident, prov in _ladder_tiers()[0] if prov == provider]
        if not on_ladder:
            raise RuntimeError(f"invoke_subagent_pinned: no {provider} model is on the "
                               f"bulk ladder; pin one with model=")
        model = on_ladder[0]
    if provider == PROVIDER_OPENROUTER:
        out = _invoke_openrouter_subagent(name, prompt, model=model, symbol=symbol)
        rec = last_bulk_call()
        _fill_meta(meta, provider=PROVIDER_OPENROUTER, model=model, fallback=False,
                   upstream=rec["upstream"], reasoning_effort=rec["reasoning_effort"],
                   profile=rec["profile"])
        return out
    if provider == PROVIDER_GEMINI:
        out = _invoke_gemini_subagent(name, prompt, timeout_s=timeout_s, model=model)
        _fill_meta(meta, provider=PROVIDER_GEMINI, model=model, fallback=False)
        return out
    if provider == "claude":
        out = _invoke_claude_subagent(name, prompt, timeout_s=timeout_s, claude_path=claude_path)
        subagent_telemetry["claude_ok"] += 1
        _fill_meta(meta, provider="claude", model="claude-cli", fallback=False)
        return out
    raise ValueError(f"invoke_subagent_pinned: unknown provider {provider!r} "
                     f"(expected 'openrouter', 'gemini' or 'claude')")


# --- Helpers ---

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction (markdown fences, trailing prose). Returns
    a dict or None, never another JSON type: every caller indexes by key. A
    common model tic wraps the object in a one-element array, so a non-object
    parse falls through to the first-{...} extraction."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            recovered = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if isinstance(recovered, dict):
            return recovered
    return None


def _reassemble_stream_json(stdout: str):
    """Rebuild the full assistant text from the CLI's stream-json output by
    concatenating every `text` block of the `assistant` events (thinking
    ignored). Returns (full_text_or_None, final_result_event_or_None,
    saw_active_rate_limit). Why: the json output format TRUNCATES a very large
    `result`, while the streamed blocks carry all of it; partial messages are
    not requested, so plain concatenation is correct."""
    texts: list[str] = []
    result_event: Optional[dict] = None
    saw_rate_limit = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate any non-JSON noise line
        etype = evt.get("type")
        if etype == "assistant":
            for block in ((evt.get("message") or {}).get("content") or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text") or "")
        elif etype == "result":
            result_event = evt
        elif etype == "rate_limit_event":
            # Only a BLOCKING status counts: the CLI also emits warning
            # statuses on calls that then complete normally.
            status = ((evt.get("rate_limit_info") or {}).get("status") or "").lower()
            if status in ("rejected", "limit_exceeded", "blocked", "exceeded"):
                saw_rate_limit = True
    return ("".join(texts) if texts else None), result_event, saw_rate_limit
