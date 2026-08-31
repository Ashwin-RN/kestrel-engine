"""
LLM adapter — one interface, three providers.

`claude_code_cli` is the default for MVP — shells out to `claude -p` so calls
bill against the Claude Code Max plan rather than the per-token API.
`anthropic` (direct API) is kept for future headless/batch automation.
`ollama` (local) is the offline fallback. Switch via
`KESTREL_LLM_PROVIDER=claude_code_cli|anthropic|ollama` env var (see core.config).

The adapter is intentionally thin:
- `generate(prompt, system=...)` returns text
- `generate_json(prompt, system=...)` returns parsed JSON (retry once on parse fail)
- Prompt caching is supported for Anthropic via `cached_system=True` — use it for
  long, stable system prompts (sector taxonomies, scorecard criteria).

All providers are duck-typed to the same surface. No ABC — when a fourth
provider appears, promote the shape into a protocol then.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from typing import Optional

from core import config

log = logging.getLogger(__name__)


# ── Anthropic (primary) ──────────────────────────────────────────────────────

class AnthropicLLM:
    def __init__(self, model: str, api_key: str):
        # Lazy import so Ollama-only users don't need the anthropic package installed.
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        cached_system: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            if cached_system:
                kwargs["system"] = [{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)
        # Concatenate all text blocks (Claude may split responses)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cached_system: bool = False,
    ) -> dict:
        json_instruction = "\n\nRespond with a single JSON object. No prose, no markdown fences."
        full_prompt = prompt + json_instruction
        for attempt in range(2):
            raw = self.generate(
                full_prompt, system=system,
                max_tokens=max_tokens, temperature=temperature,
                cached_system=cached_system,
            )
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return parsed
            log.warning("anthropic returned unparseable JSON (attempt %d): %r", attempt + 1, raw[:200])
        raise ValueError(f"Anthropic failed to produce valid JSON after 2 attempts: {raw[:300]}")


# ── Claude Code CLI (primary for MVP — uses Max plan quota, not API) ─────────

class ClaudeCodeCLI:
    """Shells out to `claude -p` so calls bill against the Max plan.

    Constraints vs. AnthropicLLM:
    - `max_tokens`, `temperature`, and `cached_system` are silently ignored —
      the CLI doesn't expose these knobs.
    - Each call is a fresh subprocess (no session persistence). Cold-start
      latency is ~1-3s on top of inference time.
    - `claude` must be on PATH (or override via `claude_path`).

    For `generate_json`, prefers `--json-schema` enforcement when a schema is
    supplied; otherwise falls back to the prompt-engineered approach.
    """

    DEFAULT_TIMEOUT_S = 300

    def __init__(self, model: str, claude_path: Optional[str] = None):
        self.model = model
        resolved = claude_path or shutil.which("claude")
        if not resolved:
            raise RuntimeError(
                "`claude` CLI not found on PATH. Install Claude Code or set "
                "claude_path explicitly."
            )
        self.claude_path = resolved

    def _build_cmd(self, system: Optional[str], json_schema: Optional[str] = None) -> list[str]:
        cmd = [
            self.claude_path, "-p",
            "--model", self.model,
            "--no-session-persistence",
            "--tools", "",   # disable all tools — we want pure reasoning
        ]
        if system:
            cmd.extend(["--system-prompt", system])
        if json_schema:
            cmd.extend(["--json-schema", json_schema])
        return cmd

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,            # ignored
        temperature: float = 0.0,          # ignored
        cached_system: bool = False,       # ignored
    ) -> str:
        cmd = self._build_cmd(system)
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.DEFAULT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude -p timed out after {self.DEFAULT_TIMEOUT_S}s") from e
        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (exit {result.returncode}): "
                f"{(result.stderr or result.stdout)[:500]}"
            )
        return result.stdout.strip()

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,            # ignored
        temperature: float = 0.0,          # ignored
        cached_system: bool = False,       # ignored
        json_schema: Optional[dict] = None,
    ) -> dict:
        if json_schema is not None:
            # CLI-enforced schema. With --json-schema + --output-format json,
            # the schema-validated payload is in the wrapper's `structured_output` field.
            cmd = self._build_cmd(system, json_schema=json.dumps(json_schema))
            cmd.extend(["--output-format", "json"])
            try:
                result = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True,
                    encoding="utf-8", timeout=self.DEFAULT_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"claude -p timed out after {self.DEFAULT_TIMEOUT_S}s") from e
            if result.returncode != 0:
                raise RuntimeError(
                    f"claude -p failed (exit {result.returncode}): "
                    f"{(result.stderr or result.stdout)[:500]}"
                )
            try:
                wrapper = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"claude -p wrapper JSON unparseable: {result.stdout[:300]}"
                ) from e
            payload = wrapper.get("structured_output")
            if not isinstance(payload, dict):
                raise ValueError(
                    "claude -p response had no `structured_output` payload "
                    f"(is_error={wrapper.get('is_error')}, result={wrapper.get('result', '')[:200]})"
                )
            return payload

        # No schema — same retry pattern as Anthropic.
        json_instruction = "\n\nRespond with a single JSON object. No prose, no markdown fences."
        full_prompt = prompt + json_instruction
        raw = ""
        for attempt in range(2):
            raw = self.generate(full_prompt, system=system)
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return parsed
            log.warning("claude -p returned unparseable JSON (attempt %d): %r", attempt + 1, raw[:200])
        raise ValueError(f"claude -p failed to produce valid JSON after 2 attempts: {raw[:300]}")


# ── OpenAI Codex CLI (council third voice — gpt-5.5 via ChatGPT OAuth) ──────

class CodexCLI:
    """Shells out to `codex exec` so calls bill against the ChatGPT plan.

    Requires:
      - Node.js + `npm install -g @openai/codex` (resolves to codex.ps1 +
        codex.cmd on Windows; bash launcher at $NPM_PREFIX/codex on Unix).
      - `codex login` already run (ChatGPT OAuth); `codex login status`
        should report "Logged in using ChatGPT".

    GPT-5.5 is included free for ChatGPT Free/Go/Plus/Pro users (limited-
    time promo, 2026). No API key needed.

    Constraints vs. AnthropicLLM:
    - `max_tokens`, `temperature`, and `cached_system` are silently ignored —
      the CLI doesn't expose those knobs (model picks its own reasoning effort).
    - System prompt is embedded in the user prompt (Codex CLI has no
      --system-prompt flag; .codex/agents/<name>.toml exists but is heavier
      than we need for council critique calls).
    - Cold start latency: 30-60s (Codex agent initialization + reasoning).
      Default timeout 300s; bump for longer prompts.
    """

    DEFAULT_TIMEOUT_S = 300

    def __init__(self, model: str = "gpt-5.5", codex_path: Optional[str] = None):
        self.model = model
        # Windows npm puts the launchers at AppData\Roaming\npm\codex{.ps1,.cmd}.
        # shutil.which on Windows finds .cmd / .ps1 / .exe via PATHEXT, so try
        # that first; explicit codex_path overrides.
        resolved = codex_path or shutil.which("codex")
        if not resolved:
            # Fallback to the .cmd shim explicitly (shutil.which sometimes
            # misses .ps1-only entries depending on shell context).
            import os
            npm_prefix = os.environ.get("APPDATA")
            if npm_prefix:
                candidate = os.path.join(npm_prefix, "npm", "codex.cmd")
                if os.path.exists(candidate):
                    resolved = candidate
        if not resolved:
            raise RuntimeError(
                "`codex` CLI not found on PATH. Install with "
                "`npm install -g @openai/codex` and run `codex login`."
            )
        self.codex_path = resolved

    def _build_cmd(self, output_file: str) -> list[str]:
        return [
            self.codex_path, "exec",
            "--model", self.model,
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--ignore-user-config",
            "--ephemeral",
            "--output-last-message", output_file,
        ]

    def _embed_system(self, prompt: str, system: Optional[str]) -> str:
        """Codex CLI has no system-prompt flag — embed it as a block in the
        user prompt. Marker headers help the model honor the contract."""
        if not system:
            return prompt
        return (
            "## SYSTEM INSTRUCTIONS\n"
            f"{system}\n\n"
            "## USER MESSAGE\n"
            f"{prompt}"
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,           # ignored
        temperature: float = 0.0,         # ignored
        cached_system: bool = False,      # ignored
    ) -> str:
        import os
        import tempfile
        # Capture --output-last-message into a temp file (cleaner than parsing
        # the multi-block stdout that includes session metadata + token counts).
        fd, out_path = tempfile.mkstemp(suffix=".txt", prefix="codex-out-")
        os.close(fd)
        try:
            cmd = self._build_cmd(out_path)
            full_prompt = self._embed_system(prompt, system)
            # Pipe prompt through stdin — Windows has a ~32KB command-line
            # length limit (WinError 206) and council critique prompts with
            # embedded dossier JSON routinely exceed that. Codex `exec` reads
            # from stdin when no positional PROMPT arg is given (per --help).
            try:
                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self.DEFAULT_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"codex exec timed out after {self.DEFAULT_TIMEOUT_S}s") from e
            if result.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed (exit {result.returncode}): "
                    f"{(result.stderr or result.stdout)[:500]}"
                )
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except OSError as e:
                raise RuntimeError(f"codex output file unreadable: {e}") from e
            if not text:
                raise RuntimeError(
                    "codex returned empty final message "
                    f"(stdout tail: {result.stdout[-300:] if result.stdout else 'empty'})"
                )
            return text
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,           # ignored
        temperature: float = 0.0,         # ignored
        cached_system: bool = False,      # ignored
    ) -> dict:
        # Same retry pattern as Anthropic / ClaudeCodeCLI.
        json_instruction = "\n\nRespond with a single JSON object. No prose, no markdown fences."
        full_prompt = prompt + json_instruction
        raw = ""
        for attempt in range(2):
            raw = self.generate(full_prompt, system=system)
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return parsed
            log.warning("codex returned unparseable JSON (attempt %d): %r", attempt + 1, raw[:200])
        raise ValueError(f"codex failed to produce valid JSON after 2 attempts: {raw[:300]}")


# ── Google Gemini (council second voice — via Node shim) ─────────────────────

class GeminiLLM:
    """Google Gemini via a thin Node.js shim (`tools/node/gemini_critic.js`).

    Why Node, not the Python `google-genai` SDK: the Python SDK pulled a large
    `google-*` dep stack that was failing to install cleanly in the project
    venv; Node already exists in the toolchain (it ships `codex` for the third
    council voice) and `@google/generative-ai` is a small standalone package.
    Same subprocess pattern as CodexCLI — council code treats all three voices
    duck-typed.

    Requires:
      - Node.js on PATH
      - `npm install` having been run in `tools/node/`
      - `GOOGLE_API_KEY` in `.env` (forwarded to the child process)

    `cached_system` is silently ignored. Thinking-token footgun applies:
    Gemini 2.5+ models burn output budget on invisible thinking, so keep
    `max_tokens` ≥ 2048 for non-trivial prompts.
    """

    DEFAULT_TIMEOUT_S = 120

    def __init__(self, model: str, api_key: str, node_path: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        resolved = node_path or shutil.which("node")
        if not resolved:
            raise RuntimeError("`node` not found on PATH. Install Node.js.")
        self.node_path = resolved
        # Resolve shim script path relative to project root
        self.shim_path = config.PROJECT_ROOT / "tools" / "node" / "gemini_critic.js"
        if not self.shim_path.exists():
            raise RuntimeError(
                f"gemini_critic.js not found at {self.shim_path}. "
                "Run `npm install` in tools/node/."
            )

    def _invoke(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        import os
        req = {
            "system": system,
            "user": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": self.model,
            "json_mode": json_mode,
        }
        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = self.api_key
        try:
            result = subprocess.run(
                [self.node_path, str(self.shim_path)],
                input=json.dumps(req),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=self.DEFAULT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"gemini_critic.js timed out after {self.DEFAULT_TIMEOUT_S}s"
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"gemini_critic.js failed (exit {result.returncode}): "
                f"{(result.stderr or result.stdout)[:500]}"
            )
        return result.stdout.strip()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cached_system: bool = False,   # ignored
    ) -> str:
        return self._invoke(prompt, system, max_tokens, temperature, json_mode=False)

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        cached_system: bool = False,   # ignored
    ) -> dict:
        raw = ""
        for attempt in range(2):
            raw = self._invoke(prompt, system, max_tokens, temperature, json_mode=True)
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return parsed
            log.warning("gemini-node returned unparseable JSON (attempt %d): %r", attempt + 1, raw[:200])
        raise ValueError(f"Gemini failed to produce valid JSON after 2 attempts: {raw[:300]}")


# ── Ollama (fallback / offline) ──────────────────────────────────────────────

class OllamaLLM:
    def __init__(self, model: str, host: Optional[str] = None):
        import ollama
        self._ollama = ollama
        self.client = ollama.Client(host=host) if host else ollama
        self.model = model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        cached_system: bool = False,   # ignored for Ollama
    ) -> str:
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            kwargs["system"] = system
        resp = self.client.generate(**kwargs)
        return resp["response"]

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cached_system: bool = False,
    ) -> dict:
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            kwargs["system"] = system
        resp = self.client.generate(**kwargs)
        raw = resp["response"]
        parsed = _try_parse_json(raw)
        if parsed is None:
            raise ValueError(f"Ollama returned unparseable JSON: {raw[:300]}")
        return parsed


# ── Factory ──────────────────────────────────────────────────────────────────

def get_llm(model: Optional[str] = None) -> "AnthropicLLM | ClaudeCodeCLI | OllamaLLM":
    """Return an LLM instance for the configured provider.
    Override `model` to use a non-default (e.g. Sonnet for escalated cases)."""
    provider = config.LLM_PROVIDER
    if provider == "claude_code_cli":
        return ClaudeCodeCLI(model or config.LLM_MODEL_PRIMARY)
    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set; cannot use Anthropic provider")
        return AnthropicLLM(model or config.LLM_MODEL_PRIMARY, config.ANTHROPIC_API_KEY)
    if provider == "ollama":
        return OllamaLLM(model or config.OLLAMA_MODEL, host=config.OLLAMA_HOST)
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} "
        "(expected 'claude_code_cli', 'anthropic', or 'ollama')"
    )


# ── Subagent invocation ─────────────────────────────────────────────────────

class RateLimitError(RuntimeError):
    """Raised when Claude CLI returns api_error_status 429.
    Caller (or invoke_subagent's fallback layer) can catch and route to Codex."""


# Telemetry — caller can read `subagent_telemetry` after a run to see fallback rate.
subagent_telemetry: dict[str, int] = {
    "claude_ok": 0, "claude_rate_limited": 0, "claude_other_error": 0,
    "gemini_ok": 0, "gemini_rate_limited": 0, "gemini_error": 0,
    "codex_ok": 0, "codex_error": 0, "no_fallback_available": 0,
}

# Stages where Gemini is the PRIMARY (saves Claude tokens). Each pinned to its
# own API key from .env (GOOGLE_API_KEY_2/3/4). KEY_1 is reserved for council
# pushback and not used here. On Gemini failure (429/HTTP/parse), invoke_subagent
# falls back to Claude, then Codex.
_GEMINI_PRIMARY_AGENTS: dict[str, int] = {
    "nucleus-peer-context":     2,
    "pathology-news-diagnosis": 3,
    "fulcrum-falling-knife":    4,
}
_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiRateLimitError(RuntimeError):
    """Gemini API returned 429. invoke_subagent treats this like any other Gemini
    failure and falls back to Claude."""


def _invoke_gemini_subagent(name: str, prompt: str, *, timeout_s: int,
                             key_index: int, model: str = _GEMINI_MODEL) -> dict:
    """Call Gemini with the Codex system prompt for this agent. Returns parsed JSON.

    Raises:
      RuntimeError — no key in env, no system prompt, HTTP error, unparseable JSON.
      GeminiRateLimitError — HTTP 429.
    """
    import os as _os
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    key = _os.environ.get(f"GOOGLE_API_KEY_{key_index}", "").strip().strip("'").strip('"')
    if not key:
        raise RuntimeError(f"GOOGLE_API_KEY_{key_index} not set in env")

    # Reuse codex toml system prompts — same JSON contracts, no extra file.
    system = _load_codex_system_prompt(name)
    if system is None:
        raise RuntimeError(
            f"subagent {name!r}: no .codex/agents/{name}.toml — needed for Gemini system prompt"
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192,
                              "responseMimeType": "application/json"},
    }).encode("utf-8")
    req = _urlreq.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with _urlreq.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except _urlerr.HTTPError as e:
        if e.code == 429:
            raise GeminiRateLimitError(
                f"Gemini KEY_{key_index} 429 on {name!r}"
            ) from e
        msg = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Gemini HTTP {e.code} on {name!r}: {msg}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini call failed on {name!r}: {type(e).__name__}: {e}") from e

    try:
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Gemini response shape unexpected on {name!r}: {str(data)[:300]}"
        ) from e
    parsed = _try_parse_json(raw)
    if parsed is None:
        raise ValueError(
            f"subagent {name!r} (Gemini) output not valid JSON: {raw[:300]}"
        )
    return parsed


def _detect_rate_limit(stdout_text: str, stderr_text: str) -> bool:
    """True if Claude CLI output looks like a 429 / 5-hour limit hit."""
    blob = (stdout_text or "") + (stderr_text or "")
    return ("api_error_status\":429" in blob
            or "You've hit your limit" in blob
            or "5-hour limit" in blob
            or '"status":"limit_exceeded"' in blob)


def _invoke_claude_subagent(name: str, prompt: str, *, timeout_s: int,
                             claude_path: Optional[str]) -> dict:
    """Original Claude-only path. Raises RateLimitError on 429."""
    import shutil as _shutil
    resolved = claude_path or _shutil.which("claude")
    if not resolved:
        raise RuntimeError("`claude` CLI not found on PATH")
    cmd = [resolved, "-p", "--agent", name, "--no-session-persistence",
           "--tools", "", "--output-format", "json"]
    try:
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                encoding="utf-8", timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"subagent {name!r} timed out after {timeout_s}s") from e

    # 429 detection: CLI sometimes returns exit 0 with a 429-wrapped result (success
    # subtype, is_error=true) and sometimes exit 1. Check the body in both cases.
    if _detect_rate_limit(result.stdout, result.stderr):
        raise RateLimitError(
            f"subagent {name!r} rate-limited by Claude: "
            f"{(result.stderr or result.stdout)[:200]}"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"subagent {name!r} failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout)[:500]}"
        )
    try:
        wrapper = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"subagent {name!r} wrapper JSON unparseable: {result.stdout[:300]}"
        ) from e
    raw = wrapper.get("result", "")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"subagent {name!r} returned empty result "
            f"(is_error={wrapper.get('is_error')})"
        )
    parsed = _try_parse_json(raw)
    if parsed is None:
        raise ValueError(
            f"subagent {name!r} output is not valid JSON: {raw[:300]}"
        )
    return parsed


def _load_codex_system_prompt(name: str) -> Optional[str]:
    """Read `developer_instructions` from `.codex/agents/<name>.toml`. None if missing."""
    import tomllib as _tomllib
    from pathlib import Path as _Path
    p = _Path(__file__).resolve().parent.parent / ".codex" / "agents" / f"{name}.toml"
    if not p.exists():
        return None
    try:
        data = _tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("developer_instructions")


def _invoke_codex_subagent(name: str, prompt: str, *, timeout_s: int) -> dict:
    """Codex fallback path. Reads system prompt from .codex/agents/<name>.toml.
    Raises RuntimeError if the toml is missing or Codex fails."""
    system = _load_codex_system_prompt(name)
    if system is None:
        raise RuntimeError(
            f"subagent {name!r} has no Codex twin at .codex/agents/{name}.toml — "
            f"cannot fall back."
        )
    codex = CodexCLI(model="gpt-5.5")
    # Bump the default timeout for fallback so a slow Codex run doesn't double-fail
    codex.DEFAULT_TIMEOUT_S = max(timeout_s, codex.DEFAULT_TIMEOUT_S)
    raw = codex.generate(prompt, system=system)
    parsed = _try_parse_json(raw)
    if parsed is None:
        raise ValueError(
            f"subagent {name!r} (Codex fallback) output not valid JSON: {raw[:300]}"
        )
    return parsed


def invoke_subagent(
    name: str,
    prompt: str,
    *,
    timeout_s: int = 300,
    claude_path: Optional[str] = None,
    allow_codex_fallback: bool = True,
) -> dict:
    """Invoke a project subagent. Tries Claude first; on 429 transparently
    falls back to Codex (same agent name in .codex/agents/<name>.toml).

    Raises:
      RuntimeError — non-rate-limit CLI failure on Claude AND no Codex fallback,
                     or Codex itself failed.
      ValueError   — agent output couldn't be parsed as JSON (after fallback).
      RateLimitError — only if allow_codex_fallback=False and Claude 429s.
    """
    # ── Gemini-primary path (saves Claude tokens for Stage 2-4 agents) ────
    gemini_key_idx = _GEMINI_PRIMARY_AGENTS.get(name)
    if gemini_key_idx is not None:
        try:
            out = _invoke_gemini_subagent(name, prompt, timeout_s=timeout_s,
                                          key_index=gemini_key_idx)
            subagent_telemetry["gemini_ok"] += 1
            return out
        except GeminiRateLimitError:
            subagent_telemetry["gemini_rate_limited"] += 1
            log.warning("Gemini KEY_%d rate-limited on %r — falling back to Claude",
                        gemini_key_idx, name)
        except (RuntimeError, ValueError) as e:
            subagent_telemetry["gemini_error"] += 1
            log.warning("Gemini failed on %r (%s) — falling back to Claude",
                        name, str(e)[:120])
        # fall through to Claude below

    try:
        out = _invoke_claude_subagent(name, prompt, timeout_s=timeout_s,
                                      claude_path=claude_path)
        subagent_telemetry["claude_ok"] += 1
        return out
    except RateLimitError:
        subagent_telemetry["claude_rate_limited"] += 1
        if not allow_codex_fallback:
            raise
        log.warning("Claude rate-limited on %r — falling back to Codex", name)
        try:
            out = _invoke_codex_subagent(name, prompt, timeout_s=timeout_s)
            subagent_telemetry["codex_ok"] += 1
            return out
        except RuntimeError as e:
            # No fallback available, or Codex failed
            if "no Codex twin" in str(e):
                subagent_telemetry["no_fallback_available"] += 1
            else:
                subagent_telemetry["codex_error"] += 1
            raise
    except (RuntimeError, ValueError):
        subagent_telemetry["claude_other_error"] += 1
        raise


# ── Helpers ──────────────────────────────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction — handles markdown fences and trailing prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find first {...} block
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None
