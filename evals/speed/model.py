"""The model-format pieces an HTTP client needs, copied from sharar at 2890ad1 (sharar/tokenizer.py,
sharar/api/chat_template.py and sharar/api/parsers/{reasoning,tools}.py), and the pinned snapshot.

The benchmarks tokenize and render chat templates on the client so every engine gets identical
token IDs; they read the pinned Qwen3.8-27B snapshot (tokenizer.json, chat_template.jinja) from the
Hugging Face cache: --snapshot, else $SHARAR_HF_CACHE, $HF_HUB_CACHE or $HF_HOME/hub.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

MODEL_ID = "RadixArk/Qwen3.8-27B-NVFP4"
MODEL_REVISION = "319f741cce68d7914884900c138a1fbb70a42f30"
MAX_CONTEXT = 262144


def snapshot_dir(cache: str | os.PathLike | None = None) -> Path:
    """The pinned snapshot directory in a Hugging Face cache."""
    roots = [cache] if cache else [os.environ.get("SHARAR_HF_CACHE"), os.environ.get("HF_HUB_CACHE"),
                                   os.environ.get("HF_HOME") and Path(os.environ["HF_HOME"]) / "hub",
                                   "/home/yke/data/huggingface-qwen38-nvfp4-dspark"]
    name = "models--" + MODEL_ID.replace("/", "--")
    for root in filter(None, roots):
        for base in (Path(root), Path(root) / "hub"):
            snap = base / name / "snapshots" / MODEL_REVISION
            if (snap / "tokenizer.json").is_file() and (snap / "chat_template.jinja").is_file():
                return snap
    raise FileNotFoundError(f"pinned snapshot {MODEL_ID}@{MODEL_REVISION[:7]} not found; set --snapshot or SHARAR_HF_CACHE")


# ---------------------------------------------------------------- tokenizer (sharar/tokenizer.py)

class SnapshotTokenizer:
    def __init__(self, snap: Path):
        self._tok = Tokenizer.from_file(str(snap / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        # The chat template already carries every special token, so never add any.
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)


# ---------------------------------------------------------------- chat template (sharar/api/chat_template.py)

def _raise_exception(message: str):
    raise jinja2.exceptions.TemplateError(message)


class ChatTemplate:
    def __init__(self, snap: Path):
        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            extensions=["jinja2.ext.loopcontrols"],
            undefined=jinja2.Undefined,
        )
        env.globals["raise_exception"] = _raise_exception
        env.filters["tojson"] = jinja2.filters.do_tojson if hasattr(jinja2.filters, "do_tojson") else None
        self._template = env.from_string((snap / "chat_template.jinja").read_text())

    def render(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: list | None = None,
        **kwargs,
    ) -> str:
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
            **kwargs,
        )


# ---------------------------------------------------------------- reasoning (sharar/api/parsers/reasoning.py)

THINK_START = "<think>"
THINK_END = "</think>"


class ReasoningParser:
    def __init__(self, in_reasoning: bool):
        self.in_reasoning = in_reasoning
        self.reasoning = ""
        self.content = ""
        self._pending = ""  # tail that might be the start of a tag
        self._seen_any = False

    def feed(self, delta: str) -> tuple[str, str]:
        """Returns (reasoning_delta, content_delta) for this piece of text."""
        text = self._pending + delta
        self._pending = ""
        r_out, c_out = "", ""
        if not self._seen_any and text:
            self._seen_any = True
            if text.startswith(THINK_START):
                self.in_reasoning = True
                text = text[len(THINK_START):]
            elif THINK_START.startswith(text):
                self._pending = text
                return "", ""
        while text:
            if self.in_reasoning:
                idx = text.find(THINK_END)
                if idx >= 0:
                    r_out += text[:idx]
                    text = text[idx + len(THINK_END):]
                    self.in_reasoning = False
                    continue
                keep = _partial_suffix(text, THINK_END)
                r_out += text[: len(text) - keep]
                self._pending = text[len(text) - keep:]
                text = ""
            else:
                c_out += text
                text = ""
        self.reasoning += r_out
        self.content += c_out
        return r_out, c_out

    def finish(self) -> tuple[str, str]:
        """Flush any held-back tail as whichever side we are on."""
        tail, self._pending = self._pending, ""
        if not tail:
            return "", ""
        if self.in_reasoning:
            self.reasoning += tail
            return tail, ""
        self.content += tail
        return "", tail


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of text that is a proper prefix of tag."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


# ---------------------------------------------------------------- tool calls (sharar/api/parsers/tools.py)

BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNC = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
TOOL_CALL_START = "<tool_call>"


def has_tool_call(text: str) -> bool:
    return TOOL_CALL_START in text


def _cast(value: str, schema: dict | None):
    if not schema:
        return value
    t = schema.get("type")
    v = value.strip()
    try:
        if t == "integer":
            return int(v)
        if t == "number":
            return float(v)
        if t == "boolean":
            return v.lower() in ("true", "1", "yes")
        if t in ("object", "array"):
            return json.loads(v)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def parse_tool_calls(text: str, tools: list[dict] | None) -> tuple[str, list[dict]]:
    """Returns (content_before_calls, openai_tool_calls)."""
    if not has_tool_call(text):
        return text, []
    schemas: dict[str, dict] = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        params = (fn.get("parameters") or {}).get("properties") or {}
        schemas[fn.get("name", "")] = params
    calls = []
    for block in BLOCK.finditer(text):
        m = FUNC.search(block.group(1))
        if not m:
            continue
        name = m.group(1).strip()
        args = {}
        for pm in PARAM.finditer(m.group(2)):
            args[pm.group(1).strip()] = _cast(pm.group(2), schemas.get(name, {}).get(pm.group(1).strip()))
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    content = text[: text.find(TOOL_CALL_START)].rstrip()
    return content, calls
