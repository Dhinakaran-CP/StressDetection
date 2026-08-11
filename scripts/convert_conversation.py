"""
Robust AI Conversation → Markdown Converter
=============================================
Reads a raw exported conversation (TXT or JSON) and produces
a lossless Markdown archive suitable for LLM context injection.

Usage
-----
    python convert_conversation.py --input chat_export.txt --output conversation.md
    python convert_conversation.py --input chat_export.json --output conversation.md
    python convert_conversation.py --input chat_export.txt    # auto-names output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────

def parse_txt(filepath: Path) -> list[dict]:
    """
    Parse a plain-text conversation file.

    Expected format
    ---------------
        User: <message text ...>

        Assistant: <message text ...>

    * Messages are delimited by **User:** or **Assistant:** labels that
      appear at the start of a line (case-insensitive, colon required).
    * Everything after the label (trimmed) is the message body.
    * Blank lines between messages are ignored.
    * Code fences, indentation, special characters are preserved verbatim.
    """
    raw = filepath.read_text(encoding="utf-8")
    messages: list[dict] = []
    lines = raw.splitlines()

    # ── 1. Identify message boundaries ──────────────────────────────
    label_pattern = re.compile(r"^(User|Assistant)\s*:\s*(.*)", re.IGNORECASE)
    boundaries: list[tuple[int, str, str]] = []  # (line_index, role, first_line_text)

    for i, line in enumerate(lines):
        m = label_pattern.match(line)
        if m:
            role = m.group(1).capitalize()  # "User" or "Assistant"
            text = m.group(2)
            boundaries.append((i, role, text))

    if not boundaries:
        # Fallback: treat the whole file as a single Assistant message
        return [{"role": "Assistant", "body": raw.rstrip()}]

    # ── 2. Build message bodies from boundaries ─────────────────────
    for idx, (start, role, first_text) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body_parts = [first_text]
        body_parts.extend(lines[start + 1:end])
        body = "\n".join(body_parts).strip()
        messages.append({"role": role, "body": body})

    return messages


def parse_json(filepath: Path) -> list[dict]:
    """
    Parse a JSON conversation file.

    Supported schemas
    -----------------
    1. Top-level array of objects with keys ``role`` / ``content``:
           [{"role": "user", "content": "..."}, ...]

    2. Object with a ``messages`` key holding the array:
           {"messages": [{"role": "user", "content": "..."}, ...]}

    3. Object with a ``conversation`` key holding the array:
           {"conversation": [{"role": "user", "content": "..."}, ...]}

    * ``role`` is normalised to "User" / "Assistant".
    * Unknown roles are mapped to "Other".
    """
    raw = json.loads(filepath.read_text(encoding="utf-8"))

    # Detect structure
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ("messages", "conversation", "chat", "history"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break
        else:
            raise ValueError(
                "JSON root is an object but contains no recognised conversation key. "
                "Expected 'messages', 'conversation', 'chat', or 'history'."
            )
    else:
        raise TypeError("JSON root must be an array or object.")

    # Normalise
    role_map = {"user": "User", "assistant": "Assistant", "system": "System"}
    messages: list[dict] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        role_raw = entry.get("role", entry.get("from", "")).strip().lower()
        role = role_map.get(role_raw, role_raw.capitalize())
        content = entry.get("content", entry.get("text", entry.get("value", "")))
        if content:
            messages.append({"role": role, "body": str(content)})

    return messages


def detect_and_parse(filepath: Path) -> list[dict]:
    """
    Auto-detect file format by extension and delegate to the
    appropriate parser.
    """
    suffix = filepath.suffix.lower()
    if suffix == ".json":
        return parse_json(filepath)
    elif suffix in (".txt", ".md", ""):
        return parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


# ──────────────────────────────────────────────────────────────────────
# Markdown generator
# ──────────────────────────────────────────────────────────────────────

def escape_code_fences(body: str) -> str:
    """
    Ensure code fences inside the message body render correctly
    when nested inside a Markdown block quote or list.

    This is a no-op for top-level messages — we output them directly.
    """
    return body


def build_markdown(messages: list[dict], input_name: str, filepath: Path) -> str:
    """
    Build the final Markdown document from parsed messages.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(messages)
    user_count = sum(1 for m in messages if m["role"] == "User")
    assistant_count = sum(1 for m in messages if m["role"] == "Assistant")

    lines: list[str] = []
    lines.append("# Conversation Archive")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Generated** : {now}")
    lines.append(f"- **Input File** : `{input_name}`")
    lines.append(f"- **Total Messages** : {total}")
    lines.append(f"- **User Messages** : {user_count}")
    lines.append(f"- **Assistant Messages** : {assistant_count}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for idx, msg in enumerate(messages, start=1):
        role = msg["role"]
        body = msg["body"]
        lines.append(f"## Conversation {idx}")
        lines.append("")
        lines.append(f"### {role}")
        lines.append("")
        lines.append(body)
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("# End of Conversation")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an exported AI conversation to a lossless Markdown archive."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input conversation file (.txt or .json).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for the output Markdown file (default: input name + .md).",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["auto", "txt", "json"],
        default="auto",
        help="Force parser format (default: auto-detect from extension).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse
    try:
        if args.format == "json":
            messages = parse_json(input_path)
        elif args.format == "txt":
            messages = parse_txt(input_path)
        else:
            messages = detect_and_parse(input_path)
    except Exception as exc:
        print(f"Error: failed to parse input: {exc}", file=sys.stderr)
        sys.exit(1)

    if not messages:
        print("Warning: no messages found in input.", file=sys.stderr)
        sys.exit(0)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_path.stem
        out_path = input_path.with_name(f"{stem}.md")

    # Build
    try:
        markdown = build_markdown(messages, input_path.name, input_path)
    except Exception as exc:
        print(f"Error: failed to generate Markdown: {exc}", file=sys.stderr)
        sys.exit(1)

    # Write
    out_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] Saved: {out_path}")
    print(f"  Messages: {len(messages)} ({sum(1 for m in messages if m['role'] == 'User')} user, "
          f"{sum(1 for m in messages if m['role'] == 'Assistant')} assistant)")


if __name__ == "__main__":
    main()