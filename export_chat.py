"""Utility script to export the current Antigravity conversation transcript to Markdown.

Usage:
    python export_chat.py [--output chat_export.md]
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

CONVERSATION_ID = "4caf6ce1-dc66-4562-8f19-dfd5862efa1b"
APP_DATA_DIR = Path(r"C:\Users\PC-47\.gemini\antigravity-ide")
TRANSCRIPT_PATH = APP_DATA_DIR / "brain" / CONVERSATION_ID / ".system_generated" / "logs" / "transcript_full.jsonl"


def extract_user_message(raw_text: str) -> str:
    """Extracts the clean user prompt from system prompt wrappers if present."""
    match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def export_conversation(output_path: Path):
    if not TRANSCRIPT_PATH.exists():
        fallback = TRANSCRIPT_PATH.with_name("transcript.jsonl")
        if not fallback.exists():
            raise FileNotFoundError(f"Transcript file not found at {TRANSCRIPT_PATH}")
        path_to_read = fallback
    else:
        path_to_read = TRANSCRIPT_PATH

    print(f"Reading conversation logs from: {path_to_read}...")

    with open(path_to_read, "r", encoding="utf-8") as f:
        steps = [json.loads(line) for line in f if line.strip()]

    md_lines = [
        "# Antigravity Chat Export - Credit Ledger Project\n",
        f"> **Exported Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> **Conversation ID**: `{CONVERSATION_ID}`\n",
        "---\n",
    ]

    turn_count = 1
    for step in steps:
        step_type = step.get("type")
        content = step.get("content", "")

        if step_type == "USER_INPUT":
            user_msg = extract_user_message(content)
            if user_msg:
                md_lines.append(f"## 👤 User (Turn {turn_count})\n")
                md_lines.append(f"{user_msg}\n")
                md_lines.append("---\n")
                turn_count += 1

        elif step_type == "PLANNER_RESPONSE":
            if content and content.strip():
                md_lines.append("## 🤖 Antigravity Assistant\n")
                md_lines.append(f"{content.strip()}\n")
                md_lines.append("---\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[OK] Chat exported successfully to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Antigravity chat history to Markdown.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="chat_export.md",
        help="Target output file path (default: chat_export.md)",
    )
    args = parser.parse_args()
    export_conversation(Path(args.output))
