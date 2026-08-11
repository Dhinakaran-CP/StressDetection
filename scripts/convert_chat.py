from pathlib import Path
from datetime import datetime

# ===========================
# Configuration
# ===========================

INPUT_FILE = "chat_export.txt"      # Paste your entire chat here
OUTPUT_FILE = "Complete_Conversation.md"

TITLE = "Complete AI Research Conversation"
DESCRIPTION = (
    "This document contains the full conversation history, "
    "including prompts and assistant responses, preserved in chronological order."
)

# ===========================
# Read Conversation
# ===========================

input_path = Path(INPUT_FILE)

if not input_path.exists():
    raise FileNotFoundError(
        f"{INPUT_FILE} not found.\n"
        "Create it and paste your complete conversation into it."
    )

conversation = input_path.read_text(
    encoding="utf-8",
    errors="ignore"
)

# ===========================
# Markdown Generation
# ===========================

markdown = f"""# {TITLE}

---

## Metadata

- Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Source    : Chat Export
- Format    : Markdown
- Description : {DESCRIPTION}

---

# Conversation

{conversation}

---

# End of Conversation
"""

# ===========================
# Save
# ===========================

Path(OUTPUT_FILE).write_text(markdown, encoding="utf-8")

print(f"Markdown saved as: {OUTPUT_FILE}")