import os
import shutil
import json
import re

# Paths
brain_dir = r"C:\Users\KISHO\.gemini\antigravity-ide\brain\587eefb6-bcb1-4a02-915a-cbd744084cdd"
workspace_dir = r"E:\Document\GitHub\StressDetectionUsingML\early_fusion"

reports_dir = os.path.join(workspace_dir, "reports")
figures_dir = os.path.join(reports_dir, "figures")
scratch_dest = os.path.join(workspace_dir, "scripts", "scratch")
logs_dest = os.path.join(workspace_dir, "logs")

# 1. Create target directories
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(scratch_dest, exist_ok=True)
os.makedirs(logs_dest, exist_ok=True)

print("Folders initialized.")

# 2. Copy Markdown files
md_files = ["critical_analysis_report.md", "implementation_plan.md", "synchronization_report.md", "task.md", "walkthrough.md"]
for f in md_files:
    src_path = os.path.join(brain_dir, f)
    dst_path = os.path.join(reports_dir, f)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied document: {f}")

# 3. Copy Media files
for f in os.listdir(brain_dir):
    if f.startswith("media__") and f.endswith(".png"):
        src_path = os.path.join(brain_dir, f)
        dst_path = os.path.join(figures_dir, f)
        shutil.copy2(src_path, dst_path)
        print(f"Copied figure: {f}")

# 4. Copy Scratch files
scratch_src = os.path.join(brain_dir, "scratch")
if os.path.exists(scratch_src):
    for f in os.listdir(scratch_src):
        if f.endswith(".py"):
            src_path = os.path.join(scratch_src, f)
            dst_path = os.path.join(scratch_dest, f)
            shutil.copy2(src_path, dst_path)
            print(f"Copied scratch script: {f}")

# 5. Copy raw JSONL logs
log_src_dir = os.path.join(brain_dir, ".system_generated", "logs")
if os.path.exists(log_src_dir):
    for f in ["transcript.jsonl", "transcript_full.jsonl"]:
        src_path = os.path.join(log_src_dir, f)
        if os.path.exists(src_path):
            dst_name = "chat_transcript.jsonl" if f == "transcript.jsonl" else "chat_transcript_full.jsonl"
            shutil.copy2(src_path, os.path.join(logs_dest, dst_name))
            print(f"Copied log file: {dst_name}")

# 6. Parse transcript.jsonl and convert to markdown chat history
transcript_path = os.path.join(log_src_dir, "transcript.jsonl")
if os.path.exists(transcript_path):
    print("Generating human-readable Chat History Markdown...")
    chat_entries = []
    
    with open(transcript_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                # Capture User Requests
                if data.get("type") == "USER_INPUT" and data.get("source") == "USER_EXPLICIT":
                    content = data.get("content", "")
                    # Extract request portion
                    req_match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
                    if req_match:
                        req_text = req_match.group(1).strip()
                    else:
                        req_text = content.strip()
                    chat_entries.append({"role": "User", "text": req_text, "step": data.get("step_index")})
                    
                # Capture Model Responses
                elif data.get("type") == "PLANNER_RESPONSE" and data.get("source") == "MODEL" and data.get("status") == "DONE":
                    content = data.get("content", "")
                    if content:
                        chat_entries.append({"role": "Assistant", "text": content.strip(), "step": data.get("step_index")})
            except Exception as e:
                pass
                
    # Sort entries by step index
    chat_entries.sort(key=lambda x: x["step"])
    
    # Write to chat_history.md
    md_content = "# Antigravity Chat History Transcript\n\nThis document logs the step-by-step engineering conversation containing project requests, design choices, analysis, and execution scripts.\n\n---\n\n"
    for entry in chat_entries:
        role = entry["role"]
        text = entry["text"]
        md_content += f"## 👤 {role} (Step {entry['step']})\n\n{text}\n\n---\n\n"
        
    chat_history_path = os.path.join(reports_dir, "chat_history.md")
    with open(chat_history_path, "w", encoding="utf-8") as file:
        file.write(md_content)
    print("Successfully compiled and wrote reports/chat_history.md!")

print("All asset transfers finalized successfully.")
