import re
with open(r'C:\Users\KISHO\.gemini\antigravity-ide\brain\ac643bd6-8eb3-4e69-9c61-859480e48a58\.system_generated\steps\1891\content.md', 'r', encoding='utf-8') as f:
    text = f.read()
matches = re.findall(r'href="/Saathviga9605/Multimodal-Stress-Detection/blob/[^"]+"', text)
for m in set(matches):
    print(m)
