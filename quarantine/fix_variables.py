import re

with open('frontend/src/pages/Dashboard.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace setResult with setCurrentResult
content = re.sub(r'\bsetResult\b', 'setCurrentResult', content)
content = re.sub(r'\bresult\b(?!Visuals)(?!s)(?!:)(?!\.percentage)', 'currentResult', content)
# Ensure we don't accidentally rename 'currentcurrentResult'
content = content.replace('currentcurrentResult', 'currentResult')

# Replace setAnalyzing with setPhase('analyzing') etc.
content = re.sub(r'setAnalyzing\(true\);?', "setPhase('analyzing');", content)
content = re.sub(r'setAnalyzing\(false\);?', "if(phase==='analyzing') setPhase('idle');", content)

# Remove the old gameCompleted logic block inside analyze handlers
old_game_block = r'''\s*if\s*\(gameCompleted\)\s*\{\s*setPostGameStress\([^)]+\);\s*setGameCompleted\(false\);\s*\}\s*else\s*\{\s*setPreGameStress\(null\);\s*setPostGameStress\(null\);\s*\}'''
content = re.sub(old_game_block, '', content)

# Remove lingering setPreGameStress, setPostGameStress, setGameCompleted
content = re.sub(r'setPreGameStress\([^)]*\);?', '', content)
content = re.sub(r'setPostGameStress\([^)]*\);?', '', content)
content = re.sub(r'setGameCompleted\([^)]*\);?', '', content)
content = re.sub(r'setStressLevel\([^)]*\);?', '', content)
content = re.sub(r'setIsGameActive\([^)]*\);?', '', content)
content = re.sub(r'setSelectedActivity\([^)]*\);?', '', content)

# Remove the old handleActivityComplete if it exists
content = re.sub(r'const handleActivityComplete = \([^)]*\) => \{[^}]*\};?', '', content)

# Fix undefined stressLevel and result in UI
content = content.replace("stressLevel={stressLevel}", "stressLevel={currentResult?.stress_level || 'Moderate'}")
content = content.replace("result={currentResult} ", "result={currentResult} ")

# Remove the old result render that caused error 'result' is not defined (Line 745 etc inside old clearAll)
# Actually let's just make clearAll properly clear phase
content = content.replace('setCurrentResult(null);', "setCurrentResult(null); setPhase('idle');")

with open('frontend/src/pages/Dashboard.js', 'w', encoding='utf-8') as f:
    f.write(content)
print("Variables fixed.")
