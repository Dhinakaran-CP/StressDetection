import re

with open('frontend/src/pages/Dashboard.js', 'r', encoding='utf-8') as f:
    content = f.read()

replacement = '''    try {
      const response = await fetch(${API_BASE}/api/multimodal/analyze, {
        method: 'POST',
        body: formData,
      });
      const data = await validateAnalysisResponse(response);
      setCurrentResult(data);
      
      // Auto-trigger game for high/extreme stress
      if (data.stress_level === 'Extreme' || data.stress_level === 'High') {
        setPhase('result');
        setTimeout(() => {
          if (window.confirm("High stress detected. Would you like to play a quick relaxation game to reduce stress?")) {
            setPhase('game');
          }
        }, 1500);
      } else {
        setPhase('result');
      }
    } catch (err) {'''

content = re.sub(r'    try \{\n      const response = await fetch.*?setCurrentResult\(data\);\n      setPhase\(\'result\'\);\n    \} catch \(err\) \{', replacement, content, flags=re.DOTALL)

with open('frontend/src/pages/Dashboard.js', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Dashboard.js with auto game trigger")
