import re

with open('frontend/src/components/AnalysisPanel.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace generateContextualSummary extraction
content = content.replace('const { stress_level, explainability, face_score, voice_score, physio_score } = result;',
'''const stress_level = result?.stress_level;
  const explainability = result?.explainability;
  const individual = result?.individual_predictions || {};
  const face_score = individual.facial;
  const voice_score = individual.voice;
  const physio_score = individual.physiological;''')

# Replace component destructing
content = re.sub(r'  const \{\n    stress_level, fused_score, confidence_score, inference_ms,\n    face_score, voice_score, physio_score,\n    explainability, modality_weights,\n  \} = result;',
'''  const stress_level = result?.stress_level || "Low";
  const fused_score = result?.stress_probability || (result?.percentage ? result.percentage / 100 : 0);
  const confidence_score = result?.confidence || 0.9;
  const explainability = result?.explainability;
  const individual = result?.individual_predictions || {};
  const face_score = individual.facial;
  const voice_score = individual.voice;
  const physio_score = individual.physiological;''', content)

with open('frontend/src/components/AnalysisPanel.jsx', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated AnalysisPanel.jsx")
