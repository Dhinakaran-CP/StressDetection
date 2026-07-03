import re

with open('frontend/src/pages/Dashboard.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace analyzeMultimodal function entirely
old_analyze_regex = re.compile(r'const analyzeMultimodal = async \(\) => \{.*?^\s*};\s*^', re.MULTILINE | re.DOTALL)
new_analyze = '''  const analyzeMultimodal = async () => {
    setError(null);
    const formData = new FormData();
    if (faceImage) formData.append('face_image', faceImage);
    if (voiceFile) formData.append('voice_audio', voiceFile);
    if (eegData) formData.append('eeg_data', eegData);
    if (gsrData) formData.append('gsr_data', gsrData);
    if (eegFile) formData.append('eeg_file', eegFile);
    if (gsrFile) formData.append('gsr_file', gsrFile);

    const validationErrors = validateAnalysisInputs({
      faceFile: faceImage,
      voiceFile,
      eegData,
      gsrData
    });

    if (validationErrors.length > 0) {
      setError(validationErrors.join(' '));
      return;
    }

    setPhase('analyzing');
    setAnalysisPayload(formData);
    setPreviousResult(null);
    setCurrentResult(null);

    try {
      const response = await fetch(${API_BASE}/api/multimodal/analyze, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      const respErrs = validateAnalysisResponse(data);
      if (respErrs.length > 0) {
        throw new Error(respErrs.join(' '));
      }

      setCurrentResult(data);
      setPhase('result');
    } catch (err) {
      setError(err.message || 'Analysis failed');
      setPhase('idle');
    }
  };

  const handleRequestGame = useCallback(() => {
    setPhase('game');
  }, []);

  const handleGameComplete = useCallback(async () => {
    if (!analysisPayload) {
      setPhase('idle');
      return;
    }
    setPhase('reanalyzing');
    setPreviousResult(currentResult);

    try {
      const response = await fetch(${API_BASE}/api/multimodal/analyze, {
        method: 'POST',
        body: analysisPayload,
      });
      const data = await response.json();
      const respErrs = validateAnalysisResponse(data);
      if (respErrs.length > 0) {
        throw new Error(respErrs.join(' '));
      }
      setCurrentResult(data);
      setPhase('comparison');
    } catch (err) {
      console.error('Re-analysis failed:', err);
      setError('Re-analysis failed. Please try again.');
      setPhase('result');
    }
  }, [analysisPayload, currentResult]);
'''

# Wait, we must make sure useCallback is imported in Dashboard.js!
content = content.replace("import React, { useState, useRef, useEffect, useMemo } from \"react\";", "import React, { useState, useRef, useEffect, useMemo, useCallback } from \"react\";")

content = old_analyze_regex.sub(new_analyze, content, count=1)

# Fix lines near 1441
content = content.replace("disabled={analyzing || !serverOnline}", "disabled={(phase === 'analyzing' || phase === 'reanalyzing') || !serverOnline}")

old_alerts = '''                {gameCompleted && !postGameStress && (
                  <div className="alert insights-fade" style={{ background: 'var(--primary-color)', color: '#000', borderRadius: '12px', padding: '1rem', marginTop: '1rem', boxShadow: '0 4px 15px rgba(0,242,255,0.3)' }}>
                    <strong>Great job!</strong> You completed the activity. Please re-upload your data and click "Analyze Stress" to evaluate your improvement!
                  </div>
                )}

                {postGameStress && preGameStress && (
                  <div className="alert insights-fade" style={{ background: 'var(--accent-light-bg)', border: '1px solid var(--primary-color)', borderRadius: '12px', padding: '1rem', marginTop: '1rem' }}>
                    <strong>?? Re-Evaluation Complete!</strong>
                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '1.1rem' }}>
                      Your stress level changed from <strong>{preGameStress.stress_percentage}%</strong> to <strong style={{ color: postGameStress.stress_percentage < preGameStress.stress_percentage ? '#4CAF50' : 'inherit' }}>{postGameStress.stress_percentage}%</strong>!
                    </p>
                  </div>
                )}'''

content = content.replace(old_alerts, "")

with open('frontend/src/pages/Dashboard.js', 'w', encoding='utf-8') as f:
    f.write(content)
print("Final fix applied.")
