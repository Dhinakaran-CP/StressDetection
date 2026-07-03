import re

with open('frontend/src/pages/Dashboard.js', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace the result section
# We find the start of {/* Error Display */}
start_idx = content.find('{/* Error Display */}')
if start_idx == -1:
    print("Could not find {/* Error Display */}")
    exit(1)

# We find {/* Input Section */}
end_idx = content.find('{/* Input Section */}')
if end_idx == -1:
    print("Could not find {/* Input Section */}")
    exit(1)

new_ui_section = '''{/* Error Display */}
          {error && (
            <div className="row mb-4">
              <div className="col-12">
                <div style={{
                  background: 'rgba(199, 69, 69, 0.1)',
                  border: '2px solid #c74545',
                  borderRadius: '8px',
                  padding: '1rem',
                  color: '#c74545'
                }}>
                  <strong>?? Error:</strong> {error}
                </div>
              </div>
            </div>
          )}

          {/* Loading states */}
          {(phase === 'analyzing' || phase === 'reanalyzing') && (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <div className="spinner-border text-primary mb-3" role="status" style={{width: '3rem', height: '3rem'}}></div>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.9rem' }}>
                {phase === 'reanalyzing'
                  ? '?? Re-analyzing after recovery...'
                  : '?? Analyzing stress indicators...'}
              </div>
            </div>
          )}

          {/* Result panel */}
          {(phase === 'result' || phase === 'comparison') && currentResult && (
            <AnalysisPanel
              result={currentResult}
              previousResult={phase === 'comparison' ? previousResult : null}
              onRequestGame={handleRequestGame}
            />
          )}

          {/* Game panel */}
          {phase === 'game' && (
            <GamePanel
              stressLevel={currentResult?.stress_level}
              onGameComplete={handleGameComplete}
              onDismiss={() => setPhase('result')}
            />
          )}

          {/* Re-analyze button after comparison */}
          {phase === 'comparison' && (
            <div style={{ marginTop: 16, textAlign: 'center' }}>
              <button onClick={() => {
                setPhase('idle');
                setCurrentResult(null);
                setPreviousResult(null);
                clearAll();
              }}
                style={{ background: 'none', border: '1px solid rgba(255,255,255,0.15)',
                         color: 'rgba(255,255,255,0.5)', borderRadius: 8,
                         padding: '8px 20px', cursor: 'pointer', fontSize: '0.82rem' }}>
                Start New Analysis
              </button>
            </div>
          )}

          '''

content = content[:start_idx] + new_ui_section + content[end_idx:]

# 2. Wrap input section in phase === 'idle'
input_section_str = "{/* Input Section */}"
input_section_idx = content.find(input_section_str)
if input_section_idx != -1:
    content = content[:input_section_idx] + "{phase === 'idle' && (\n          " + content[input_section_idx:]

# 3. Find where the input section ends. The old code had:
#                 {postGameStress && preGameStress && ( ... )}
#               </div>
#             </div>
#           </div>
#         </>
#       )}
#       {/* FOOTER */}

# Let's replace {analyzing ? 'Analyzing...' : 'Analyze Stress'} with 'Analyze Stress'
content = content.replace("{analyzing ? 'Analyzing...' : 'Analyze Stress'}", "'Analyze Stress'")

# Replace old alert boxes for game completion
old_alerts = '''                {gameCompleted && !postGameStress && (
                  <div className="alert insights-fade" style={{ background: 'var(--primary-color)', color: '#000', borderRadius: '12px', padding: '1rem', marginTop: '1rem', boxShadow: '0 4px 15px rgba(0,242,255,0.3)' }}>
                    <strong>Great job!</strong> You completed the activity. Please re-upload your data and click "Analyze Stress" to evaluate your improvement!
                  </div>
                )}

                {postGameStress && preGameStress && (
                  <div className="alert insights-fade" style={{ background: 'var(--accent-light-bg)', border: '1px solid var(--primary-color)', borderRadius: '12px', padding: '1rem', marginTop: '1rem' }}>
                    <strong>Follow-up Analysis:</strong> Before activity: {preGameStress.percentage.toFixed(1)}% | After activity: {postGameStress.percentage.toFixed(1)}%
                    <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>
                      {postGameStress.percentage < preGameStress.percentage 
                        ? '?? Stress decreased! The activity helped.' 
                        : '?? Stress did not decrease. You may need more rest.'}
                    </div>
                  </div>
                )}'''

content = content.replace(old_alerts, "")

with open('frontend/src/pages/Dashboard.js', 'w', encoding='utf-8') as f:
    f.write(content)
print("JSX updated successfully.")
