import re

with open('frontend/src/pages/Dashboard.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace currentResult={currentResult} with result={currentResult} for AnalysisPanel
content = content.replace('<AnalysisPanel\n              currentResult={currentResult}', '<AnalysisPanel\n              result={currentResult}')

# Add InsightCards and CopilotMessage to the result view
replacement = '''          {/* Result panel */}
          {(phase === 'currentResult' || phase === 'comparison' || phase === 'result') && currentResult && (
            <div className="result-view">
              <AnalysisPanel
                result={currentResult}
                previousResult={phase === 'comparison' ? previousResult : null}
                onRequestGame={handleRequestGame}
              />
              <div className="row mt-4">
                <div className="col-md-6 mb-4">
                  <InsightCards result={currentResult} />
                </div>
                <div className="col-md-6 mb-4">
                  <CopilotMessage result={currentResult} />
                </div>
              </div>
            </div>
          )}'''

content = re.sub(r'          \{\/\* Result panel \*\/\}.*?          \)', replacement, content, flags=re.DOTALL)

with open('frontend/src/pages/Dashboard.js', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Dashboard.js")
