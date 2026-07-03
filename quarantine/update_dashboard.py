import re

with open('frontend/src/pages/Dashboard.js', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add imports
content = content.replace(
    'import StressChatbot from "../components/StressChatbot";',
    'import StressChatbot from "../components/StressChatbot";\nimport { validateAnalysisInputs, validateAnalysisResponse } from "../utils/validateInputs";'
)

# 2. Update states
state_old = '''  // API interaction states
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);

  // Recovery interaction states
  const [stressLevel, setStressLevel] = useState("Moderate");
  const [isGameActive, setIsGameActive] = useState(false);
  const [selectedActivity, setSelectedActivity] = useState(null);
  const [recoveryScore, setRecoveryScore] = useState(72);
  const [calmStreak, setCalmStreak] = useState(8);
  const [reward, setReward] = useState(null);
  const [gameCompleted, setGameCompleted] = useState(false);
  const [preGameStress, setPreGameStress] = useState(null);
  const [postGameStress, setPostGameStress] = useState(null);'''

state_new = '''  // Phase state machine
  const [phase, setPhase] = useState('idle'); // 'idle' | 'analyzing' | 'result' | 'game' | 'reanalyzing' | 'comparison'
  const [currentResult, setCurrentResult] = useState(null);
  const [previousResult, setPreviousResult] = useState(null);
  const [analysisPayload, setAnalysisPayload] = useState(null);

  // Legacy UI states
  const [error, setError] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [recoveryScore, setRecoveryScore] = useState(72);
  const [calmStreak, setCalmStreak] = useState(8);
  const [reward, setReward] = useState(null);'''

content = content.replace(state_old, state_new)

# 3. Update analyzeMultimodal
analyze_old = '''  const analyzeMultimodal = async () => {
    if (!faceImage && !voiceFile && !eegData && !gsrData && !eegFile && !gsrFile) {
      setError('Please provide at least one input (image, audio, EEG, or GSR data)');
      return;
    }
    setAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      if (faceImage) formData.append('face_image', faceImage);
      if (voiceFile) formData.append('voice_audio', voiceFile);
      if (eegData) formData.append('eeg_data', eegData);
      if (gsrData) formData.append('gsr_data', gsrData);
      if (eegFile) formData.append('eeg_file', eegFile);
      if (gsrFile) formData.append('gsr_file', gsrFile);

      const response = await fetch(${API_BASE}/api/multimodal/analyze, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.status === 'success') {
        if (gameCompleted) {
          setPostGameStress(data);
          setGameCompleted(false);
        } else {
          setPreGameStress(null);
          setPostGameStress(null);
        }
        setResult(data);
      } else {
        setError(data.message || 'Analysis failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message + '. Is the server running on port 5000?');
    } finally {
      setAnalyzing(false);
    }
  };'''

analyze_new = '''  const analyzeMultimodal = async () => {
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
  }, [analysisPayload, currentResult]);'''

content = content.replace(analyze_old, analyze_new)

with open('frontend/src/pages/Dashboard.js', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully.")
