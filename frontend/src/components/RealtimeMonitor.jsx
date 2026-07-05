import React, { useEffect, useState, useRef } from 'react';
import { API_BASE } from '../config';
import FaceStream from './FaceStream';
import WaveformRecorder from './WaveformRecorder';
import CalibrationWizard from './CalibrationWizard';

const LEVEL_COLOR = { 
  Low: 'var(--primary-color)',       // Dynamic theme primary color
  Moderate: '#FF9800',  // Orange
  High: '#F44336'       // Red
};

function IndicatorBar({ label, value, scale = 1.0, format, extra, valueColor }) {
  const percent = Math.min(100, Math.max(0, (value / scale) * 100));
  const barColor = valueColor || (percent > 75 ? '#F44336' : percent > 40 ? '#FF9800' : 'var(--primary-color)');
  
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-color)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>{label}</span>
          {extra}
        </div>
        <span style={{ fontWeight: 'bold', fontFamily: 'monospace', color: barColor }}>{format(value)}</span>
      </div>
      <div style={{ width: '100%', height: 6, background: 'var(--bar-bg)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ 
          width: `${percent}%`, 
          height: '100%', 
          background: barColor, 
          borderRadius: 3, 
          transition: 'width 0.3s ease-out, background-color 0.3s ease' 
        }} />
      </div>
    </div>
  );
}

const faceParamDetails = {
  jaw_displacement: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Distance between the nose tip and the chin, normalized by eye distance.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Measured using facial landmarks 4 (nose tip) and 152 (chin) normalized by 33 & 263 (eyes).<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Open and close your mouth, or speak. You will see the Jaw Displacement bar rise and fall.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Stress often causes an involuntary clenched jaw (reduced displacement) or jaw drops under sudden shock.
    </div>
  ),
  jaw_width: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Width of the lower jaw, detecting masseter muscle contraction.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Horizontal distance between lower jaw landmarks 172 and 397 normalized by eye distance.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Clench your teeth tightly together. You will see the Jaw Width bar increase as the masseter muscles contract.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Teeth-clenching is a primary physical, involuntary reaction to stress, anger, and tension.
    </div>
  ),
  blink_velocity: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> The speed at which you close and open your eyes.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Rate of change of the Eye Aspect Ratio (EAR) across consecutive frames.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Blink rapidly or hard. The Blink Velocity metric will spike.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Stress increases autonomic nervous system arousal, elevating blink velocity and frequency.
    </div>
  ),
  brow_tension: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Downward contraction and pulling together of the eyebrows.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Distance from eyebrows to eyes normalized by face height.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Furrow your brows or frown. You'll see the Brow Tension bar rise.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Furrowing the brow is a universal indicator of concentration, anger, or distress.
    </div>
  ),
  lip_compression: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Squeezing the lips tightly together, making them thin.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Vertical mouth gap divided by horizontal mouth width.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Press your lips tightly together into a thin line. Lip Compression bar will rise.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Lip compression is an involuntary subconscious cue for anxiety, holding back speech, or cognitive load.
    </div>
  ),
  head_tilt: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Lateral head tilt angle and movement.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> The angle of the line connecting left and right eyes relative to the horizontal axis.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Tilt your head to the side. The Head Tilt degrees will increase.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Restlessness, frequent head adjustments, or rigid posture are correlated with discomfort and stress.
    </div>
  )
};

const voiceParamDetails = {
  f0_mean: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Fundamental frequency (mean pitch) of your voice.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Autocorrelation of voiced audio frames restricted to calibrated pitch bounds.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Speak in a high-pitched voice, then a low-pitched voice. The Pitch Hz value will change.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Tension in laryngeal muscles from stress tightens the vocal cords, raising fundamental pitch.
    </div>
  ),
  jitter_percent: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Micro-instability and cycle-to-cycle frequency variations of vocal vibrations.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Relative Average Perturbation (RAP) of pitch periods between consecutive frames.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Speak with a shaky, trembling voice, or whisper. You will see Jitter percent rise.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Physiological stress reduces laryngeal muscle stability, leading to higher jitter.
    </div>
  ),
  shimmer_db: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Cycle-to-cycle variation in the amplitude (loudness) of the vocal fold vibration.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Average decibel amplitude deviation between adjacent voiced frames.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Speak with an unstable, trembling loudness. Shimmer will increase.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Stress causes irregular vocal fold closure, which makes loudness fluctuate microscopically.
    </div>
  ),
  speaking_rate_proxy: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Speaking rate and breathiness proxy using zero-crossing rate.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Frequency of sign changes in the audio waveform over time.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Speak extremely fast, or blow air/sigh into the microphone. Speaking Rate bar will rise.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Agitation, panic, or anxiety increases speech rate and shallow breathiness, elevating ZCR.
    </div>
  ),
  voice_intensity: (
    <div>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• What it is:</span> Vocal loudness and energy.<br/>
      <span style={{ color: 'var(--primary-color)', fontWeight: 'bold' }}>• Calculation:</span> Root-Mean-Square (RMS) energy of the audio chunk relative to calibrated noise threshold.<br/>
      <span style={{ color: '#FF9800', fontWeight: 'bold' }}>• How to Test:</span> Speak loudly or shout, then whisper. Voice Intensity bar will scale.<br/>
      <span style={{ color: '#F44336', fontWeight: 'bold' }}>• Stress Impact:</span> Stress triggers fight responses (elevated loudness) or freeze/anxiety responses (muted volume).
    </div>
  )
};

export default function RealtimeMonitor() {
  const [active, setActive] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [faceScore, setFaceScore] = useState(null);
  const [voiceScore, setVoiceScore] = useState(null);
  const [faceIndicators, setFaceIndicators] = useState(null);
  const [voiceIndicators, setVoiceIndicators] = useState(null);
  const esRef = useRef(null);

  // Parameter explorer states
  const [selectedFaceParam, setSelectedFaceParam] = useState('');
  const [selectedVoiceParam, setSelectedVoiceParam] = useState('');

  // Server Connection Status
  const [serverStatus, setServerStatus] = useState('disconnected'); // 'connected', 'connecting', 'disconnected'

  // Calibration states
  const [calibrationPhase, setCalibrationPhase] = useState('idle');
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const silenceRmsRef = useRef([]);

  // Smooth UI display values
  const [smoothFusedScore, setSmoothFusedScore] = useState(0);
  const [smoothFaceScore, setSmoothFaceScore] = useState(null);
  const [smoothVoiceScore, setSmoothVoiceScore] = useState(null);

  // Automatic Background Health Ping
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (response.ok) {
          setServerStatus('connected');
        } else {
          setServerStatus('disconnected');
        }
      } catch (err) {
        setServerStatus('disconnected');
      }
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Easing/smoothing loops for high-responsiveness display updates
  useEffect(() => {
    if (!active) {
      setSmoothFusedScore(0);
      return;
    }
    const target = result && result.fused_score !== undefined ? result.fused_score * 100 : 0;
    const interval = setInterval(() => {
      setSmoothFusedScore(prev => {
        const diff = target - prev;
        if (Math.abs(diff) < 0.5) return target;
        return prev + diff * 0.15; // Smooth exponential step
      });
    }, 50);
    return () => clearInterval(interval);
  }, [result, active]);

  useEffect(() => {
    if (!active || faceScore === null) {
      setSmoothFaceScore(null);
      return;
    }
    const target = faceScore;
    const interval = setInterval(() => {
      setSmoothFaceScore(prev => {
        if (prev === null) return target;
        const diff = target - prev;
        if (Math.abs(diff) < 0.01) return target;
        return prev + diff * 0.15;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [faceScore, active]);

  useEffect(() => {
    if (!active || voiceScore === null) {
      setSmoothVoiceScore(null);
      return;
    }
    const target = voiceScore;
    const interval = setInterval(() => {
      setSmoothVoiceScore(prev => {
        if (prev === null) return target;
        const diff = target - prev;
        if (Math.abs(diff) < 0.01) return target;
        return prev + diff * 0.15;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [voiceScore, active]);

  const connectSSE = () => {
    setServerStatus('connecting');
    const es = new EventSource(`${API_BASE}/api/stream/fused`);
    
    es.onopen = () => {
      setServerStatus('connected');
    };
    
    es.onmessage = (e) => {
      setServerStatus('connected');
      try {
        const data = JSON.parse(e.data);
        if (data.status === 'active') {
          setResult(data);
          setHistory(h => [
            ...h.slice(-29), 
            {
              t: new Date().toLocaleTimeString(),
              score: Math.round(data.fused_score * 100),
            }
          ]);
          // Sync modality scores from fused stream to match decay timing (Fix voice score card display)
          if (data.per_modality) {
            if (data.per_modality.face !== undefined && data.per_modality.face !== null) {
              setFaceScore(data.per_modality.face.score);
            } else {
              setFaceScore(null);
            }
            if (data.per_modality.voice !== undefined && data.per_modality.voice !== null) {
              setVoiceScore(data.per_modality.voice.score);
            } else {
              setVoiceScore(null);
            }
          }
        } else if (data.status === 'waiting') {
          setResult(data);
          setFaceScore(null);
          setVoiceScore(null);
        }
      } catch (err) {
        console.error("SSE parse error: ", err);
      }
    };
    es.onerror = () => {
      console.error("SSE Connection failed.");
      setServerStatus('disconnected');
      es.close();
    };
    esRef.current = es;
  };

  const startMonitoring = () => {
    setActive(true);
    setResult(null);
    setHistory([]);
    setFaceScore(null);
    setVoiceScore(null);
    setFaceIndicators(null);
    setVoiceIndicators(null);
    setSmoothFusedScore(0);
    setSmoothFaceScore(null);
    setSmoothVoiceScore(null);

    if (!isCalibrated) {
      setCalibrating(true);
      return; // Guided CalibrationWizard will trigger connectSSE on Complete
    }

    connectSSE();
  };

  const handleCalibrationComplete = (calibration) => {
    setIsCalibrated(true);
    setCalibrating(false);
    setCalibrationPhase('idle');
    console.log("[Calibration] Guided baseline loaded successfully: ", calibration);
    
    // Start active monitoring session
    connectSSE();
  };

  const resetCalibration = () => {
    setIsCalibrated(false);
    stopMonitoring();
  };

  const stopMonitoring = () => {
    setActive(false);
    setCalibrating(false);
    setCalibrationPhase('idle');
    setFaceScore(null);
    setVoiceScore(null);
    setFaceIndicators(null);
    setVoiceIndicators(null);
    setSmoothFusedScore(0);
    setSmoothFaceScore(null);
    setSmoothVoiceScore(null);
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  };

  const handleVoiceChunk = async (blob) => {
    try {
      const response = await fetch(`${API_BASE}/api/stream/voice?user_id=default`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: blob,
      });
      const data = await response.json();
      
      if (data && data.indicators !== undefined) {
        setVoiceIndicators(data.indicators);
        
        if (calibrationPhase === 'silence') {
          if (data.indicators.voice_intensity !== undefined) {
            silenceRmsRef.current.push(data.indicators.voice_intensity);
            console.log("[Calibration] Added silence RMS sample:", data.indicators.voice_intensity);
          }
        } else if (calibrationPhase === 'voice') {
          await fetch(`${API_BASE}/api/calibrate/voice_sample`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default', indicators: data.indicators, features: data.features }),
          });
          console.log("[Calibration] Posted voice sample indicators and features.");
        }
      }
      
      // Update vocal stress level only in normal monitoring mode, ignoring null (silence) to avoid immediate clear
      if (calibrationPhase === 'idle' && !calibrating && data && data.score !== undefined && data.score !== null) {
        setVoiceScore(data.score);
      }
    } catch (err) {
      console.error("Failed to POST voice chunk: ", err);
    }
  };

  useEffect(() => {
    return () => {
      if (esRef.current) esRef.current.close();
    };
  }, []);

  const displayLevel = smoothFusedScore > 70 ? 'High' : smoothFusedScore > 40 ? 'Moderate' : 'Low';
  const levelColor = active && !calibrating ? LEVEL_COLOR[displayLevel] : 'var(--text-muted)';
  const stressPercent = Math.round(smoothFusedScore);

  return (
    <div className="neon-card fade-in-up" style={{ padding: 24, marginTop: 16 }}>
      {/* Pulse animation styles for server connection */}
      <style>{`
        @keyframes pulseGlow {
          0% { transform: scale(0.92); opacity: 0.5; }
          50% { transform: scale(1.15); opacity: 1; }
          100% { transform: scale(0.92); opacity: 0.5; }
        }
      `}</style>
      
      <h3 className="text-center mb-4 neon-text" style={{ fontSize: '1.8rem' }}>🧠 Real-Time Multimodal Monitoring</h3>
      
      {/* Controls & Overall Fused Assessment */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, marginBottom: 24, borderBottom: 'var(--glass-border)', paddingBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {!active ? (
            <button 
              className="btn btn-primary" 
              onClick={startMonitoring} 
              disabled={serverStatus === 'disconnected'}
              style={{ fontSize: '1.05rem', padding: '12px 28px' }}
            >
              ▶ Start Session
            </button>
          ) : (
            <div style={{ display: 'flex', gap: 12 }}>
              <button className="btn" onClick={stopMonitoring} style={{ background: '#D32F2F', color: '#fff', fontSize: '1.05rem', padding: '12px 28px', boxShadow: '0 4px 15px rgba(211, 47, 47, 0.3)' }}>
                ⏹ Stop Session
              </button>
              {isCalibrated && (
                <button className="btn" onClick={resetCalibration} style={{ background: 'transparent', border: '1px solid var(--primary-color)', color: 'var(--primary-color)', fontSize: '1.05rem', padding: '12px 24px' }}>
                  🔄 Recalibrate
                </button>
              )}
            </div>
          )}
        </div>

        {active && !calibrating && result && result.status === 'active' && (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 16,
            background: 'var(--card-bg)',
            padding: '10px 24px',
            borderRadius: 12,
            border: `1.5px solid ${levelColor}`,
            boxShadow: `0 0 15px ${levelColor}22`
          }}>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Fused Multimodal Stress Level:</span>
            <span style={{ 
              fontWeight: 800, 
              color: levelColor, 
              fontSize: '1.4rem', 
              textShadow: `0 0 10px ${levelColor}44` 
            }}>
              {displayLevel} ({stressPercent}%)
            </span>
          </div>
        )}
      </div>

      {/* Guided Calibration Wizard Container */}
      {active && calibrating && (
        <div style={{ marginBottom: 28 }} className="fade-in-up">
          <CalibrationWizard 
            userId="default"
            silenceRmsRef={silenceRmsRef}
            onPhaseChange={(phase) => setCalibrationPhase(phase)}
            onComplete={handleCalibrationComplete}
          />
        </div>
      )}

      {/* Main Grid: Face (Modality 1) & Voice (Modality 2) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 24, marginBottom: 24 }}>
        
        {/* Face Card */}
        <div style={{ 
          background: 'var(--card-bg)',
          borderRadius: 16,
          padding: 20,
          border: 'var(--glass-border)',
          boxShadow: 'var(--glass-shadow)',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: 'var(--glass-border)', paddingBottom: 10 }}>
            <h4 style={{ margin: 0, fontSize: '1.1rem', color: 'var(--text-color)' }}>📸 Facial Modality</h4>
            {active && !calibrating && (
              <span style={{ 
                fontWeight: 700, 
                color: smoothFaceScore === null ? 'var(--text-muted)' : smoothFaceScore > 0.7 ? '#F44336' : smoothFaceScore > 0.4 ? '#FF9800' : 'var(--primary-color)',
                fontSize: '0.9rem',
                padding: '4px 10px',
                borderRadius: 6,
                background: 'var(--accent-light-bg)',
                border: smoothFaceScore === null ? '1px solid rgba(120,120,120,0.1)' : smoothFaceScore > 0.7 ? '1px solid #F4433655' : smoothFaceScore > 0.4 ? '1px solid #FF980055' : 'var(--glass-border)'
              }}>
                {smoothFaceScore === null ? 'No Face Detected' : `Stress: ${Math.round(smoothFaceScore * 100)}%`}
              </span>
            )}
            {active && calibrating && (
              <span style={{ fontWeight: 600, color: '#FF9800', fontSize: '0.85rem' }}>
                {calibrationPhase === 'face' ? 'RECORDING BASELINE...' : 'CAMERA PREVIEW'}
              </span>
            )}
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <FaceStream 
              active={active} 
              calibrationMode={calibrationPhase === 'face'}
              userId="default"
              onResult={(data) => {
                if (!calibrating && data && data.score !== undefined) {
                  setFaceScore(data.score);
                }
              }} 
              onIndicatorsUpdate={(indicators) => {
                setFaceIndicators(indicators);
                if (indicators === null) {
                  setFaceScore(null);
                }
              }}
            />
          </div>

          <div style={{ background: 'var(--accent-light-bg)', borderRadius: 10, padding: 14, border: 'var(--glass-border)', flexGrow: 1 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 12 }}>Live Tracked Facial Patterns</div>
            {active && faceIndicators ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <IndicatorBar label="Jaw Displacement (Mouth Open)" value={faceIndicators.jaw_displacement} scale={2.0} format={(v) => `${(v).toFixed(2)}`} />
                <IndicatorBar label="Jaw Width (Masseter Clench)" value={faceIndicators.jaw_angle_width} scale={2.5} format={(v) => `${Math.round(v * 100)}%`} />
                <IndicatorBar label="Blink Velocity (Stress Blink)" value={faceIndicators.blink_velocity} scale={5.0} format={(v) => `${(v).toFixed(2)} /s`} />
                <IndicatorBar label="Brow Tension (Contraction)" value={faceIndicators.forehead_tension} scale={0.5} format={(v) => `${Math.round(v * 100)}%`} />
                <IndicatorBar label="Lip Compression" value={faceIndicators.lip_compression} scale={0.5} format={(v) => `${Math.round(v * 100)}%`} />
                <IndicatorBar label="Head Tilt / Movement" value={faceIndicators.head_tilt} scale={30.0} format={(v) => `${(v).toFixed(1)}°`} />
              </div>
            ) : (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px 0' }}>
                {active ? (faceIndicators === null ? "No face detected in video frame" : "Extracting landmarks...") : "Start session to enable camera tracking"}
              </div>
            )}
          </div>
          
          {/* Biomarker Guide Dropdown for Face */}
          <div style={{ marginTop: 8, borderTop: 'var(--glass-border)', paddingTop: 12 }}>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: 6, fontWeight: 600 }}>💡 Face Biomarker Guide & Tester</label>
            <select 
              value={selectedFaceParam} 
              onChange={(e) => setSelectedFaceParam(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'var(--card-bg)',
                color: 'var(--text-color)',
                border: 'var(--glass-border)',
                borderRadius: 6,
                fontSize: '0.8rem',
                outline: 'none',
                cursor: 'pointer',
                fontFamily: 'inherit'
              }}
            >
              <option value="">-- Choose facial biomarker to test/learn --</option>
              <option value="jaw_displacement">Jaw Displacement (Mouth Open)</option>
              <option value="jaw_width">Jaw Width (Masseter Clench)</option>
              <option value="blink_velocity">Blink Velocity (Stress Blink)</option>
              <option value="brow_tension">Brow Tension (Contraction)</option>
              <option value="lip_compression">Lip Compression</option>
              <option value="head_tilt">Head Tilt / Movement</option>
            </select>
            
            {selectedFaceParam && (
              <div className="fade-in-up" style={{
                marginTop: 10,
                background: 'var(--accent-light-bg)',
                border: 'var(--glass-border)',
                borderRadius: 8,
                padding: 12,
                fontSize: '0.75rem',
                lineHeight: '1.45',
                color: 'var(--text-color)',
                boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.2)'
              }}>
                {faceParamDetails[selectedFaceParam]}
              </div>
            )}
          </div>
        </div>

        {/* Voice Card */}
        <div style={{ 
          background: 'var(--card-bg)',
          borderRadius: 16,
          padding: 20,
          border: 'var(--glass-border)',
          boxShadow: 'var(--glass-shadow)',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: 'var(--glass-border)', paddingBottom: 10 }}>
            <h4 style={{ margin: 0, fontSize: '1.1rem', color: 'var(--text-color)' }}>🎤 Vocal Modality</h4>
            {active && !calibrating && (
              <span style={{ 
                fontWeight: 700, 
                color: smoothVoiceScore === null ? 'var(--text-muted)' : smoothVoiceScore > 0.7 ? '#F44336' : smoothVoiceScore > 0.4 ? '#FF9800' : 'var(--primary-color)',
                fontSize: '0.9rem',
                padding: '4px 10px',
                borderRadius: 6,
                background: 'var(--accent-light-bg)',
                border: smoothVoiceScore === null ? '1px solid rgba(120,120,120,0.1)' : smoothVoiceScore > 0.7 ? '1px solid #F4433655' : smoothVoiceScore > 0.4 ? '1px solid #FF980055' : 'var(--glass-border)'
              }}>
                {smoothVoiceScore === null ? 'Silent / Ambient' : `Stress: ${Math.round(smoothVoiceScore * 100)}%`}
              </span>
            )}
            {active && calibrating && (
              <span style={{ fontWeight: 600, color: '#FF9800', fontSize: '0.85rem' }}>
                {calibrationPhase === 'silence' ? 'RECORDING SILENCE...' : calibrationPhase === 'voice' ? 'RECORDING VOICE...' : 'MICROPHONE PREVIEW'}
              </span>
            )}
          </div>
          
          <WaveformRecorder
            continuous={active}
            chunkIntervalMs={1000}
            onChunk={handleVoiceChunk}
            voiceScore={smoothVoiceScore}
          />

          <div style={{ background: 'var(--accent-light-bg)', borderRadius: 10, padding: 14, border: 'var(--glass-border)', flexGrow: 1 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 12 }}>Acoustic Stress Biomarkers</div>
            {active && voiceIndicators ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <IndicatorBar label="Pitch (F0 Mean)" value={voiceIndicators.f0_mean} scale={300.0} format={(v) => `${Math.round(v)} Hz`} />
                <IndicatorBar 
                  label="Jitter (Micro-instability)" 
                  value={voiceIndicators.jitter_percent} 
                  scale={5.0} 
                  format={(v) => `${(v).toFixed(2)}%`}
                  valueColor={voiceIndicators.jitter_reliable !== false ? null : '#FF9800'}
                  extra={voiceIndicators.jitter_reliable === false && (
                    <span style={{
                      fontSize: '0.65rem', color: '#FF9800',
                      background: '#FF980022', borderRadius: 4,
                      padding: '1px 6px', fontWeight: 'bold', border: '1px solid #FF980044'
                    }}>
                      ⚠ noisy mic
                    </span>
                  )}
                />
                <IndicatorBar label="Shimmer (Amplitude Var)" value={voiceIndicators.shimmer_db} scale={3.0} format={(v) => `${(v).toFixed(2)} dB`} />
                <IndicatorBar label="Speaking Rate (ZCR)" value={voiceIndicators.speaking_rate_proxy} scale={0.2} format={(v) => `${(v * 100).toFixed(0)}%`} />
                <IndicatorBar label="Voice Intensity" value={voiceIndicators.voice_intensity} scale={0.3} format={(v) => `${Math.round(v * 100)}%`} />
              </div>
            ) : (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '20px 0' }}>
                {active ? "Waiting for voice buffer..." : "Start session to enable microphone"}
              </div>
            )}
          </div>

          {/* Biomarker Guide Dropdown for Voice */}
          <div style={{ marginTop: 8, borderTop: 'var(--glass-border)', paddingTop: 12 }}>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: 6, fontWeight: 600 }}>💡 Voice Biomarker Guide & Tester</label>
            <select 
              value={selectedVoiceParam} 
              onChange={(e) => setSelectedVoiceParam(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'var(--card-bg)',
                color: 'var(--text-color)',
                border: 'var(--glass-border)',
                borderRadius: 6,
                fontSize: '0.8rem',
                outline: 'none',
                cursor: 'pointer',
                fontFamily: 'inherit'
              }}
            >
              <option value="">-- Choose vocal biomarker to test/learn --</option>
              <option value="f0_mean">Pitch (F0 Mean)</option>
              <option value="jitter_percent">Jitter (Micro-instability)</option>
              <option value="shimmer_db">Shimmer (Amplitude Var)</option>
              <option value="speaking_rate_proxy">Speaking Rate (ZCR)</option>
              <option value="voice_intensity">Voice Intensity</option>
            </select>
            
            {selectedVoiceParam && (
              <div className="fade-in-up" style={{
                marginTop: 10,
                background: 'var(--accent-light-bg)',
                border: 'var(--glass-border)',
                borderRadius: 8,
                padding: 12,
                fontSize: '0.75rem',
                lineHeight: '1.45',
                color: 'var(--text-color)',
                boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.2)'
              }}>
                {voiceParamDetails[selectedVoiceParam]}
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Dynamic Reliability and Weights panel */}
      {active && !calibrating && result && result.status === 'active' && result.weights && (
        <div style={{ background: 'var(--accent-light-bg)', padding: 16, borderRadius: 12, border: 'var(--glass-border)', marginBottom: 24 }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>Dynamic Sensor Reliability Weights</div>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            {Object.entries(result.weights).map(([mode, weight]) => (
              <div key={mode} style={{ flex: '1 1 150px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', padding: '6px 12px', background: 'var(--card-bg)', borderRadius: 8, border: 'var(--glass-border)' }}>
                <span style={{ textTransform: 'capitalize', color: 'var(--text-color)' }}>{mode} sensor weight:</span>
                <span style={{ fontWeight: 700, color: 'var(--primary-color)' }}>{Math.round(weight * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Professional Clinical-Grade Real-Time Stress Trend Chart */}
      {active && !calibrating && history.length > 1 && (() => {
        const chartPoints = history.map((h, i) => {
          const x = 45 + (i / Math.max(1, history.length - 1)) * 440; // width is 500, scalable region
          const y = 130 - (h.score / 100) * 115; // bottom at y=130, height scaled
          return { x, y };
        });

        const lastPt = chartPoints[chartPoints.length - 1];
        const lastScore = history[history.length - 1]?.score || 0;
        const trendColor = lastScore > 70 ? LEVEL_COLOR.High : lastScore > 40 ? LEVEL_COLOR.Moderate : LEVEL_COLOR.Low;

        const strokePath = chartPoints.length > 0 
          ? `M ${chartPoints.map(p => `${p.x},${p.y}`).join(' L ')}` 
          : '';

        const fillPath = chartPoints.length > 0 
          ? `M 45,130 L ${chartPoints.map(p => `${p.x},${p.y}`).join(' L ')} L ${chartPoints[chartPoints.length - 1].x},130 Z` 
          : '';

        return (
          <div style={{ 
            background: 'var(--card-bg)', 
            padding: 20, 
            borderRadius: 16, 
            border: 'var(--glass-border)',
            boxShadow: 'var(--glass-shadow)',
            marginBottom: 24
          }}>
            <div style={{ 
              fontSize: '0.85rem', 
              color: 'var(--text-color)', 
              marginBottom: 16, 
              fontWeight: 600,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span>📈 Real-Time Multi-Modal Stress Trend</span>
              <span style={{ 
                fontFamily: 'monospace', 
                fontSize: '0.75rem', 
                color: 'var(--text-muted)',
                background: 'var(--accent-light-bg)',
                padding: '2px 8px',
                borderRadius: 4
              }}>
                Resolution: 1.0s/step
              </span>
            </div>
            
            <svg viewBox="0 0 500 150" width="100%" height="150" style={{ display: 'block', overflow: 'visible' }}>
              <defs>
                <linearGradient id="trendAreaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={trendColor} stopOpacity="0.25" />
                  <stop offset="100%" stopColor={trendColor} stopOpacity="0.00" />
                </linearGradient>
              </defs>
              
              {/* Horizontal Grid lines */}
              <line x1="45" y1="15" x2="485" y2="15" stroke="var(--glass-border)" strokeWidth="1" strokeOpacity="0.3" />
              <line x1="45" y1="49.5" x2="485" y2="49.5" stroke="rgba(244,67,54,0.12)" strokeWidth="1" strokeDasharray="4,4" />
              <line x1="45" y1="84" x2="485" y2="84" stroke="rgba(255,152,0,0.12)" strokeWidth="1" strokeDasharray="4,4" />
              <line x1="45" y1="130" x2="485" y2="130" stroke="var(--glass-border)" strokeWidth="1" />

              {/* Y Axis Labels */}
              <text x="36" y="19" fill="#F44336" fontSize="9px" fontFamily="monospace" textAnchor="end" fontWeight="bold">100%</text>
              <text x="36" y="53" fill="rgba(244,67,54,0.7)" fontSize="9px" fontFamily="monospace" textAnchor="end">70%</text>
              <text x="36" y="87.5" fill="rgba(255,152,0,0.7)" fontSize="9px" fontFamily="monospace" textAnchor="end">40%</text>
              <text x="36" y="133" fill="var(--primary-color)" fontSize="9px" fontFamily="monospace" textAnchor="end" fontWeight="bold">0%</text>

              {/* Threshold region labels */}
              <text x="480" y="26" fill="rgba(244,67,54,0.25)" fontSize="8px" fontFamily="monospace" textAnchor="end" fontWeight="bold">HIGH STRESS</text>
              <text x="480" y="60" fill="rgba(255,152,0,0.25)" fontSize="8px" fontFamily="monospace" textAnchor="end" fontWeight="bold">MODERATE</text>
              <text x="480" y="110" fill="var(--primary-color)" fillOpacity="0.25" fontSize="8px" fontFamily="monospace" textAnchor="end" fontWeight="bold">CALM</text>

              {/* Filled Area Chart */}
              {fillPath && <path d={fillPath} fill="url(#trendAreaGradient)" />}

              {/* Stroke line (glowing) */}
              {strokePath && (
                <path 
                  d={strokePath} 
                  fill="none" 
                  stroke={trendColor} 
                  strokeWidth="2.5" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                  style={{ filter: `drop-shadow(0px 0px 4px ${trendColor}88)` }}
                />
              )}

              {/* Pulse glow dot for latest score */}
              {lastPt && (
                <>
                  <circle cx={lastPt.x} cy={lastPt.y} r={4.5} fill={trendColor} />
                  <circle cx={lastPt.x} cy={lastPt.y} r={9} fill="none" stroke={trendColor} strokeWidth="1.5">
                    <animate attributeName="r" values="4.5;13;4.5" dur="1.8s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.8;0;0.8" dur="1.8s" repeatCount="indefinite" />
                  </circle>
                </>
              )}
            </svg>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 12 }}>
              <span>Session Start</span>
              <span style={{ fontWeight: 'bold', color: trendColor }}>Current: {lastScore}%</span>
              <span>Now</span>
            </div>
          </div>
        );
      })()}

      {active && !calibrating && (!result || result.status === 'waiting') && (
        <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)' }} className="fade-in-up">
          <div className="spinner-border text-info mb-2" role="status" style={{ width: '1.5rem', height: '1.5rem' }}></div>
          <p style={{ fontStyle: 'italic', margin: 0, fontSize: '0.85rem' }}>
            Waiting for face and voice feed telemetry...
          </p>
        </div>
      )}

      {!active && (
        <p className="text-center" style={{ color: 'var(--text-muted)', margin: 0, fontStyle: 'italic', fontSize: '0.9rem' }}>
          Session inactive. Click "Start Session" to begin real-time calibration and diagnostics.
        </p>
      )}
    </div>
  );
}
