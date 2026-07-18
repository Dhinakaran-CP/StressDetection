import React, { useEffect, useState, useRef } from 'react';
import { API_BASE } from '../config';
import FaceStream from './FaceStream';
import WaveformRecorder from './WaveformRecorder';
import CalibrationWizard from './CalibrationWizard';

export default function RealtimeMonitor() {
  const [active, setActive] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [faceScore, setFaceScore] = useState(null);
  const [voiceScore, setVoiceScore] = useState(null);
  const [faceIndicators, setFaceIndicators] = useState(null);
  const [voiceIndicators, setVoiceIndicators] = useState(null);
  const esRef = useRef(null);
  const voicePostPendingRef = useRef(false);
  const [modelMetadata, setModelMetadata] = useState(null);
  const [fallbackStatus, setFallbackStatus] = useState(null);

  // Parameter guide states
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

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const [metaRes, fallbackRes] = await Promise.all([
          fetch(`${API_BASE}/api/model/version`),
          fetch(`${API_BASE}/api/fallback/status`)
        ]);
        if (metaRes.ok) {
          const metaData = await metaRes.json();
          setModelMetadata(metaData);
        }
        if (fallbackRes.ok) {
          const fallbackData = await fallbackRes.json();
          setFallbackStatus(fallbackData);
        }
      } catch (err) {
        console.error("Failed to load model metadata:", err);
      }
    };
    if (active) {
      fetchMetadata();
    }
  }, [active]);

  // Smooth fused score easing
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
        return prev + diff * 0.15;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [result, active]);

  // Smooth face score easing
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

  // Smooth voice score easing
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
      return;
    }

    connectSSE();
  };

  const handleCalibrationComplete = (calibration) => {
    setIsCalibrated(true);
    setCalibrating(false);
    setCalibrationPhase('idle');
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
    if (voicePostPendingRef.current) return;
    voicePostPendingRef.current = true;
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
          }
        } else if (calibrationPhase === 'voice') {
          await fetch(`${API_BASE}/api/calibrate/voice_sample`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default', indicators: data.indicators, features: data.features }),
          });
        }
      }
      
      if (calibrationPhase === 'idle' && !calibrating && data && data.score !== undefined && data.score !== null) {
        setVoiceScore(data.score);
      }
    } catch (err) {
      console.error("Failed to POST voice chunk: ", err);
    } finally {
      voicePostPendingRef.current = false;
    }
  };

  useEffect(() => {
    return () => {
      if (esRef.current) esRef.current.close();
    };
  }, []);

  const displayLevel = smoothFusedScore > 70 ? 'High' : smoothFusedScore > 40 ? 'Moderate' : 'Low';
  const scoreColor = smoothFusedScore > 70 ? '#ba1a1a' : smoothFusedScore > 40 ? '#dce9ff' : '#0e3b69';

  return (
    <div className="space-y-8 select-none">
      {/* Session controller / calibration overlay */}
      {active && calibrating && (
        <div className="bg-surface rounded-3xl border border-outline-variant/30 p-8 shadow-sm">
          <CalibrationWizard 
            userId="default"
            silenceRmsRef={silenceRmsRef}
            onPhaseChange={(phase) => setCalibrationPhase(phase)}
            onComplete={handleCalibrationComplete}
          />
        </div>
      )}

      {/* Main Grid: Webcam + Waveform vs. Vitals Cards */}
      <div className="grid grid-cols-12 gap-8 items-stretch">
        
        {/* Left Column: Live camera feed bento card */}
        <section className="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-[32px] overflow-hidden shadow-sm flex flex-col justify-between relative border border-outline-variant/10 min-h-[500px]">
          
          {/* Webcam / Preview stream */}
          <div className="relative flex-1 bg-slate-950 flex items-center justify-center overflow-hidden aspect-video">
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

            {/* Custom face mesh absolute points overlay mockup if no stream */}
            {!active && (
              <div className="absolute inset-0 pointer-events-none opacity-40">
                <div className="w-1.5 h-1.5 bg-[#a5c8ff] rounded-full absolute shadow-glow" style={{ top: '40%', left: '45%' }}></div>
                <div className="w-1.5 h-1.5 bg-[#a5c8ff] rounded-full absolute shadow-glow" style={{ top: '40%', left: '55%' }}></div>
                <div className="w-1.5 h-1.5 bg-[#a5c8ff] rounded-full absolute shadow-glow" style={{ top: '50%', left: '50%' }}></div>
                <div className="w-1.5 h-1.5 bg-[#a5c8ff] rounded-full absolute shadow-glow" style={{ top: '58%', left: '42%' }}></div>
                <div className="w-1.5 h-1.5 bg-[#a5c8ff] rounded-full absolute shadow-glow" style={{ top: '58%', left: '58%' }}></div>
              </div>
            )}

            {/* Live Indicator Badge */}
            <div className="absolute top-6 left-6 flex items-center gap-3 bg-black/35 backdrop-blur-md px-4 py-2 rounded-full border border-white/20">
              <div className={`w-2 h-2 rounded-full ${active ? 'bg-error animate-pulse' : 'bg-secondary'}`}></div>
              <span className="text-white font-label-caps text-[10px] tracking-widest uppercase">
                {active ? 'Live Signal' : 'Stream Closed'}
              </span>
            </div>
          </div>

          {/* Voice recorder audio waveform strip */}
          <div className="bg-primary flex flex-col px-8 py-4 justify-center border-t border-outline-variant/10">
            <WaveformRecorder
              continuous={active}
              chunkIntervalMs={1000}
              onChunk={handleVoiceChunk}
              voiceScore={smoothVoiceScore}
              onIndicatorsUpdate={(indicators) => {
                if (active && !calibrating) {
                  setVoiceIndicators(indicators);
                }
              }}
            />
          </div>
        </section>

        {/* Right Column: Live vitals sidebar cards */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
          
          {/* Card 1: Stress Level score */}
          <div className="bg-surface-container-lowest p-6 rounded-[24px] shadow-sm border border-outline-variant/10 flex flex-col justify-between min-h-[110px]">
            <div className="flex justify-between items-start">
              <div className="p-2 bg-surface-container rounded-lg flex items-center justify-center">
                <span className="material-symbols-outlined text-primary text-lg">psychology</span>
              </div>
              <span className="font-label-caps text-[11px] text-on-surface-variant font-semibold tracking-wider">SYM PATHETIC LOAD</span>
            </div>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="font-data-metric text-[36px] text-on-surface font-bold">
                {active && !calibrating ? Math.round(smoothFusedScore) : '--'}
              </span>
              <span className="text-on-surface-variant font-label-caps text-xs">%</span>
            </div>
            <div className="h-1 bg-surface-container-high rounded-full overflow-hidden mt-3">
              <div className="h-full bg-primary" style={{ width: active && !calibrating ? `${smoothFusedScore}%` : '0%' }}></div>
            </div>
          </div>

          {/* Confidence Card */}
          <div className="bg-surface-container-lowest p-6 rounded-[24px] shadow-sm border border-outline-variant/10 flex flex-col justify-between min-h-[110px]">
            <div className="flex justify-between items-start">
              <div className="p-2 bg-surface-container rounded-lg flex items-center justify-center">
                <span className="material-symbols-outlined text-primary text-lg">verified_user</span>
              </div>
              <span className="font-label-caps text-[11px] text-on-surface-variant font-semibold tracking-wider">PREDICTION CONFIDENCE</span>
            </div>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="font-data-metric text-[36px] text-on-surface font-bold text-primary">
                {active && !calibrating ? Math.round(Math.max(smoothFusedScore, 100 - smoothFusedScore)) : '--'}
              </span>
              <span className="text-on-surface-variant font-label-caps text-xs">%</span>
            </div>
            <div className="text-[10px] text-on-surface-variant font-semibold italic mt-2">
              {active && !calibrating && Math.abs(smoothFusedScore - 50) < 10 ? "⚠️ Score is close to boundary. High uncertainty." : "✓ High model certainty."}
            </div>
          </div>

          {/* Active Model & Resilience Badge */}
          <div className="bg-surface-container-lowest p-6 rounded-[24px] shadow-sm border border-outline-variant/10 flex flex-col gap-3">
            <span className="font-label-caps text-[11px] text-on-surface-variant font-semibold tracking-wider flex items-center gap-1.5">
              <span className="material-symbols-outlined text-[16px] text-primary">settings_suggest</span>
              MODEL ORCHESTRATION & RESILIENCE
            </span>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center bg-surface-container-low p-2.5 rounded-lg border border-outline-variant/10">
                <span className="font-medium text-on-surface-variant">Selected Engine</span>
                <span className="font-bold text-primary">{fallbackStatus?.active_model || "SSVB-CASA-AIS"}</span>
              </div>
              <div className="flex justify-between items-center bg-surface-container-low p-2.5 rounded-lg border border-outline-variant/10">
                <span className="font-medium text-on-surface-variant">Resilience Mode</span>
                {fallbackStatus?.fallback_active ? (
                  <span className="px-2.5 py-0.5 rounded-full bg-error-container text-on-error-container text-[10px] font-bold font-label-caps tracking-wide">FALLBACK ACTIVE</span>
                ) : (
                  <span className="px-2.5 py-0.5 rounded-full bg-primary-container/10 text-primary text-[10px] font-bold font-label-caps tracking-wide">DYNAMIC ROUTING</span>
                )}
              </div>
            </div>
          </div>

          {/* Modality Contribution Chart (Dynamic weights) */}
          {active && !calibrating && result && (
            <div className="bg-surface-container-lowest p-6 rounded-[24px] shadow-sm border border-outline-variant/10 space-y-4">
              <span className="font-label-caps text-[11px] text-on-surface-variant font-semibold tracking-wider flex items-center gap-1.5">
                <span className="material-symbols-outlined text-[16px] text-primary">analytics</span>
                DYNAMIC MODALITY CONTRIBUTIONS
              </span>
              <div className="space-y-3.5">
                {Object.entries(result.weights || {}).map(([modality, weight]) => (
                  <div key={modality} className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold capitalize">
                      <span className="text-on-surface-variant">{modality} stream</span>
                      <span className="text-primary">{Math.round(weight * 100)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-surface-container rounded-full overflow-hidden">
                      <div className="h-full bg-primary rounded-full transition-all duration-500" style={{ width: `${weight * 100}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Compact Biometrics diagnostics */}
          <div className="bg-surface-container-lowest p-6 rounded-[24px] shadow-sm border border-outline-variant/10 space-y-4">
            <span className="font-label-caps text-[11px] text-on-surface-variant font-semibold tracking-wider flex items-center gap-1.5">
              <span className="material-symbols-outlined text-[16px] text-primary">biotech</span>
              BIOMETRIC RUNTIME TELEMETRY
            </span>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-surface-container p-3 rounded-xl border border-outline-variant/5">
                <div className="text-[10px] font-bold text-outline font-label-caps tracking-wide">HR (BPM)</div>
                <div className="text-sm font-bold text-on-surface mt-1">{active && !calibrating ? (faceIndicators?.heart_rate || 74) : '--'}</div>
              </div>
              <div className="bg-surface-container p-3 rounded-xl border border-outline-variant/5">
                <div className="text-[10px] font-bold text-outline font-label-caps tracking-wide">BLINK/MIN</div>
                <div className="text-sm font-bold text-on-surface mt-1">{active && faceIndicators ? Math.round(faceIndicators.blink_velocity * 10) : '--'}</div>
              </div>
              <div className="bg-surface-container p-3 rounded-xl border border-outline-variant/5">
                <div className="text-[10px] font-bold text-outline font-label-caps tracking-wide">JITTER</div>
                <div className="text-sm font-bold text-on-surface mt-1">{active && voiceIndicators ? `${voiceIndicators.jitter_percent.toFixed(2)}%` : '--'}</div>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Dynamic Session Actions & Guide Drawer */}
      <div className="flex flex-col md:flex-row gap-6 items-center justify-between bg-surface-container-low p-6 rounded-2xl border border-outline-variant/10">
        <div className="flex items-center gap-4">
          {!active ? (
            <button
              onClick={startMonitoring}
              disabled={serverStatus === 'disconnected'}
              className="bg-primary text-on-primary font-bold px-8 py-3.5 rounded-xl hover:opacity-90 active:scale-95 transition-all shadow-md flex items-center gap-2"
            >
              <span className="material-symbols-outlined text-sm">play_arrow</span>
              Start Session
            </button>
          ) : (
            <div className="flex gap-3">
              <button
                onClick={stopMonitoring}
                className="bg-error text-white font-bold px-8 py-3.5 rounded-xl hover:opacity-90 active:scale-95 transition-all shadow-md flex items-center gap-2"
              >
                <span className="material-symbols-outlined text-sm">stop</span>
                Stop Session
              </button>
              {isCalibrated && (
                <button
                  onClick={resetCalibration}
                  className="border border-primary text-primary hover:bg-surface-container-high font-bold px-6 py-3 rounded-xl transition-all"
                >
                  Recalibrate
                </button>
              )}
            </div>
          )}
          <span className="text-xs text-on-surface-variant font-medium">
            Status: <strong className={serverStatus === 'connected' ? 'text-primary' : 'text-error'}>{serverStatus.toUpperCase()}</strong>
          </span>
        </div>

        {active && !calibrating && result && result.status === 'active' && (
          <div className="bg-white/60 border border-outline-variant/20 px-6 py-3 rounded-xl flex items-center gap-4">
            <span className="text-xs text-on-surface-variant font-medium">Stress Assessment:</span>
            <span className="font-bold text-sm text-primary uppercase font-label-caps tracking-wider">
              {displayLevel} ({Math.round(smoothFusedScore)}%)
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
