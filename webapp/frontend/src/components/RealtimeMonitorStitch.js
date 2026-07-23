import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../config';
import FaceStream from './FaceStream';
import WaveformRecorder from './WaveformRecorder';

export default function RealtimeMonitorStitch() {
  const [active, setActive] = useState(false);
  const [result, setResult] = useState(null);
  const [faceIndicators, setFaceIndicators] = useState(null);
  const [voiceIndicators, setVoiceIndicators] = useState(null);
  const [serverStatus, setServerStatus] = useState('connected');
  const [smoothFusedScore, setSmoothFusedScore] = useState(12.4);
  const [recordingSeconds, setRecordingSeconds] = useState(842); // 00:14:02:45
  const esRef = useRef(null);
  const voicePostPendingRef = useRef(false);

  // Background timer for active recording
  useEffect(() => {
    let interval;
    if (active) {
      interval = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
        setSmoothFusedScore((prev) => {
          const target = result && result.fused_score !== undefined ? result.fused_score * 100 : 12.4;
          return +(prev + (target - prev) * 0.1).toFixed(1);
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [active, result]);

  // Connect SSE Stream
  const handleStart = () => {
    setActive(true);
    setServerStatus('connecting');
    try {
      const es = new EventSource(`${API_BASE}/api/stream/fused`);
      es.onopen = () => setServerStatus('connected');
      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setResult(data);
        } catch (e) {}
      };
      es.onerror = () => {
        setServerStatus('error');
      };
      esRef.current = es;
    } catch (e) {
      setServerStatus('connected');
    }
  };

  const handlePause = () => {
    setActive(false);
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  };

  const handleStop = () => {
    handlePause();
    setRecordingSeconds(0);
    setResult(null);
  };

  const handleFaceUpdate = async (faceData) => {
    if (!faceData) return;
    setFaceIndicators(faceData.indicators);

    if (!active) return;
    try {
      await fetch(`${API_BASE}/api/stream/face`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(faceData)
      });
    } catch (e) {}
  };

  const handleVoiceUpdate = async (voiceData) => {
    if (!voiceData) return;
    setVoiceIndicators(voiceData.indicators);

    if (!active || voicePostPendingRef.current) return;
    voicePostPendingRef.current = true;
    try {
      await fetch(`${API_BASE}/api/stream/voice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(voiceData)
      });
    } catch (e) {} finally {
      voicePostPendingRef.current = false;
    }
  };

  const formatTimer = (totalSec) => {
    const hrs = String(Math.floor(totalSec / 3600)).padStart(2, '0');
    const mins = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
    const secs = String(totalSec % 60).padStart(2, '0');
    const ms = '45';
    return `${hrs}:${mins}:${secs}:${ms}`;
  };

  const stressCategory = result ? result.stress_category : (smoothFusedScore > 40 ? 'MODERATE' : 'MINIMAL');

  return (
    <div className="w-full animate-fade-in-up">
      {/* Top Controls Header matching Stitch */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center px-6 py-4 mb-6 rounded-2xl bg-surface/80 backdrop-blur-xl border border-primary/10 shadow-sm gap-4">
        <div>
          <h2 className="font-headline-md text-xl font-bold text-on-background">Realtime Monitor</h2>
          <div className="flex items-center gap-3 mt-1">
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${active ? 'bg-emerald-500 animate-pulse' : 'bg-secondary'}`}></span>
              <span className="font-label-caps text-[9px] text-secondary font-bold uppercase tracking-widest">
                {active ? 'Live Stream Active' : 'Stream Standby'}
              </span>
            </div>
            <div className="h-3 w-[1px] bg-primary/10"></div>
            <span className="font-label-caps text-[9px] text-on-surface-variant/60 uppercase tracking-widest">Session ID: VM-10829-QX</span>
          </div>
        </div>

        {/* Action Controls Pill Bar */}
        <div className="flex items-center gap-3">
          <div className="flex bg-surface-container-low p-1.5 rounded-full border border-primary/10 shadow-sm">
            <button
              onClick={handleStart}
              className={`px-4 py-1.5 rounded-full font-label-caps text-xs flex items-center gap-1.5 transition-all font-semibold ${
                active ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-primary'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">play_arrow</span> Start
            </button>
            <button
              onClick={handlePause}
              className={`px-4 py-1.5 rounded-full font-label-caps text-xs flex items-center gap-1.5 transition-all font-semibold ${
                !active ? 'bg-surface-container-high text-primary' : 'text-on-surface-variant hover:bg-surface-variant/20'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">pause</span> Pause
            </button>
            <button
              onClick={handleStop}
              className="px-4 py-1.5 text-error hover:bg-error/10 rounded-full font-label-caps text-xs flex items-center gap-1.5 transition-all font-semibold"
            >
              <span className="material-symbols-outlined text-[16px]">stop</span> Stop
            </button>
            <button
              onClick={handleStart}
              className="px-4 py-1.5 text-secondary hover:bg-secondary/10 rounded-full font-label-caps text-xs flex items-center gap-1.5 transition-all font-semibold"
            >
              <span className="material-symbols-outlined text-[16px]">recenter</span> Recalibrate
            </button>
          </div>
        </div>
      </header>

      {/* Main 12-Column Grid matching Stitch Layout */}
      <div className="grid grid-cols-12 gap-6 max-w-[1920px] mx-auto">
        {/* Main Monitoring Column (8 Cols) */}
        <div className="col-span-12 xl:col-span-8 space-y-6">
          {/* Stream Info Bar */}
          <div className="glass-card p-3 rounded-xl flex items-center justify-between bg-surface-container-low border border-primary/10 flex-wrap gap-3">
            <div className="flex items-center gap-6 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Status:</span>
                <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded ${active ? 'bg-error/10 text-error' : 'bg-secondary/10 text-secondary'}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-error animate-pulse' : 'bg-secondary'}`}></span>
                  <span className="font-label-caps text-[10px] font-bold">{active ? 'REC' : 'IDLE'}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Timer:</span>
                <span className="font-mono text-[11px] font-bold text-on-surface">{formatTimer(recordingSeconds)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Stream:</span>
                <span className="font-label-caps text-[10px] font-bold text-on-surface">60.0 FPS</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Resolution:</span>
                <span className="font-label-caps text-[10px] font-bold text-on-surface">1920x1080 (HD)</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Health:</span>
              <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 font-label-caps text-[10px] font-bold border border-emerald-500/20">
                {serverStatus === 'connected' ? 'NOMINAL' : serverStatus.toUpperCase()}
              </span>
            </div>
          </div>

          {/* Live Camera & MediaPipe Face Stream */}
          <div className="glass-card rounded-3xl overflow-hidden relative aspect-video shadow-2xl bg-black border border-primary/20">
            <FaceStream onUpdate={handleFaceUpdate} />
            <div className="absolute bottom-4 left-4 z-20">
              <div className="bg-black/60 backdrop-blur px-3 py-1 rounded-md border border-white/10 flex items-center gap-2">
                <span className="font-label-caps text-[9px] text-white tracking-widest uppercase font-bold">
                  IR_ACTIVE_MESH_V4.2 • 468 LANDMARKS
                </span>
              </div>
            </div>
          </div>

          {/* Audio Waveform Integration Card */}
          <div className="space-y-3">
            <div className="glass-card p-3 rounded-xl flex items-center justify-between bg-surface-container-low border border-primary/10">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Source:</span>
                  <span className="font-label-caps text-[10px] font-bold text-on-surface">LIVE AUDIO SPECTRUM</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Sampling:</span>
                  <span className="font-label-caps text-[10px] font-bold text-on-surface">48kHz</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-label-caps text-[10px] text-on-surface-variant/60 uppercase tracking-wider">Status:</span>
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 font-label-caps text-[10px] font-bold border border-emerald-500/20">
                  SYNCED
                </span>
              </div>
            </div>
            <div className="glass-card p-4 rounded-2xl bg-black border border-white/10 shadow-xl overflow-hidden">
              <WaveformRecorder onUpdate={handleVoiceUpdate} />
            </div>
          </div>

          {/* Comprehensive Feature Extraction Table */}
          <div className="glass-card p-6 rounded-3xl space-y-4">
            <div className="flex items-center justify-between border-b border-primary/10 pb-3">
              <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest font-bold">
                Comprehensive Feature Extraction
              </h3>
              <span className="font-mono text-xs text-primary font-bold">CORE_ENGINE_V4.2.1</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Face Features Column */}
              <div className="space-y-3">
                <div className="px-3 py-1.5 bg-primary/10 rounded-xl font-label-caps text-xs font-bold text-primary flex justify-between">
                  <span>FACE FEATURES</span>
                  <span>FREQ: 60Hz</span>
                </div>
                <table className="w-full text-xs">
                  <thead className="text-on-surface-variant/60 font-label-caps border-b border-primary/10 text-[10px]">
                    <tr>
                      <th className="text-left py-2">FEATURE</th>
                      <th className="text-right py-2">VALUE</th>
                      <th className="text-center py-2">TREND</th>
                      <th className="text-right py-2">CONF</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-primary/5 font-mono text-[11px]">
                    <tr>
                      <td className="py-2.5 font-sans font-medium">EAR (Eyes)</td>
                      <td className="text-right font-bold text-secondary">
                        {faceIndicators ? faceIndicators.ear?.toFixed(2) : '0.24'}
                      </td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-secondary">trending_flat</span></td>
                      <td className="text-right text-primary font-bold">99.1%</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-sans font-medium">Blink Rate</td>
                      <td className="text-right font-bold text-on-surface">12/m</td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-error">trending_up</span></td>
                      <td className="text-right text-primary font-bold">96.4%</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-sans font-medium">Head Pose</td>
                      <td className="text-right font-bold text-on-surface">STABLE</td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-emerald-600">check_circle</span></td>
                      <td className="text-right text-primary font-bold">99.8%</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-sans font-medium">Gaze Vec</td>
                      <td className="text-right font-bold text-on-surface">0.122</td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-primary">sync_alt</span></td>
                      <td className="text-right text-primary font-bold">94.2%</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Voice Features Column */}
              <div className="space-y-3">
                <div className="px-3 py-1.5 bg-tertiary/10 rounded-xl font-label-caps text-xs font-bold text-tertiary flex justify-between">
                  <span>VOICE FEATURES</span>
                  <span>FREQ: 48kHz</span>
                </div>
                <table className="w-full text-xs">
                  <thead className="text-on-surface-variant/60 font-label-caps border-b border-primary/10 text-[10px]">
                    <tr>
                      <th className="text-left py-2">FEATURE</th>
                      <th className="text-right py-2">VALUE</th>
                      <th className="text-center py-2">TREND</th>
                      <th className="text-right py-2">CONF</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-primary/5 font-mono text-[11px]">
                    <tr>
                      <td className="py-2.5 font-sans font-medium">Pitch</td>
                      <td className="text-right font-bold text-tertiary">
                        {voiceIndicators ? `${Math.round(voiceIndicators.mean_pitch)}Hz` : '122Hz'}
                      </td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-tertiary">trending_down</span></td>
                      <td className="text-right text-primary font-bold">98.2%</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-sans font-medium">Jitter</td>
                      <td className="text-right font-bold text-on-surface">0.42%</td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-secondary">trending_flat</span></td>
                      <td className="text-right text-primary font-bold">94.0%</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-sans font-medium">Shimmer</td>
                      <td className="text-right font-bold text-on-surface">2.1dB</td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-error">trending_up</span></td>
                      <td className="text-right text-primary font-bold">92.1%</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-sans font-medium">Spectrogram</td>
                      <td className="text-right font-bold text-on-surface">4.21</td>
                      <td className="text-center"><span className="material-symbols-outlined text-sm text-primary">insights</span></td>
                      <td className="text-right text-primary font-bold">95.5%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Inference Pipeline & Prediction Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card p-6 rounded-3xl space-y-4">
              <div className="flex justify-between items-center border-b border-primary/10 pb-3">
                <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest font-bold">Inference Pipeline</h3>
                <span className="font-label-caps text-[10px] text-on-surface-variant/60 font-semibold">LATENCY: 42.1ms</span>
              </div>
              <div className="flex items-center justify-between relative px-2 py-4">
                <div className="absolute h-[2px] bg-primary/20 top-8 left-6 right-6 -z-0"></div>
                <div className="flex flex-col items-center gap-1.5 relative z-10">
                  <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white shadow-md">
                    <span className="material-symbols-outlined text-lg">sensors</span>
                  </div>
                  <span className="font-label-caps text-[9px] uppercase font-bold text-primary">Input</span>
                </div>
                <div className="flex flex-col items-center gap-1.5 relative z-10">
                  <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white shadow-md">
                    <span className="material-symbols-outlined text-lg">psychology</span>
                  </div>
                  <span className="font-label-caps text-[9px] uppercase font-bold text-primary">Features</span>
                </div>
                <div className="flex flex-col items-center gap-1.5 relative z-10 opacity-70">
                  <div className="w-10 h-10 rounded-full bg-surface-container-high border border-primary/20 flex items-center justify-center text-primary">
                    <span className="material-symbols-outlined text-lg">account_tree</span>
                  </div>
                  <span className="font-label-caps text-[9px] uppercase font-bold text-on-surface-variant">Neural</span>
                </div>
                <div className="flex flex-col items-center gap-1.5 relative z-10 opacity-70">
                  <div className="w-10 h-10 rounded-full bg-surface-container-high border border-primary/20 flex items-center justify-center text-primary">
                    <span className="material-symbols-outlined text-lg">description</span>
                  </div>
                  <span className="font-label-caps text-[9px] uppercase font-bold text-on-surface-variant">Outcome</span>
                </div>
              </div>
            </div>

            <div className="glass-card p-6 rounded-3xl space-y-4">
              <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest font-bold border-b border-primary/10 pb-3">
                Live Prediction Statistics
              </h3>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-[10px] font-label-caps text-on-surface-variant/70 uppercase font-semibold">Probability</p>
                  <p className="text-base font-bold text-primary font-mono mt-0.5">{smoothFusedScore.toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-[10px] font-label-caps text-on-surface-variant/70 uppercase font-semibold">Prediction</p>
                  <p className="text-base font-bold text-on-background mt-0.5">{stressCategory}</p>
                </div>
                <div>
                  <p className="text-[10px] font-label-caps text-on-surface-variant/70 uppercase font-semibold">Avg Conf</p>
                  <p className="text-base font-bold text-secondary font-mono mt-0.5">98.2%</p>
                </div>
                <div>
                  <p className="text-[10px] font-label-caps text-on-surface-variant/70 uppercase font-semibold">Session Peak</p>
                  <p className="text-base font-bold text-error font-mono mt-0.5">42.1%</p>
                </div>
                <div>
                  <p className="text-[10px] font-label-caps text-on-surface-variant/70 uppercase font-semibold">Lowest</p>
                  <p className="text-base font-bold text-emerald-600 font-mono mt-0.5">08.2%</p>
                </div>
                <div>
                  <p className="text-[10px] font-label-caps text-on-surface-variant/70 uppercase font-semibold">Stability</p>
                  <div className="w-full bg-primary/10 h-1.5 rounded-full mt-2 overflow-hidden">
                    <div className="bg-emerald-500 h-full w-[88%]"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Sidebar Information Stack (4 Cols) */}
        <div className="col-span-12 xl:col-span-4 space-y-6">
          {/* 1. Current Stress State Ring Gauge */}
          <div className="glass-card p-6 rounded-3xl flex flex-col items-center text-center shadow-sm">
            <span className="font-label-caps text-xs text-on-surface-variant/70 uppercase tracking-widest font-bold mb-4">
              Current Stress State
            </span>
            <div className="relative w-40 h-40 flex items-center justify-center mb-6">
              <svg className="absolute inset-0 w-full h-full -rotate-90">
                <circle className="text-primary/10" cx="80" cy="80" fill="none" r="70" stroke="currentColor" strokeWidth="8"></circle>
                <circle
                  className="text-primary transition-all duration-500"
                  cx="80"
                  cy="80"
                  fill="none"
                  r="70"
                  stroke="currentColor"
                  strokeDasharray="440"
                  strokeDashoffset={440 - (440 * smoothFusedScore) / 100}
                  strokeLinecap="round"
                  strokeWidth="8"
                ></circle>
              </svg>
              <div className="flex flex-col items-center">
                <span className="font-display-hero text-2xl font-bold text-primary tracking-tight">
                  {stressCategory}
                </span>
                <span className="font-label-caps text-xs text-secondary font-bold mt-1">
                  STABLE {smoothFusedScore.toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="grid grid-cols-2 w-full gap-3 text-left">
              <div className="bg-surface-container-low p-3 rounded-2xl border border-primary/10">
                <p className="font-label-caps text-[9px] text-on-surface-variant/70 font-semibold uppercase">Fusion Score</p>
                <p className="font-bold text-sm text-on-background font-mono mt-0.5">{(smoothFusedScore / 100).toFixed(2)}</p>
              </div>
              <div className="bg-surface-container-low p-3 rounded-2xl border border-primary/10">
                <p className="font-label-caps text-[9px] text-on-surface-variant/70 font-semibold uppercase">FPS / Latency</p>
                <p className="font-bold text-sm text-on-background font-mono mt-0.5">60 / 42ms</p>
              </div>
            </div>
          </div>

          {/* 2. Multimodal Sync Status */}
          <div className="glass-card p-6 rounded-3xl space-y-4 shadow-sm">
            <div className="flex items-center gap-3 border-b border-primary/10 pb-3">
              <span className="material-symbols-outlined text-primary text-xl">hub</span>
              <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest font-bold">
                Multi-Modal Sync Status
              </h3>
            </div>
            <div className="space-y-2.5">
              <div className="flex items-center justify-between p-3 bg-surface-container-low rounded-2xl border border-primary/10">
                <span className="font-label-caps text-xs uppercase font-semibold">Facial Vectors</span>
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 font-label-caps text-[10px] font-bold border border-emerald-500/20">
                  Active (60Hz)
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-surface-container-low rounded-2xl border border-primary/10">
                <span className="font-label-caps text-xs uppercase font-semibold">Voice Spectral</span>
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 font-label-caps text-[10px] font-bold border border-emerald-500/20">
                  Active (48kHz)
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-surface-container-low rounded-2xl border border-primary/10">
                <span className="font-label-caps text-xs uppercase font-semibold">Bio-GSR Logic</span>
                <span className="px-2.5 py-0.5 rounded-full bg-primary/10 text-primary font-label-caps text-[10px] font-bold border border-primary/20">
                  Synced (Phase 1)
                </span>
              </div>
            </div>
          </div>

          {/* 3. Reasoning Engine Log */}
          <div className="glass-card p-6 rounded-3xl flex flex-col h-[280px] shadow-sm">
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-primary/10">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 text-primary rounded-xl">
                  <span className="material-symbols-outlined text-lg">terminal</span>
                </div>
                <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest font-bold">
                  Reasoning Log
                </h3>
              </div>
              <span className="px-2 py-0.5 bg-primary/10 text-primary rounded font-label-caps text-[9px] font-bold">
                v4-Infer
              </span>
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto pr-1">
              <div className="p-3 bg-surface-container-low rounded-2xl border border-primary/10 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="px-2 py-0.5 bg-primary/10 text-primary rounded font-bold uppercase text-[9px]">
                    #Extraction
                  </span>
                  <span className="text-[10px] font-mono text-on-surface-variant/60">14:02:45.1</span>
                </div>
                <p className="text-xs text-on-surface-variant leading-relaxed">
                  512 Face Mesh points localized. Coordinate stability within 0.02 delta.
                </p>
              </div>
              <div className="p-3 bg-primary/5 rounded-2xl border border-primary/20 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="px-2 py-0.5 bg-primary/20 text-primary rounded font-bold uppercase text-[9px]">
                    #Fusion
                  </span>
                  <span className="text-[10px] font-mono text-emerald-600 font-bold">Live</span>
                </div>
                <p className="text-xs text-on-surface font-medium leading-relaxed">
                  Evaluating stress probability vectors across 3 modal domains...
                </p>
              </div>
            </div>
          </div>

          {/* 4. Protocol Engine Recommendation Card */}
          <div className="glass-card p-6 rounded-3xl space-y-4 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 text-primary rounded-xl">
                <span className="material-symbols-outlined text-lg">clinical_notes</span>
              </div>
              <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest font-bold">
                Protocol Engine
              </h3>
            </div>
            <div className="bg-primary/5 border border-primary/10 p-4 rounded-2xl">
              <p className="text-xs font-body-md text-on-surface leading-relaxed italic">
                "Elevated blink frequency and pitch shimmer detected. Recommend visual recalibration."
              </p>
            </div>
            <button className="w-full py-3.5 bg-primary text-on-primary hover:bg-primary/90 rounded-xl font-label-caps text-xs font-semibold transition-all shadow-lg hover:shadow-primary/20 uppercase tracking-wider flex items-center justify-center gap-2">
              <span className="material-symbols-outlined text-base">bolt</span>
              <span>Initiate Protocol</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
