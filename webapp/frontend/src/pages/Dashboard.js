import React, { useState, useRef, useEffect } from "react";
import RealtimeMonitor from "../components/RealtimeMonitor";
import AnalysisPanel from "../components/AnalysisPanel";
import CopilotMessage from "../components/CopilotMessage";
import StressChatbot from "../components/StressChatbot";
import { validateAnalysisInputs, validateAnalysisResponse } from "../utils/validateInputs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

import { API_BASE } from "../config";

export default function Dashboard({ dashboardMode, showCopilot, setShowCopilot, onRequestRecovery }) {
  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (response.ok) {
          const data = await response.json();
          setServerOnline(data.status === 'ok');
        } else {
          setServerOnline(false);
        }
      } catch (err) {
        setServerOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // File states
  const [faceImage, setFaceImage] = useState(null);
  const [facePreview, setFacePreview] = useState(null);
  const [voiceFile, setVoiceFile] = useState(null);
  const [voicePreviewUrl, setVoicePreviewUrl] = useState(null);
  const [eegData, setEegData] = useState("");
  const [gsrData, setGsrData] = useState("");
  const [eegFile, setEegFile] = useState(null);
  const [gsrFile, setGsrFile] = useState(null);

  // Graph preview states
  const [eegPreviewData, setEegPreviewData] = useState([]);
  const [eegPreviewKeys, setEegPreviewKeys] = useState([]);
  const [gsrPreviewData, setGsrPreviewData] = useState([]);
  const [gsrPreviewKeys, setGsrPreviewKeys] = useState([]);

  // Live capture in upload panel
  const [liveFaceResult, setLiveFaceResult] = useState(null);
  const [liveVoiceResult, setLiveVoiceResult] = useState(null);
  const [isMicRecording, setIsMicRecording] = useState(false);

  // Phase state machine
  const [phase, setPhase] = useState('idle'); // 'idle' | 'analyzing' | 'currentResult' | 'reanalyzing'
  const [currentResult, setCurrentResult] = useState(null);
  const [previousResult, setPreviousResult] = useState(null);

  // Legacy UI states
  const [error, setError] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);

  // Muse stream states
  const [museDuration, setMuseDuration] = useState(20);
  const [museFilename, setMuseFilename] = useState("uploads/eeg_session.csv");
  const [museCollecting, setMuseCollecting] = useState(false);
  const [musePoints, setMusePoints] = useState([]);
  const [museSessionError, setMuseSessionError] = useState(null);
  const [museElapsed, setMuseElapsed] = useState(0);

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const micStreamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const micCanvasRef = useRef(null);
  const processorRef = useRef(null);
  const museIntervalRef = useRef(null);

  // Scroll to top on navigation/redirect (e.g. mode or phase change)
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [dashboardMode, phase]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
      if (facePreview) URL.revokeObjectURL(facePreview);
      stopWebcam();
      stopMicRecording();
      if (museIntervalRef.current) clearInterval(museIntervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voicePreviewUrl, facePreview]);

  useEffect(() => {
    if (!webcamActive || !videoRef.current || !streamRef.current) {
      return;
    }
    const video = videoRef.current;
    video.srcObject = streamRef.current;
    video.muted = true;
    video.playsInline = true;
    video.play().catch(() => { });
  }, [webcamActive]);

  const parseDelimitedSeries = (text, keyName = "value") => {
    const values = (text || "")
      .split(/[,\s]+/)
      .map(v => parseFloat(v.trim()))
      .filter(v => !isNaN(v));
    return values.map((val, idx) => ({ index: idx, [keyName]: val }));
  };

  const handleFaceUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFaceImage(file);
      const url = URL.createObjectURL(file);
      setFacePreview(url);
      setLiveFaceResult(null);
      setError(null);
    }
  };

  const handleVoiceUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVoiceFile(file);
      const url = URL.createObjectURL(file);
      setVoicePreviewUrl(url);
      setLiveVoiceResult(null);
      setError(null);
    }
  };

  const handleEegTextChange = (text) => {
    setEegData(text);
    if (text.trim() === "") {
      setEegPreviewData([]);
      setEegPreviewKeys([]);
      return;
    }
    const parsed = parseDelimitedSeries(text, 'EEG_Signal');
    setEegPreviewData(parsed);
    setEegPreviewKeys(['EEG_Signal']);
  };

  const handleGsrTextChange = (text) => {
    setGsrData(text);
    if (text.trim() === "") {
      setGsrPreviewData([]);
      setGsrPreviewKeys([]);
      return;
    }
    const parsed = parseDelimitedSeries(text, 'GSR_Signal');
    setGsrPreviewData(parsed);
    setGsrPreviewKeys(['GSR_Signal']);
  };

  const handleEegFileUpload = (file) => {
    if (!file) return;
    setEegFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length > 0) {
        const firstLine = lines[0];
        const isHeader = firstLine.split(',').some(cell => isNaN(parseFloat(cell)));
        const dataLines = isHeader ? lines.slice(1) : lines;
        const keys = [];
        const numCols = dataLines[0] ? dataLines[0].split(',').length : 0;
        
        for (let i = 0; i < numCols; i++) {
          keys.push(`Ch_${i + 1}`);
        }

        const data = dataLines.map((line, rowIdx) => {
          const cells = line.split(',').map(c => parseFloat(c.trim()));
          const row = { index: rowIdx };
          cells.forEach((val, colIdx) => {
            if (!isNaN(val) && colIdx < keys.length) {
              row[keys[colIdx]] = val;
            }
          });
          return row;
        });

        setEegPreviewData(data);
        setEegPreviewKeys(keys);
      }
    };
    reader.readAsText(file);
  };

  const handleGsrFileUpload = (file) => {
    if (!file) return;
    setGsrFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length > 0) {
        const firstLine = lines[0];
        const isHeader = firstLine.split(',').some(cell => isNaN(parseFloat(cell)));
        const dataLines = isHeader ? lines.slice(1) : lines;
        const keys = [];
        const numCols = dataLines[0] ? dataLines[0].split(',').length : 0;
        
        for (let i = 0; i < numCols; i++) {
          keys.push(`Ch_${i + 1}`);
        }

        const data = dataLines.map((line, rowIdx) => {
          const cells = line.split(',').map(c => parseFloat(c.trim()));
          const row = { index: rowIdx };
          cells.forEach((val, colIdx) => {
            if (!isNaN(val) && colIdx < keys.length) {
              row[keys[colIdx]] = val;
            }
          });
          return row;
        });

        setGsrPreviewData(data);
        setGsrPreviewKeys(keys);
      }
    };
    reader.readAsText(file);
  };

  // Webcam helpers
  const startWebcam = async () => {
    setError(null);
    setLiveFaceResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      setWebcamActive(true);
    } catch (err) {
      setError("Unable to access camera. Please check permissions.");
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setWebcamActive(false);
  };

  const captureWebcam = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "webcam_snap.png", { type: "image/png" });
        setFaceImage(file);
        setFacePreview(URL.createObjectURL(file));
        stopWebcam();
      }
    }, 'image/png');
  };

  const analyzeLiveWebcam = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('face_image', blob, 'live_frame.png');
      try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        if (response.ok) {
          setLiveFaceResult(data);
        } else {
          setError(data.error || "Failed to analyze live webcam frame.");
        }
      } catch (err) {
        setError("Network error communicating with flask server.");
      }
    }, 'image/png');
  };

  // Audio Recorder helpers
  const startMicRecording = async () => {
    setError(null);
    setLiveVoiceResult(null);
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;

      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      source.connect(analyser);

      const bufferSize = 4096;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsMicRecording(true);
      drawLiveAudioWaveform();
    } catch (err) {
      setError("Unable to access microphone. Please check permissions.");
    }
  };

  const drawLiveAudioWaveform = () => {
    if (!micCanvasRef.current || !analyserRef.current) return;
    const canvas = micCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!analyserRef.current) return;
      requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = '#f8f9ff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2;
      ctx.strokeStyle = '#0e3b69';
      ctx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  };

  const stopMicRecording = (shouldAnalyze = false) => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    analyserRef.current = null;
    setIsMicRecording(false);

    if (shouldAnalyze && audioChunksRef.current.length > 0) {
      const totalLength = audioChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
      const mergedSamples = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of audioChunksRef.current) {
        mergedSamples.set(chunk, offset);
        offset += chunk.length;
      }

      const wavBlob = bufferToWav(mergedSamples, 16000);
      const file = new File([wavBlob], "voice_capture.wav", { type: "audio/wav" });
      setVoiceFile(file);
      setVoicePreviewUrl(URL.createObjectURL(file));
      analyzeLiveVoice(file);
    }
  };

  const analyzeLiveVoice = async (file) => {
    const formData = new FormData();
    formData.append('voice_audio', file, 'voice_live.wav');
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setLiveVoiceResult(data);
      } else {
        setError(data.error || "Failed to analyze live audio sample.");
      }
    } catch (err) {
      setError("Network error communicating with flask server.");
    }
  };

  // Muse stream capture
  const startMuseCapture = async () => {
    setMusePoints([]);
    setMuseSessionError(null);
    setMuseElapsed(0);
    try {
      const res = await fetch(`${API_BASE}/api/muse/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration: museDuration, filename: museFilename }),
      });
      const data = await res.json();
      if (res.ok && data.status === 'started') {
        setMuseCollecting(true);
        if (museIntervalRef.current) clearInterval(museIntervalRef.current);
        let timeCount = 0;
        museIntervalRef.current = setInterval(async () => {
          timeCount += 2;
          setMuseElapsed(timeCount);
          await pollMuseStatus();
          if (timeCount >= museDuration) {
            clearInterval(museIntervalRef.current);
            setMuseCollecting(false);
            analyzeMuseRecording();
          }
        }, 2000);
      } else {
        setMuseSessionError(data.message || 'Could not start Muse connection.');
      }
    } catch (e) {
      setMuseSessionError('Failed to communicate with Muse receiver api.');
    }
  };

  const stopMuseCapture = async () => {
    if (museIntervalRef.current) clearInterval(museIntervalRef.current);
    setMuseCollecting(false);
    try {
      await fetch(`${API_BASE}/api/muse/stop`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  const pollMuseStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/muse/status`);
      const data = await res.json();
      if (res.ok) {
        if (data.points && data.points.length > 0) {
          setMusePoints(data.points);
        }
        if (data.status === 'completed' && data.prediction) {
          if (museIntervalRef.current) clearInterval(museIntervalRef.current);
          setMuseCollecting(false);
          setCurrentResult(data.prediction);
          setPhase('currentResult');
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const analyzeMuseRecording = async () => {
    setPhase('analyzing');
    try {
      const res = await fetch(`${API_BASE}/api/muse/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: museFilename }),
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setCurrentResult(data);
        setPhase('currentResult');
      } else {
        setError(data.message || 'Failed to extract metrics from Muse log file.');
        setPhase('idle');
      }
    } catch (e) {
      setError('Connection dropped during Muse signal analytics.');
      setPhase('idle');
    }
  };

  // Multimodal analytics trigger
  const analyzeMultimodal = async () => {
    setError(null);
    const validationError = validateAnalysisInputs({
      faceImage,
      voiceFile,
      eegData,
      gsrData,
      eegFile,
      gsrFile,
    });
    if (validationError) {
      setError(validationError);
      return;
    }

    setPhase('analyzing');
    const formData = new FormData();

    if (faceImage) formData.append('face_image', faceImage);
    if (voiceFile) formData.append('voice_audio', voiceFile);
    if (eegFile) formData.append('eeg_file', eegFile);
    else if (eegData) formData.append('eeg_data', eegData);
    if (gsrFile) formData.append('gsr_file', gsrFile);
    else if (gsrData) formData.append('gsr_data', gsrData);

    try {
      const response = await fetch(`${API_BASE}/api/predict/upload?user_id=default`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      const responseError = validateAnalysisResponse(data, response.ok);
      if (responseError) {
        setError(responseError);
        setPhase('idle');
        return;
      }

      if (currentResult) {
        setPreviousResult(currentResult);
        setPhase('comparison');
      } else {
        setPhase('currentResult');
      }
      setCurrentResult(data);
    } catch (err) {
      setError("Failed to run stress analytics pipeline. Server might be offline.");
      setPhase('idle');
    }
  };

  const clearAll = () => {
    setFaceImage(null);
    setFacePreview(null);
    setVoiceFile(null);
    setVoicePreviewUrl(null);
    setEegData("");
    setGsrData("");
    setEegFile(null);
    setGsrFile(null);
    setEegPreviewData([]);
    setEegPreviewKeys([]);
    setGsrPreviewData([]);
    setGsrPreviewKeys([]);
    setLiveFaceResult(null);
    setLiveVoiceResult(null);
    setError(null);
    setPreviousResult(null);
    setCurrentResult(null);
    setPhase('idle');
  };

  // Audio helpers
  function bufferToWav(buffer, sampleRate) {
    const bufferLength = buffer.length;
    const wavBuffer = new ArrayBuffer(44 + bufferLength * 2);
    const view = new DataView(wavBuffer);
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + bufferLength * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, bufferLength * 2, true);
    floatTo16BitPCM(view, 44, buffer);
    return new Blob([view], { type: 'audio/wav' });
  }

  function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Handle active sub views based on dashboardMode
  return (
    <>
      {dashboardMode === 'realtime' ? (
        <RealtimeMonitor />
      ) : (
        <div className="space-y-8">
      {error && (
        <div className="p-4 bg-error-container text-on-error-container text-xs rounded-xl flex items-center gap-2 border border-error/20">
          <span className="material-symbols-outlined text-[18px]">warning</span>
          <span><strong>Error:</strong> {error}</span>
        </div>
      )}

      {/* Loading overlay */}
      {(phase === 'analyzing' || phase === 'reanalyzing') && (
        <div className="bg-surface-container-lowest border border-outline-variant/10 rounded-3xl p-16 text-center shadow-sm max-w-lg mx-auto flex flex-col items-center justify-center">
          <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center text-primary mb-4 animate-spin">
            <span className="material-symbols-outlined text-2xl">sync</span>
          </div>
          <h4 className="font-headline-sm text-headline-sm text-primary mb-2">
            {phase === 'reanalyzing' ? 'Re-analyzing Vitals...' : 'Computing Intelligence Metrics...'}
          </h4>
          <p className="text-xs text-on-surface-variant max-w-xs leading-relaxed">
            Please wait while our multimodal diagnostic models combine and process the telemetry streams.
          </p>
        </div>
      )}

      {/* Analysis Results / Comparison panels */}
      {(phase === 'currentResult' || phase === 'comparison' || phase === 'result') && currentResult && (
        <div className="space-y-6">
          <div className="flex justify-between items-center bg-surface-container-low p-4 rounded-2xl border border-outline-variant/10">
            <button
              onClick={clearAll}
              className="px-4 py-2 border border-outline text-on-surface-variant rounded-xl font-bold font-label-caps text-[11px] tracking-wider hover:bg-surface-container-high active:scale-[0.98] transition-all flex items-center gap-1.5"
            >
              <span className="material-symbols-outlined text-xs">arrow_back</span>
              Back to Data Upload
            </button>
            
            {phase === 'comparison' ? (
              <span className="text-xs font-bold text-primary font-label-caps tracking-widest uppercase">
                ✨ Comparison View Active
              </span>
            ) : currentResult.model_used ? (
              <span className="text-[10px] bg-primary-container/15 text-primary font-bold px-3 py-1.5 rounded-lg font-label-caps tracking-wider border border-primary/15">
                ⚙️ {currentResult.model_used}
              </span>
            ) : null}
          </div>

          <AnalysisPanel
            result={currentResult}
            previousResult={phase === 'comparison' ? previousResult : null}
            onRequestGame={onRequestRecovery}
          />

          {phase === 'comparison' && (
            <div className="text-center pt-4">
              <button
                onClick={clearAll}
                className="px-8 py-3 border border-outline-variant/35 text-on-surface-variant hover:bg-surface-container-low font-bold text-xs tracking-wider font-label-caps rounded-xl"
              >
                Start New Analysis
              </button>
            </div>
          )}
        </div>
      )}

      {/* Data Upload panel (Idle state) */}
      {phase === 'idle' && (
        <>
          <section className="bg-surface-container-lowest rounded-3xl p-8 border border-outline-variant/10 shadow-sm space-y-8">
            <div className="text-center space-y-2 max-w-lg mx-auto">
              <h3 className="font-headline-sm text-headline-sm text-primary">Provide Your Diagnostics</h3>
              <p className="text-xs text-on-surface-variant leading-relaxed">
                Upload any combination of facial images, voice recordings, or physiological CSV files for comprehensive stress diagnostics.
              </p>
            </div>

            {/* Bento Grid upload panels */}
            <div className="grid md:grid-cols-2 gap-6">
              
              {/* Facial input card */}
              <div className="bg-surface-container-low/40 p-6 rounded-2xl border border-outline-variant/10 flex flex-col justify-between min-h-[300px]">
                <div className="text-center space-y-1 mb-4">
                  <div className="text-primary flex justify-center"><span className="material-symbols-outlined text-[40px]">photo_camera</span></div>
                  <h4 className="font-headline-sm text-base text-primary font-bold">Facial Analysis</h4>
                  <p className="text-[11px] text-on-surface-variant">Upload a portrait photo or capture a webcam feed.</p>
                </div>

                {facePreview ? (
                  <div className="relative rounded-xl overflow-hidden border border-outline-variant/30 max-h-48 mb-4 flex items-center justify-center bg-black">
                    <img src={facePreview} alt="Face preview" className="max-h-48 w-full object-cover" />
                    <button
                      onClick={() => { setFaceImage(null); setFacePreview(null); }}
                      className="absolute top-2 right-2 bg-error text-white p-1.5 rounded-lg shadow-md hover:opacity-90 transition-all flex items-center justify-center"
                    >
                      <span className="material-symbols-outlined text-xs">delete</span>
                    </button>
                  </div>
                ) : webcamActive ? (
                  <div className="space-y-4 mb-4">
                    <video ref={videoRef} autoPlay muted playsInline className="w-full h-40 bg-black rounded-xl object-cover border border-outline-variant/20" />
                    <div className="flex gap-2 text-[10px] font-label-caps font-bold">
                      <button onClick={captureWebcam} className="flex-1 bg-primary text-on-primary py-2.5 rounded-lg shadow hover:opacity-90">Capture</button>
                      <button onClick={analyzeLiveWebcam} className="flex-1 bg-primary-container text-white py-2.5 rounded-lg shadow hover:opacity-90">Analyze Frame</button>
                      <button onClick={stopWebcam} className="flex-1 border border-outline text-on-surface-variant py-2.5 rounded-lg hover:bg-surface-container-high">Cancel</button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <input type="file" accept="image/*" onChange={handleFaceUpload} className="w-full text-xs text-on-surface-variant file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border file:border-primary/20 file:text-xs file:font-semibold file:bg-primary-container/10 file:text-primary file:cursor-pointer hover:file:bg-primary-container/20" />
                    <button onClick={startWebcam} className="w-full border border-primary text-primary font-bold text-xs tracking-wider font-label-caps py-3 rounded-xl hover:bg-primary-container/5 transition-all flex items-center justify-center gap-1.5">
                      <span className="material-symbols-outlined text-[16px]">videocam</span>
                      Use Webcam
                    </button>
                  </div>
                )}
                
                {liveFaceResult && (
                  <div className="mt-4 p-3 bg-white border border-outline-variant/10 rounded-xl flex justify-between items-center text-xs">
                    <span className="font-semibold text-on-surface-variant">Live Frame Result:</span>
                    <strong className="text-primary font-bold font-data-metric uppercase text-[10px]">
                      {liveFaceResult.stress_level} ({Number(liveFaceResult.percentage).toFixed(1)}%)
                    </strong>
                  </div>
                )}
              </div>

              {/* Vocal input card */}
              <div className="bg-surface-container-low/40 p-6 rounded-2xl border border-outline-variant/10 flex flex-col justify-between min-h-[300px]">
                <div className="text-center space-y-1 mb-4">
                  <div className="text-primary flex justify-center"><span className="material-symbols-outlined text-[40px]">mic</span></div>
                  <h4 className="font-headline-sm text-base text-primary font-bold">Vocal Strain</h4>
                  <p className="text-[11px] text-on-surface-variant">Upload an audio capture or speak directly into mic.</p>
                </div>

                {voicePreviewUrl ? (
                  <div className="space-y-4 mb-4">
                    <audio controls src={voicePreviewUrl} className="w-full mt-2" />
                    <button
                      onClick={() => { setVoiceFile(null); setVoicePreviewUrl(null); }}
                      className="w-full border border-error text-error py-2.5 rounded-xl hover:bg-error-container/10 transition-colors font-bold text-xs font-label-caps tracking-wider"
                    >
                      Remove Audio
                    </button>
                  </div>
                ) : isMicRecording ? (
                  <div className="space-y-4 mb-4">
                    <div className="w-full h-32 bg-white rounded-xl overflow-hidden border border-outline-variant/20 flex flex-col items-center justify-center relative">
                      <div className="absolute inset-0 bg-primary/5 animate-pulse"></div>
                      <canvas ref={micCanvasRef} width={280} height={100} className="w-full h-full block z-10" />
                      <div className="absolute top-2 left-3 flex items-center gap-2 z-10">
                         <span className="w-2 h-2 rounded-full bg-error animate-pulse"></span>
                         <span className="text-[10px] font-bold text-error uppercase tracking-wider">Recording</span>
                      </div>
                    </div>
                    <div className="flex gap-2 text-[10px] font-label-caps font-bold">
                       <button onClick={() => stopMicRecording(true)} className="flex-1 bg-primary text-on-primary py-2.5 rounded-lg shadow hover:opacity-90 flex items-center justify-center gap-1.5"><span className="material-symbols-outlined text-[14px]">stop</span> Stop & Analyze</button>
                       <button onClick={() => stopMicRecording(false)} className="flex-1 border border-outline text-on-surface-variant py-2.5 rounded-lg hover:bg-surface-container-high">Cancel</button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <input type="file" accept="audio/*" onChange={handleVoiceUpload} className="w-full text-xs text-on-surface-variant file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border file:border-primary/20 file:text-xs file:font-semibold file:bg-primary-container/10 file:text-primary file:cursor-pointer hover:file:bg-primary-container/20" />
                    <button
                      onClick={startMicRecording}
                      className="w-full border border-primary text-primary font-bold text-xs tracking-wider font-label-caps py-3 rounded-xl hover:bg-primary-container/5 transition-all flex items-center justify-center gap-1.5"
                    >
                      <span className="material-symbols-outlined text-[16px]">mic</span>
                      Use Microphone
                    </button>
                  </div>
                )}
                
                {liveVoiceResult && (
                  <div className="mt-4 p-3 bg-white border border-outline-variant/10 rounded-xl flex justify-between items-center text-xs">
                    <span className="font-semibold text-on-surface-variant">Live Voice Result:</span>
                    <strong className="text-primary font-bold font-data-metric uppercase text-[10px]">
                      {liveVoiceResult.stress_level} ({Number(liveVoiceResult.percentage).toFixed(1)}%)
                    </strong>
                  </div>
                )}
              </div>

              {/* Physiological data card */}
              <div className="col-span-2 bg-surface-container-low/40 p-6 rounded-2xl border border-outline-variant/10 space-y-6">
                <div className="text-center space-y-1">
                  <div className="text-primary flex justify-center"><span className="material-symbols-outlined text-[40px]">monitor_heart</span></div>
                  <h4 className="font-headline-sm text-base text-primary font-bold">Physiological Data Streams</h4>
                  <p className="text-[11px] text-on-surface-variant">Manual CSV upload or real-time Muse 2 EEG telemetry connector.</p>
                </div>



                {/* Manual csv input / text inputs */}
                <div className="grid md:grid-cols-2 gap-6 pt-4 border-t border-outline-variant/10">
                  <div className="space-y-4">
                    <div className="space-y-1">
                      <label className="text-[10px] font-bold uppercase tracking-wider text-outline px-1 block">EEG Sensor values (Text area)</label>
                      <textarea value={eegData} onChange={(e) => handleEegTextChange(e.target.value)} placeholder="e.g. 0.5, 0.7, 0.6, 0.8, 0.65..." className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary text-xs font-semibold" rows="2" />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[10px] font-bold uppercase tracking-wider text-outline px-1 block">EEG CSV File Upload</label>
                      <input type="file" accept=".csv,.txt" onChange={(e) => handleEegFileUpload(e.target.files[0] || null)} className="w-full text-xs text-on-surface-variant file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border file:border-primary/20 file:text-xs file:font-semibold file:bg-primary-container/10 file:text-primary file:cursor-pointer" />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-1">
                      <label className="text-[10px] font-bold uppercase tracking-wider text-outline px-1 block">GSR Conductance values (Text area)</label>
                      <textarea value={gsrData} onChange={(e) => handleGsrTextChange(e.target.value)} placeholder="e.g. 2.1, 2.3, 2.5, 2.4, 2.6..." className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary text-xs font-semibold" rows="2" />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[10px] font-bold uppercase tracking-wider text-outline px-1 block">GSR CSV File Upload</label>
                      <input type="file" accept=".csv,.txt" onChange={(e) => handleGsrFileUpload(e.target.files[0] || null)} className="w-full text-xs text-on-surface-variant file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border file:border-primary/20 file:text-xs file:font-semibold file:bg-primary-container/10 file:text-primary file:cursor-pointer" />
                    </div>
                  </div>
                </div>

                {/* Recharts graph previews for uploaded CSV */}
                {(eegPreviewData.length > 0 || gsrPreviewData.length > 0) && (
                  <div className="grid md:grid-cols-2 gap-4 pt-6 border-t border-outline-variant/10">
                    {eegPreviewData.length > 0 && (
                      <div className="bg-white p-4 rounded-xl border border-outline-variant/10 shadow-sm space-y-3">
                        <h5 className="text-[10px] text-primary uppercase font-bold tracking-wider font-label-caps">EEG Signal preview</h5>
                        <div className="h-44 w-full">
                          <ResponsiveContainer>
                            <LineChart data={eegPreviewData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(120, 120, 120, 0.15)" />
                              <XAxis dataKey="index" tick={{ fill: '#737780', fontSize: 9 }} />
                              <YAxis tick={{ fill: '#737780', fontSize: 9 }} />
                              <Tooltip />
                              {eegPreviewKeys.map((key, idx) => (
                                <Line key={key} type="monotone" dataKey={key} dot={false} strokeWidth={1.5} stroke={["#0e3b69", "#2c5282", "#bc6c25", "#c74545"][idx % 4]} />
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}
                    {gsrPreviewData.length > 0 && (
                      <div className="bg-white p-4 rounded-xl border border-outline-variant/10 shadow-sm space-y-3">
                        <h5 className="text-[10px] text-primary uppercase font-bold tracking-wider font-label-caps">GSR Signal preview</h5>
                        <div className="h-44 w-full">
                          <ResponsiveContainer>
                            <LineChart data={gsrPreviewData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(120, 120, 120, 0.15)" />
                              <XAxis dataKey="index" tick={{ fill: '#737780', fontSize: 9 }} />
                              <YAxis tick={{ fill: '#737780', fontSize: 9 }} />
                              <Tooltip />
                              {gsrPreviewKeys.map((key, idx) => (
                                <Line key={key} type="monotone" dataKey={key} dot={false} strokeWidth={1.5} stroke={["#2c5282", "#0e3b69", "#bc6c25"][idx % 3]} />
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Form actions */}
            <div className="flex gap-4 items-center justify-center pt-6 border-t border-outline-variant/10">
              <button
                onClick={analyzeMultimodal}
                disabled={(phase === 'analyzing' || phase === 'reanalyzing') || !serverOnline}
                className="bg-primary text-on-primary font-bold text-xs font-label-caps tracking-wider px-10 py-4 rounded-xl shadow hover:opacity-95 active:scale-95 disabled:opacity-50 transition-all"
              >
                Analyze Stress Index
              </button>
              <button
                onClick={clearAll}
                className="border border-outline text-on-surface-variant font-bold text-xs font-label-caps tracking-wider px-8 py-4 rounded-xl hover:bg-surface-container active:scale-95 transition-all"
              >
                Clear All Fields
              </button>
            </div>
          </section>

          {/* Guide banner card */}
          <section className="bg-surface-container-lowest rounded-3xl p-8 border border-outline-variant/10 shadow-sm space-y-6">
            <h3 className="font-headline-sm text-headline-sm text-primary text-center">Bento Diagnostics Guide</h3>
            <div className="grid md:grid-cols-3 gap-6 pt-2 text-xs leading-relaxed text-on-surface-variant font-medium">
              <div className="space-y-2">
                <h4 className="font-bold text-sm text-primary font-label-caps uppercase tracking-wider">📸 Facial Data</h4>
                <p>Ensure your face is clearly lit. The system tracks 18 features including jaw clenches, brow descent, and lip compression ratios to detect involuntary sympathetic nervous indices.</p>
              </div>
              <div className="space-y-2">
                <h4 className="font-bold text-sm text-primary font-label-caps uppercase tracking-wider">🎤 Voice Data</h4>
                <p>Provide a short voice sample (3-5 seconds). We evaluate vocal tremor indicators (fundamental frequency standard deviation, jitter percent, and amplitude shimmers).</p>
              </div>
              <div className="space-y-2">
                <h4 className="font-bold text-sm text-primary font-label-caps uppercase tracking-wider">⚡ Physiological Data</h4>
                <p>Input raw text arrays or upload CSV log sheets. Supports multi-channel raw EEG brainwave amplitudes and galvanic skin response (GSR) conduction cycles.</p>
              </div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-2xl border border-outline-variant/15 text-center text-xs font-semibold text-primary">
              💡 Pro Tip: For maximum confidence, provide multiple modalities together. Our models automatically weigh sensor reliability depending on input quality.
            </div>
          </section>
        </>
      )}
      </div>
      )}

      {/* Dynamic Chatbot Panel */}
      <StressChatbot
        stressLevel={currentResult?.stress_level || 'Moderate'}
        stressPercentage={currentResult ? currentResult.fused_score * 100 : null}
        open={showCopilot}
        onClose={() => setShowCopilot(false)}
      />
    </>
  );
}