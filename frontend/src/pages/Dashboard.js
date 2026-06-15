import React, { useState, useRef, useEffect, useCallback } from "react";
import RealtimeMonitor from "../components/RealtimeMonitor";
import "../theme.css";
import AnalysisPanel from "../components/AnalysisPanel";
import CopilotMessage from "../components/CopilotMessage";
import GamePanel from "../components/GamePanel";
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

const API_BASE = "http://127.0.0.1:5000";

// Standard PCM WAV encoder helpers
function bufferToWav(buffer, sampleRate) {
  const bufferLength = buffer.length;
  const wavBuffer = new ArrayBuffer(44 + bufferLength * 2);
  const view = new DataView(wavBuffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + bufferLength * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
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

export default function Dashboard({ theme, toggleTheme }) {
  const [mode, setMode] = useState('upload'); // 'upload' or 'realtime'
  const [serverOnline, setServerOnline] = useState(true);
  
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
  const [phase, setPhase] = useState('idle'); // 'idle' | 'analyzing' | 'currentResult' | 'game' | 'reanalyzing' | 'comparison'
  const [currentResult, setCurrentResult] = useState(null);
  const [previousResult, setPreviousResult] = useState(null);
  const [analysisPayload, setAnalysisPayload] = useState(null);

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
  const waveformFrameRef = useRef(null);
  const micCanvasRef = useRef(null);
  const processorRef = useRef(null);

  // Background Server Health check
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        setServerOnline(response.ok);
      } catch (err) {
        setServerOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
      if (facePreview) URL.revokeObjectURL(facePreview);
      stopWebcam();
      stopMicRecording();
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
    video.play().catch(() => {});
  }, [webcamActive]);

  const parseDelimitedSeries = (text, keyName = "value") => {
    const values = (text || "")
      .split(/[\s,;]+/)
      .map((item) => Number(item.trim()))
      .filter((item) => Number.isFinite(item));

    return values.slice(0, 300).map((value, index) => ({ index, [keyName]: value }));
  };

  const parseSignalCsvForPreview = (file, preferredHeaders = []) =>
    new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const text = String(reader.currentResult || "");
          const rows = text
            .split(/\r?\n/)
            .map((row) => row.trim())
            .filter(Boolean)
            .slice(0, 1200);

          if (rows.length === 0) {
            resolve({ data: [], keys: [] });
            return;
          }

          const firstRow = rows[0].split(",").map((cell) => cell.trim());
          const hasHeader = firstRow.some((cell) => Number.isNaN(Number(cell)));
          const headers = hasHeader ? firstRow : firstRow.map((_, index) => `col_${index}`);
          const bodyRows = hasHeader ? rows.slice(1) : rows;

          const normalizedPreferred = preferredHeaders.map((item) => item.toLowerCase());
          const selectedIndexes = headers
            .map((header, index) => ({ header, index }))
            .filter(({ header }) => {
              const normalized = header.toLowerCase().replace(/\s+/g, "");
              if (normalized.includes("timestamp") || normalized === "time") return false;
              return (
                normalizedPreferred.length === 0 ||
                normalizedPreferred.includes(normalized) ||
                normalizedPreferred.includes(header.toLowerCase())
              );
            })
            .map((entry) => entry.index);

          const fallBackIndexes =
            selectedIndexes.length > 0
              ? selectedIndexes
              : headers
                  .map((header, index) => ({ header, index }))
                  .filter(({ header }) => !header.toLowerCase().includes("timestamp"))
                  .map((entry) => entry.index)
                  .slice(0, 4);

          const safeIndexes = fallBackIndexes.slice(0, 5);
          const safeKeys = safeIndexes.map((idx) => headers[idx].replace(/\s+/g, "") || `col_${idx}`);

          const points = [];
          for (let i = 0; i < bodyRows.length && points.length < 280; i += 1) {
            const cells = bodyRows[i].split(",").map((cell) => cell.trim());
            const point = { index: points.length };
            let hasAny = false;

            safeIndexes.forEach((sourceIdx, kIdx) => {
              const value = Number(cells[sourceIdx]);
              if (Number.isFinite(value)) {
                point[safeKeys[kIdx]] = value;
                hasAny = true;
              }
            });

            if (hasAny) points.push(point);
          }

          resolve({ data: points, keys: safeKeys });
        } catch (_err) {
          resolve({ data: [], keys: [] });
        }
      };

      reader.onerror = () => resolve({ data: [], keys: [] });
      reader.readAsText(file);
    });

  const stopMicRecording = (shouldAnalyze = false) => {
    if (waveformFrameRef.current) {
      cancelAnimationFrame(waveformFrameRef.current);
      waveformFrameRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setIsMicRecording(false);

    if (shouldAnalyze) {
      const samples = audioChunksRef.current;
      if (samples && samples.length > 0) {
        const sampleRate = 16000;
        const wavBlob = bufferToWav(samples, sampleRate);
        const file = new File([wavBlob], "live-recording.wav", { type: "audio/wav" });
        setVoiceFile(file);
        const url = URL.createObjectURL(wavBlob);
        setVoicePreviewUrl(url);
        analyzeVoiceFile(file);
      }
    }
    audioChunksRef.current = [];
  };

  const analyzeVoiceFile = async (fileToAnalyze) => {
    try {
      const formData = new FormData();
      formData.append("file", fileToAnalyze);
      const response = await fetch(`${API_BASE}/api/voice/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.status === "success") {
        setLiveVoiceResult(data);
      } else {
        setError(data.message || "Live voice analysis failed.");
      }
    } catch (err) {
      setError(`Live voice analysis failed: ${err.message}`);
    }
  };

  const startMicRecording = async () => {
    try {
      setError(null);
      setLiveVoiceResult(null);
      audioChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      micStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      source.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(...inputData);
      };

      const updateWaveform = () => {
        if (!analyserRef.current || !micCanvasRef.current) return;
        const canvas = micCanvasRef.current;
        const ctx = canvas.getContext("2d");
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
          if (!analyserRef.current || !micCanvasRef.current) return;
          waveformFrameRef.current = requestAnimationFrame(draw);

          analyserRef.current.getByteTimeDomainData(dataArray);

          const styles = getComputedStyle(canvas);
          const bgColor = styles.getPropertyValue('--chat-bg').trim() || '#050510';
          const themePrimary = styles.getPropertyValue('--primary-color').trim() || '#00f2ff';

          ctx.fillStyle = bgColor;
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          ctx.lineWidth = 2;
          ctx.strokeStyle = themePrimary;
          ctx.beginPath();

          const sliceWidth = canvas.width / bufferLength;
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
      updateWaveform();

      setIsMicRecording(true);
    } catch (err) {
      setError(`Could not start microphone recording: ${err.message}`);
      stopMicRecording(false);
    }
  };

  const analyzeLiveWebcam = async () => {
    if (!videoRef.current || !canvasRef.current) {
      setError("Webcam is not active. Please start webcam first.");
      return;
    }
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      const base64Image = canvas.toDataURL("image/jpeg", 0.9);
      const response = await fetch(`${API_BASE}/api/webcam/capture`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image }),
      });

      const data = await response.json();
      if (data.status === "success") {
        setLiveFaceResult(data);
      } else {
        setError(data.message || "Live webcam analysis failed.");
      }
    } catch (err) {
      setError(`Live webcam analysis failed: ${err.message}`);
    }
  };

  const handleFaceUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload a valid image file (JPG, JPEG, PNG)');
        return;
      }
      setFaceImage(file);
      setFacePreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleVoiceUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('audio/')) {
        setError('Please upload a valid audio file');
        return;
      }
      try {
        setError(null);
        const arrayBuffer = await file.arrayBuffer();
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioCtx({ sampleRate: 16000 });
        
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);
        const wavBlob = bufferToWav(channelData, 16000);
        
        const cleanName = file.name.replace(/\.[^/.]+$/, "") + ".wav";
        const wavFile = new File([wavBlob], cleanName, { type: 'audio/wav' });
        
        audioCtx.close();
        
        setVoiceFile(wavFile);
        setVoicePreviewUrl(URL.createObjectURL(wavBlob));
      } catch (err) {
        console.error(err);
        setError(`Failed to process audio file: ${err.message}. Please upload a standard WAV or MP3 file.`);
      }
    }
  };

  const handleEegTextChange = (value) => {
    setEegData(value);
    const preview = parseDelimitedSeries(value, "EEG");
    setEegPreviewData(preview);
    setEegPreviewKeys(preview.length > 0 ? ["EEG"] : []);
  };

  const handleGsrTextChange = (value) => {
    setGsrData(value);
    const preview = parseDelimitedSeries(value, "GSR");
    setGsrPreviewData(preview);
    setGsrPreviewKeys(preview.length > 0 ? ["GSR"] : []);
  };

  const handleEegFileUpload = async (file) => {
    setEegFile(file || null);
    if (!file) {
      setEegPreviewData([]);
      setEegPreviewKeys([]);
      return;
    }
    const preview = await parseSignalCsvForPreview(file, ["tp9", "af7", "af8", "tp10", "rightaux", "right aux"]);
    setEegPreviewData(preview.data);
    setEegPreviewKeys(preview.keys);
  };

  const handleGsrFileUpload = async (file) => {
    setGsrFile(file || null);
    if (!file) {
      setGsrPreviewData([]);
      setGsrPreviewKeys([]);
      return;
    }
    const preview = await parseSignalCsvForPreview(file, []);
    setGsrPreviewData(preview.data);
    setGsrPreviewKeys(preview.keys.slice(0, 2));
  };

  const startWebcam = async () => {
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 960 },
          height: { ideal: 540 },
          facingMode: "user",
        }
      });
      streamRef.current = stream;
      setWebcamActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
        videoRef.current.play().catch(() => {});
      }
    } catch (err) {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      streamRef.current = null;
      setWebcamActive(false);
      setError('Could not access webcam: ' + err.message);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
  };

  const captureWebcam = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
        setFaceImage(file);
        setFacePreview(URL.createObjectURL(file));
        stopWebcam();
      }, 'image/jpeg');
    }
  };

    const analyzeMultimodal = async () => {
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
      const response = await fetch(`${API_BASE}/api/multimodal/analyze`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      if (data.status === 'error' || data.error) {
        setError(data.message || data.error || 'Analysis failed');
        setPhase('idle');
        return;
      }
      
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
      const response = await fetch(`${API_BASE}/api/multimodal/analyze`, {
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
  const pollMuseStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/muse/status?limit=280`);
      const data = await response.json();
      if (data.status !== 'success') return;

      setMuseCollecting(Boolean(data.collecting));
      setMuseElapsed(Number(data.elapsed_seconds || 0));
      setMusePoints(Array.isArray(data.points) ? data.points : []);

      if (data.error) {
        setMuseSessionError(data.error);
      }
      if (data.prediction && data.prediction.status === 'success') {
        setCurrentResult(data.prediction);
      }
    } catch (err) {
      setMuseSessionError('Could not poll Muse status: ' + err.message);
    }
  };

  const startMuseCapture = async () => {
    setMuseSessionError(null);
    try {
      const response = await fetch(`${API_BASE}/api/muse/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          duration: Number(museDuration),
          filename: museFilename,
        }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setMuseCollecting(true);
        setMuseElapsed(0);
        setMusePoints([]);
      } else {
        setMuseSessionError(data.message || 'Could not start Muse recording.');
      }
    } catch (err) {
      setMuseSessionError('Could not start Muse recording: ' + err.message);
    }
  };

  const stopMuseCapture = async () => {
    try {
      await fetch(`${API_BASE}/api/muse/stop`, { method: 'POST' });
      setMuseCollecting(false);
      await pollMuseStatus();
    } catch (err) {
      setMuseSessionError('Could not stop Muse recording: ' + err.message);
    }
  };

  useEffect(() => {
    let timer = null;
    if (museCollecting) {
      timer = setInterval(() => {
        pollMuseStatus();
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [museCollecting]);

  const clearAll = () => {
    setFaceImage(null);
    setFacePreview(null);
    setVoiceFile(null);
    setVoicePreviewUrl(null);
    setEegData("");
    setGsrData("");
    setEegFile(null);
    setGsrFile(null);
    setCurrentResult(null); setPhase('idle');
    setError(null);
    setMusePoints([]);
    setMuseSessionError(null);
    setMuseElapsed(0);
    setEegPreviewData([]);
    setEegPreviewKeys([]);
    setGsrPreviewData([]);
    setGsrPreviewKeys([]);
    setLiveFaceResult(null);
    setLiveVoiceResult(null);
    stopWebcam();
    stopMicRecording(false);
  };



  return (
    <div className="container py-5" style={{ position: 'relative' }}>
      {/* SHUTDOWN & RESTART CONTROLS */}
      <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 1000, display: 'flex', gap: '10px' }}>
        <button 
          title="Restart Backend Server"
          style={{
            width: '45px',
            height: '45px',
            borderRadius: '50%',
            backgroundColor: '#fd7e14',
            color: 'white',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 0 10px rgba(253, 126, 20, 0.6)'
          }}
          onClick={() => {
            if(window.confirm("Are you sure you want to restart the backend server?")) {
              fetch('http://127.0.0.1:5000/api/restart/backend', { method: 'POST' })
                .then(() => alert("Backend is restarting. Please wait a few seconds before analyzing again."))
                .catch(e => console.error(e));
            }
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="1 4 1 10 7 10"></polyline>
            <polyline points="23 20 23 14 17 14"></polyline>
            <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path>
          </svg>
        </button>
        <button 
          title="Shutdown Backend Server"
          style={{
            width: '45px',
            height: '45px',
            borderRadius: '50%',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 0 10px rgba(220, 53, 69, 0.6)'
          }}
          onClick={() => {
            if(window.confirm("Are you sure you want to shut down the backend server?")) {
              fetch('http://127.0.0.1:5000/api/shutdown/backend', { method: 'POST' })
                .then(() => alert("Backend is shutting down."))
                .catch(e => console.error(e));
            }
          }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18.36 6.64a9 9 0 1 1-12.73 0"></path>
            <line x1="12" y1="2" x2="12" y2="12"></line>
          </svg>
        </button>
      </div>

      <div className="text-center mb-5">
        <h2 className="neon-text">Multimodal Stress Detection</h2>
        <p className="lead">Intelligent stress analysis using facial, vocal, and physiological indicators</p>

        <div style={{ display: "flex", justifyContent: "center", gap: "1.5rem", flexWrap: "wrap", alignItems: "center" }}>
          {/* Connection Status Badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontSize: '0.75rem',
            padding: '6px 14px',
            borderRadius: 20,
            background: 'var(--accent-light-bg)',
            border: `1px solid ${serverOnline ? 'var(--primary-color)' : 'rgba(244, 67, 54, 0.35)'}`,
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            transition: 'border-color 0.3s ease',
            marginTop: '1rem'
          }}>
            <span style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: serverOnline ? 'var(--primary-color)' : '#F44336',
              display: 'inline-block',
              animation: serverOnline ? 'pulseGlow 1.8s infinite ease-in-out' : 'none',
              boxShadow: `0 0 8px ${serverOnline ? 'var(--primary-color)' : '#F44336'}`
            }} />
            <span style={{
              color: serverOnline ? 'var(--primary-color)' : '#F44336',
              fontWeight: 700,
              letterSpacing: '0.6px',
              fontFamily: 'monospace'
            }}>
              {serverOnline ? 'SERVER ONLINE' : 'SERVER OFFLINE'}
            </span>
          </div>

          <div style={{ display: "flex", gap: "12px", marginTop: "1rem" }}>
            <button
              className={`btn ${mode === 'upload' && phase !== 'game' ? 'btn-neon' : 'btn-outline-neon'}`}
              onClick={() => {
                setMode('upload');
                setPhase(currentResult ? 'result' : 'idle');
              }}
            >
              📂 Upload & Livestreams
            </button>
            <button
              className={`btn ${mode === 'realtime' && phase !== 'game' ? 'btn-neon' : 'btn-outline-neon'}`}
              onClick={() => {
                setMode('realtime');
                setPhase('idle');
              }}
            >
              🔴 Real-Time Streaming
            </button>
            <button
              className={`btn ${phase === 'game' ? 'btn-neon' : 'btn-outline-neon'}`}
              onClick={() => setPhase('game')}
            >
              🎮 Relaxation Game
            </button>
          </div>

          <div style={{ display: "flex", gap: "12px", marginTop: "1rem" }}>
            <button
              className="btn btn-outline-neon"
              onClick={toggleTheme}
            >
              🎨 Style: {theme === 'cyber' ? 'Cyber ⇆ Earthy' : 'Earthy ⇆ Cyber'}
            </button>
          </div>
        </div>
      </div>

      {mode === 'realtime' && phase !== 'game' ? (
        <RealtimeMonitor />
      ) : (
        <>
          {/* Error Display */}
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
          {(phase === 'currentResult' || phase === 'comparison' || phase === 'result') && currentResult && (
            <div className="result-view fade-in-up">
              {/* Navigation Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                <button 
                  onClick={clearAll}
                  className="btn btn-outline-neon"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    padding: '8px 16px',
                    fontSize: '0.85rem',
                    borderRadius: 8
                  }}
                >
                  <span>←</span> Back to Data Upload
                </button>
                
                {phase === 'comparison' && (
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                    ✨ Comparison View Active
                  </span>
                )}
              </div>

              {/* Main Analysis Card (Full Width) */}
              <div className="mb-4">
                <AnalysisPanel
                  result={currentResult}
                  previousResult={phase === 'comparison' ? previousResult : null}
                  onRequestGame={handleRequestGame}
                  theme={theme}
                />
              </div>

              {/* Sub-cards Row (AI Insights) */}
              <div className="row">
                <div className="col-12 mb-4">
                  <CopilotMessage 
                    stressLevel={currentResult?.stress_level} 
                    explainability={currentResult?.explainability} 
                  />
                </div>
              </div>
            </div>
          )}

          {/* Game panel */}
          {phase === 'game' && (
            <GamePanel
              stressLevel={currentResult?.stress_level}
              onGameComplete={handleGameComplete}
              onDismiss={() => setPhase(currentResult ? 'result' : 'idle')}
            />
          )}

          {/* Re-analyze button after comparison */}
          {phase === 'comparison' && (
            <div style={{ marginTop: 16, textAlign: 'center' }}>
              <button onClick={() => {
                setPhase('idle');
                setCurrentResult(null); setPhase('idle');
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

          {/* Input Section */}
          {phase === 'idle' && (
          <>
          <div className="row">
            <div className="col-12">
              <div className="neon-card">
                <h3 className="text-center mb-4">Provide Your Data</h3>
                <p className="text-center" style={{color: 'var(--text-muted)', marginBottom: '2rem'}}>
                  Upload any combination of facial images, voice recordings, or physiological data for comprehensive stress analysis
                </p>

                <div className="row">
                  {/* Facial Input */}
                  <div className="col-md-6 mb-4">
                    <div style={{
                      border: '2px dashed rgba(120, 120, 120, 0.3)',
                      borderRadius: '12px',
                      padding: '1.5rem',
                      background: 'rgba(120, 120, 120, 0.05)'
                    }}>
                      <div className="text-center mb-3">
                        <span style={{fontSize: '3rem'}}>📸</span>
                        <h4>Facial Analysis</h4>
                        <p style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>
                          Upload a photo or use webcam
                        </p>
                      </div>

                      {facePreview && (
                        <div style={{marginBottom: '1rem', position: 'relative'}}>
                          <img 
                            src={facePreview} 
                            alt="Preview" 
                            style={{
                              width: '100%',
                              borderRadius: '8px',
                              maxHeight: '200px',
                              objectFit: 'cover'
                            }}
                          />
                          <button
                            onClick={() => {
                              setFaceImage(null);
                              setFacePreview(null);
                            }}
                            className="btn btn-danger"
                            style={{
                              position: 'absolute',
                              top: '8px',
                              right: '8px',
                              padding: '0.25rem 0.5rem',
                              fontSize: '0.875rem'
                            }}
                          >
                            Remove
                          </button>
                        </div>
                      )}

                      {webcamActive && (
                        <div style={{marginBottom: '1rem'}}>
                          <video 
                            ref={videoRef} 
                            autoPlay 
                            muted
                            playsInline
                            onLoadedMetadata={() => {
                              if (videoRef.current) {
                                videoRef.current.play().catch(() => {});
                              }
                            }}
                            style={{
                              width: '100%',
                              minHeight: '260px',
                              borderRadius: '8px',
                              background: 'rgba(0, 0, 0, 0.35)',
                              objectFit: 'cover'
                            }}
                          />
                          <button
                            onClick={captureWebcam}
                            className="btn btn-neon w-100 mt-2"
                          >
                            📷 Capture Photo
                          </button>
                          <button
                            onClick={analyzeLiveWebcam}
                            disabled={!serverOnline}
                            className="btn btn-outline-neon w-100 mt-2"
                          >
                            {serverOnline ? '⚡ Analyze Live Frame' : '🔌 Server Offline'}
                          </button>
                          <button
                            onClick={stopWebcam}
                            className="btn btn-outline-neon w-100 mt-2"
                          >
                            Cancel
                          </button>

                          {liveFaceResult && (
                            <div className="currentResult-panel-card" style={{ marginTop: '0.75rem' }}>
                              <small>Live Facial Result</small>
                              <div style={{ fontWeight: 700 }}>
                                {liveFaceResult.stress_level || liveFaceResult.predicted_class} ({Number(liveFaceResult.percentage || 0).toFixed(1)}%)
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {!webcamActive && !facePreview && (
                        <>
                          <input
                            type="file"
                            accept="image/*"
                            onChange={handleFaceUpload}
                            className="form-control mb-2"
                          />
                          <button
                            onClick={startWebcam}
                            className="btn btn-outline-neon w-100"
                          >
                            📹 Use Webcam
                          </button>
                        </>
                      )}
                      <canvas ref={canvasRef} style={{display: 'none'}} />
                    </div>
                  </div>

                  {/* Voice Input */}
                  <div className="col-md-6 mb-4">
                    <div style={{
                      border: '2px dashed rgba(120, 120, 120, 0.3)',
                      borderRadius: '12px',
                      padding: '1.5rem',
                      background: 'rgba(120, 120, 120, 0.05)'
                    }}>
                      <div className="text-center mb-3">
                        <span style={{fontSize: '3rem'}}>🎤</span>
                        <h4>Voice Analysis</h4>
                        <p style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>
                          Upload an audio recording
                        </p>
                      </div>

                      {voicePreviewUrl && (
                        <div style={{marginBottom: '1rem'}}>
                          <audio 
                            controls 
                            src={voicePreviewUrl} 
                            style={{width: '100%', marginBottom: '0.5rem'}} 
                          />
                          <button
                            onClick={() => {
                              setVoiceFile(null);
                              setVoicePreviewUrl(null);
                            }}
                            className="btn btn-danger w-100"
                            style={{fontSize: '0.875rem'}}
                          >
                            Remove Audio
                          </button>
                        </div>
                      )}

                      {!voicePreviewUrl && (
                        <>
                          <input
                            type="file"
                            accept="audio/*"
                            onChange={handleVoiceUpload}
                            className="form-control"
                          />
                          <small style={{color: 'var(--text-muted)', display: 'block', marginTop: '0.5rem'}}>
                            Supported: WAV, MP3, OGG, M4A, WEBM
                          </small>

                          <div style={{ display: 'flex', gap: '0.6rem', marginTop: '0.6rem', flexWrap: 'wrap' }}>
                            <button
                              type="button"
                              className="btn btn-neon"
                              onClick={startMicRecording}
                              disabled={isMicRecording}
                            >
                              🎙️ Start Mic
                            </button>
                            <button
                              type="button"
                              className="btn btn-outline-neon"
                              onClick={() => stopMicRecording(true)}
                              disabled={!isMicRecording}
                            >
                              ⏹️ Stop & Analyze
                            </button>
                          </div>

                          <div style={{ width: '100%', height: 100, marginTop: '0.75rem' }}>
                            <canvas
                              ref={micCanvasRef}
                              width={320}
                              height={100}
                              style={{ width: '100%', height: 100, display: 'block', background: 'var(--chat-bg)', borderRadius: 8 }}
                            />
                          </div>

                          {liveVoiceResult && (
                            <div className="currentResult-panel-card" style={{ marginTop: '0.75rem' }}>
                              <small>Live Voice Result</small>
                              <div style={{ fontWeight: 700 }}>
                                {liveVoiceResult.stress_level || liveVoiceResult.predicted_class} ({Number(liveVoiceResult.percentage || 0).toFixed(1)}%)
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>

                  {/* Physiological Input */}
                  <div className="col-12 mb-4">
                    <div style={{
                      border: '2px dashed rgba(120, 120, 120, 0.3)',
                      borderRadius: '12px',
                      padding: '1.5rem',
                      background: 'rgba(120, 120, 120, 0.05)'
                    }}>
                      <div className="text-center mb-3">
                        <span style={{fontSize: '3rem'}}>🧠⚡</span>
                        <h4>Physiological Data</h4>
                        <p style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>
                          Enter EEG and GSR data as comma-separated values, or use Muse 2 live stream
                        </p>
                      </div>

                      <div style={{
                        marginBottom: '1.5rem',
                        border: '1px solid rgba(120, 120, 120, 0.3)',
                        borderRadius: '10px',
                        padding: '1rem',
                        background: 'var(--accent-light-bg)'
                      }}>
                        <h5 style={{ marginBottom: '0.75rem' }}>Muse 2 Real-Time Stream</h5>
                        <p style={{ color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                          Uses muselsl command: python -m muselsl record --duration X --filename uploads/eeg_session.csv
                        </p>

                        <div className="row">
                          <div className="col-md-4 mb-2">
                            <label className="form-label"><strong>Duration (seconds)</strong></label>
                            <input
                              type="number"
                              min="5"
                              max="1800"
                              className="form-control"
                              value={museDuration}
                              onChange={(e) => setMuseDuration(e.target.value)}
                            />
                          </div>
                          <div className="col-md-8 mb-2">
                            <label className="form-label"><strong>CSV output path</strong></label>
                            <input
                              type="text"
                              className="form-control"
                              value={museFilename}
                              onChange={(e) => setMuseFilename(e.target.value)}
                              placeholder="uploads/eeg_session.csv"
                            />
                          </div>
                        </div>

                        <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                          <button
                            type="button"
                            className="btn btn-neon"
                            disabled={museCollecting}
                            onClick={startMuseCapture}
                          >
                            {museCollecting ? 'Collecting...' : 'Start Muse Stream'}
                          </button>
                          <button
                            type="button"
                            className="btn btn-outline-neon"
                            disabled={!museCollecting}
                            onClick={stopMuseCapture}
                          >
                            Stop Stream
                          </button>
                          <button
                            type="button"
                            className="btn btn-outline-neon"
                            onClick={pollMuseStatus}
                          >
                            Refresh Status
                          </button>
                        </div>

                        <div style={{ marginTop: '0.75rem', color: 'var(--text-muted)' }}>
                          <strong>Status:</strong> {museCollecting ? 'Collecting live data' : 'Idle'} | <strong>Elapsed:</strong> {museElapsed}s
                        </div>

                        {museSessionError && (
                          <div style={{ marginTop: '0.5rem', color: '#c74545' }}>
                            <strong>Error:</strong> {museSessionError}
                          </div>
                        )}

                        <div style={{ width: '100%', height: 280, marginTop: '1rem' }}>
                          <ResponsiveContainer>
                            <LineChart data={musePoints}>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(120, 120, 120, 0.2)" />
                              <XAxis dataKey="timestamp" tick={{ fill: 'var(--text-color)', fontSize: 12 }} />
                              <YAxis tick={{ fill: 'var(--text-color)', fontSize: 12 }} />
                              <Tooltip />
                              <Legend />
                              <Line type="monotone" dataKey="TP9" stroke="#4f772d" dot={false} strokeWidth={2} />
                              <Line type="monotone" dataKey="AF7" stroke="#8d9740" dot={false} strokeWidth={2} />
                              <Line type="monotone" dataKey="AF8" stroke="#bc6c25" dot={false} strokeWidth={2} />
                              <Line type="monotone" dataKey="TP10" stroke="#c74545" dot={false} strokeWidth={2} />
                              <Line type="monotone" dataKey="RightAUX" stroke="#6a4c93" dot={false} strokeWidth={2} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>

                        <small style={{ color: 'var(--text-muted)' }}>
                          Expected columns: timestamps, TP9, AF7, AF8, TP10, Right AUX. Prediction is triggered automatically when recording finishes.
                        </small>
                      </div>

                      <div className="row">
                        <div className="col-md-6 mb-3">
                          <label className="form-label">
                            <strong>🧠 EEG Data</strong>
                          </label>
                          <textarea
                            value={eegData}
                            onChange={(e) => handleEegTextChange(e.target.value)}
                            placeholder="e.g., 0.5, 0.7, 0.6, 0.8, 0.65, 0.72..."
                            className="form-control"
                            rows="3"
                          />
                          <small style={{color: 'var(--text-muted)'}}>
                            Enter brainwave measurement values
                          </small>
                          <div style={{marginTop: '0.5rem'}}>
                            <input
                              type="file"
                              accept=".csv,.txt"
                              onChange={(e) => handleEegFileUpload(e.target.files[0] || null)}
                              className="form-control"
                            />
                            <small style={{color: 'var(--text-muted)'}}>
                              Optional: upload EEG machine export (CSV/TXT)
                            </small>
                          </div>
                        </div>

                        <div className="col-md-6 mb-3">
                          <label className="form-label">
                            <strong>⚡ GSR Data</strong>
                          </label>
                          <textarea
                            value={gsrData}
                            onChange={(e) => handleGsrTextChange(e.target.value)}
                            placeholder="e.g., 2.1, 2.3, 2.5, 2.4, 2.6, 2.2..."
                            className="form-control"
                            rows="3"
                          />
                          <small style={{color: 'var(--text-muted)'}}>
                            Enter skin conductance values
                          </small>
                          <div style={{marginTop: '0.5rem'}}>
                            <input
                              type="file"
                              accept=".csv,.txt"
                              onChange={(e) => handleGsrFileUpload(e.target.files[0] || null)}
                              className="form-control"
                            />
                            <small style={{color: 'var(--text-muted)'}}>
                              Optional: upload GSR export (CSV/TXT)
                            </small>
                          </div>
                        </div>
                      </div>

                      {(eegPreviewData.length > 0 || gsrPreviewData.length > 0) && (
                        <div className="row mt-2">
                          <div className="col-md-6 mb-3">
                            <div className="currentResult-panel-card">
                              <h5 className="currentResult-section-title">EEG Preview Chart</h5>
                              <div style={{ width: '100%', height: 220 }}>
                                <ResponsiveContainer>
                                  <LineChart data={eegPreviewData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(120, 120, 120, 0.2)" />
                                    <XAxis dataKey="index" tick={{ fill: 'var(--text-color)', fontSize: 11 }} />
                                    <YAxis tick={{ fill: 'var(--text-color)', fontSize: 11 }} />
                                    <Tooltip />
                                    <Legend />
                                    {eegPreviewKeys.map((key, idx) => (
                                      <Line
                                        key={key}
                                        type="monotone"
                                        dataKey={key}
                                        dot={false}
                                        strokeWidth={2}
                                        stroke={["#4f772d", "#8d9740", "#bc6c25", "#c74545", "#6a4c93"][idx % 5]}
                                      />
                                    ))}
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          </div>

                          <div className="col-md-6 mb-3">
                            <div className="currentResult-panel-card">
                              <h5 className="currentResult-section-title">GSR Preview Chart</h5>
                              <div style={{ width: '100%', height: 220 }}>
                                <ResponsiveContainer>
                                  <LineChart data={gsrPreviewData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(120, 120, 120, 0.2)" />
                                    <XAxis dataKey="index" tick={{ fill: 'var(--text-color)', fontSize: 11 }} />
                                    <YAxis tick={{ fill: 'var(--text-color)', fontSize: 11 }} />
                                    <Tooltip />
                                    <Legend />
                                    {gsrPreviewKeys.map((key, idx) => (
                                      <Line
                                        key={key}
                                        type="monotone"
                                        dataKey={key}
                                        dot={false}
                                        strokeWidth={2}
                                        stroke={["#8d9740", "#4f772d", "#bc6c25"][idx % 3]}
                                      />
                                    ))}
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="text-center mt-4" style={{ marginBottom: '1.5rem' }}>
                  <button
                    onClick={analyzeMultimodal}
                    disabled={(phase === 'analyzing' || phase === 'reanalyzing') || !serverOnline}
                    className="btn btn-neon"
                    style={{
                      padding: '0.75rem 3rem',
                      fontSize: '1.1rem',
                      marginRight: '1rem',
                      borderRadius: '30px'
                    }}
                  >
                    Analyze Stress
                  </button>
                  <button
                    onClick={clearAll}
                    className="btn btn-outline-neon"
                    style={{
                      padding: '0.75rem 2rem',
                      fontSize: '1.1rem',
                      borderRadius: '30px'
                    }}
                  >
                    Clear All
                  </button>
                </div>


              </div>
            </div>
          </div>

          {/* How to Use Section */}
          <div className="row mt-5">
            <div className="col-12">
              <div className="neon-card">
                <h3 className="text-center mb-4">How to Use the Dashboard</h3>
                <div className="row">
                  <div className="col-md-4">
                    <h4>📸 Facial Data</h4>
                    <ul className="list-unstyled">
                      <li>• Upload a clear photo of your face</li>
                      <li>• Or use webcam for live capture</li>
                      <li>• Ensure good lighting</li>
                      <li>• Look directly at camera</li>
                    </ul>
                  </div>
                  <div className="col-md-4">
                    <h4>🎤 Voice Data</h4>
                    <ul className="list-unstyled">
                      <li>• Upload a voice recording</li>
                      <li>• Speak naturally for 3-5 seconds</li>
                      <li>• Minimize background noise</li>
                      <li>• Use standard audio formats</li>
                    </ul>
                  </div>
                  <div className="col-md-4">
                    <h4>⚡ Physiological Data</h4>
                    <ul className="list-unstyled">
                      <li>• Enter comma-separated values</li>
                      <li>• EEG: Brainwave measurements</li>
                      <li>• GSR: Skin conductance values</li>
                      <li>• Use sensor device outputs</li>
                    </ul>
                  </div>
                </div>
                <div className="alert" style={{
                  background: 'var(--accent-light-bg)',
                  border: '1px solid rgba(120, 120, 120, 0.3)',
                  marginTop: '1.5rem',
                  textAlign: 'center',
                  color: 'var(--text-color)'
                }}>
                  <strong>💡 Pro Tip:</strong> For best results, provide multiple data sources. 
                  The system uses advanced multimodal fusion to combine insights from all available inputs.
                </div>
              </div>
            </div>
          </div>
          </>
          )}
        </>
      )}

      <StressChatbot
        stressLevel={currentResult?.stress_level || 'Moderate'}
        stressPercentage={currentResult ? currentResult.fused_score * 100 : null}
      />
    </div>
  );
}