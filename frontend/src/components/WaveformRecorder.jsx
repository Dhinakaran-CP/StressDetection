import React, { useEffect, useRef, useState } from 'react';

// Standard PCM WAV encoder helper
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

export default function WaveformRecorder({ continuous, chunkIntervalMs = 2000, onChunk, voiceScore = null }) {
  const [recording, setRecording] = useState(false);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const audioChunksRef = useRef([]);

  const voiceScoreRef = useRef(voiceScore);

  useEffect(() => {
    voiceScoreRef.current = voiceScore;
  }, [voiceScore]);

  useEffect(() => {
    if (continuous) {
      startRecording();
    } else {
      stopRecording();
    }

    return () => {
      stopRecording();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [continuous]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      streamRef.current = stream;
      setRecording(true);

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      // Initialize AudioContext at 16kHz so it resamples automatically at the Web Audio API level
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      audioContextRef.current = audioCtx;

      // 1. Setup Analyser for Visualization
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;
      source.connect(analyser);

      // 2. Setup ScriptProcessor to collect PCM samples
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      source.connect(processor);
      processor.connect(audioCtx.destination);

      audioChunksRef.current = [];

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Push copy of samples to our buffer
        audioChunksRef.current.push(...inputData);
      };

      // 3. Regularly encode to WAV and send to parent (sliding window approach)
      intervalRef.current = setInterval(() => {
        const samples = audioChunksRef.current;
        if (samples.length >= audioCtx.sampleRate * 0.5) {
          // Keep last 2 seconds (32000 samples at 16kHz) for sliding window analysis
          const windowSize = audioCtx.sampleRate * 2;
          const sliceStart = Math.max(0, samples.length - windowSize);
          const windowSamples = samples.slice(sliceStart);
          
          // Truncate the original buffer to free up memory and keep the window
          audioChunksRef.current = windowSamples;

          // Encode windowSamples to WAV Blob
          const wavBlob = bufferToWav(windowSamples, audioCtx.sampleRate);
          
          // Emit chunk
          onChunk(wavBlob);
        }
      }, chunkIntervalMs);

      drawWaveform();
    } catch (err) {
      console.error("Error accessing microphone for waveform recorder:", err);
    }
  };

  const stopRecording = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (e) {}
      audioContextRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    setRecording(false);
    clearCanvas();
  };
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const styles = getComputedStyle(canvas);
    const bgColor = styles.getPropertyValue('--chat-bg').trim() || '#050510';
    const gridColor = styles.getPropertyValue('--glass-border').trim() || 'rgba(0, 242, 255, 0.2)';
    
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Draw flat center line
    ctx.lineWidth = 2;
    ctx.strokeStyle = gridColor;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!analyserRef.current) return;
      animationFrameRef.current = requestAnimationFrame(draw);

      analyser.getByteTimeDomainData(dataArray);

      const styles = getComputedStyle(canvas);
      const bgColor = styles.getPropertyValue('--chat-bg').trim() || '#050510';
      const themePrimary = styles.getPropertyValue('--primary-color').trim() || '#00f2ff';

      ctx.fillStyle = bgColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      let visualColor = themePrimary;
      const currentScore = voiceScoreRef.current;
      if (currentScore !== null) {
        if (currentScore > 0.7) {
          visualColor = '#F44336'; // Red for High
        } else if (currentScore > 0.4) {
          visualColor = '#FF9800'; // Orange for Moderate
        }
      }

      ctx.lineWidth = 3;
      // Glowing line effect
      ctx.shadowBlur = 8;
      ctx.shadowColor = visualColor;
      ctx.strokeStyle = visualColor;

      ctx.beginPath();
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0; // Normalized amplitude around 1.0
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
      ctx.shadowBlur = 0; // reset blur
    };

    draw();
  };

  const currentScore = voiceScore;
  const statusColor = recording 
    ? (currentScore > 0.7 ? '#F44336' : currentScore > 0.4 ? '#FF9800' : 'var(--primary-color)')
    : 'var(--text-muted)';

  return (
    <div style={{ border: 'var(--glass-border)', borderRadius: 12, overflow: 'hidden', padding: 12, background: 'var(--card-bg)', boxShadow: 'var(--glass-shadow)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>Vocal Audio Monitor</span>
        <span style={{ fontSize: '0.75rem', color: statusColor, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ 
            width: 8, 
            height: 8, 
            borderRadius: '50%', 
            background: statusColor, 
            display: 'inline-block', 
            boxShadow: recording ? `0 0 8px ${statusColor === 'var(--primary-color)' ? 'var(--primary-color)' : statusColor}` : 'none' 
          }}></span>
          {recording ? 'STREAMING' : 'STANDBY'}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={320}
        height={100}
        style={{ width: '100%', height: 100, display: 'block', background: 'var(--chat-bg)', borderRadius: 8 }}
      />
    </div>
  );
}
