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

function computeAcousticIndicators(samples, sampleRate) {
  // 1. Voice Intensity (RMS)
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    sumSq += samples[i] * samples[i];
  }
  const rms = Math.sqrt(sumSq / samples.length);
  
  // If too silent, return silence indicators
  if (rms < 0.003) {
    return {
      f0_mean: 0.0,
      jitter_percent: 0.0,
      shimmer_db: 0.0,
      speaking_rate_proxy: 0.0,
      voice_intensity: rms,
      jitter_reliable: true
    };
  }

  // 2. Zero Crossing Rate (ZCR)
  let zeroCrossings = 0;
  for (let i = 1; i < samples.length; i++) {
    if ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0)) {
      zeroCrossings++;
    }
  }
  const zcr = zeroCrossings / samples.length;

  // 3. Pitch Tracking via Autocorrelation (YIN-like discrete autocorrelation)
  // Pitch limits: 75Hz - 300Hz
  const minLag = Math.floor(sampleRate / 300); // ~53 samples at 16kHz
  const maxLag = Math.floor(sampleRate / 75);  // ~213 samples at 16kHz
  
  // To compute F0, we can analyze in frames of 512 samples with 50% overlap
  const frameSize = 512;
  const hopSize = 256;
  const pitches = [];
  const peakAmplitudes = [];
  
  for (let offset = 0; offset + frameSize <= samples.length; offset += hopSize) {
    const frame = samples.slice(offset, offset + frameSize);
    
    // Find local max amplitude
    let localMax = 0.0001;
    for (let i = 0; i < frameSize; i++) {
      const absVal = Math.abs(frame[i]);
      if (absVal > localMax) localMax = absVal;
    }
    peakAmplitudes.push(localMax);
    
    // Autocorrelation for pitch
    let bestLag = -1;
    let bestR = -Infinity;
    
    for (let lag = minLag; lag <= maxLag; lag++) {
      let r = 0;
      for (let i = 0; i < frameSize - lag; i++) {
        r += frame[i] * frame[i + lag];
      }
      if (r > bestR) {
        bestR = r;
        bestLag = lag;
      }
    }
    
    // Verify peak strength to reject voice silence
    let energy = 0;
    for (let i = 0; i < frameSize; i++) {
      energy += frame[i] * frame[i];
    }
    
    if (bestLag > 0 && energy > 0.01) {
      const pitch = sampleRate / bestLag;
      if (pitch >= 75 && pitch <= 300) {
        pitches.push(pitch);
      }
    }
  }
  
  // Calculate average pitch (F0 Mean)
  let f0_mean = 0.0;
  const validPitches = pitches.filter(p => p > 0);
  if (validPitches.length > 0) {
    const sum = validPitches.reduce((a, b) => a + b, 0);
    f0_mean = sum / validPitches.length;
  }
  
  // 4. Jitter (Micro-instability)
  let jitter_percent = 0.0;
  if (validPitches.length > 2) {
    const smoothedPitches = [];
    for (let i = 0; i < validPitches.length; i++) {
      if (i === 0 || i === validPitches.length - 1) {
        smoothedPitches.push(validPitches[i]);
      } else {
        const window = [validPitches[i-1], validPitches[i], validPitches[i+1]].sort((a,b)=>a-b);
        smoothedPitches.push(window[1]);
      }
    }
    let diffSum = 0;
    const periods = smoothedPitches.map(p => 1.0 / p);
    for (let i = 1; i < periods.length; i++) {
      diffSum += Math.abs(periods[i] - periods[i - 1]);
    }
    const meanPeriod = periods.reduce((a, b) => a + b, 0) / periods.length;
    if (meanPeriod > 0) {
      jitter_percent = (diffSum / (periods.length - 1)) / meanPeriod * 100.0;
    }
  }
  
  // 5. Shimmer (Amplitude variation)
  let shimmer_db = 0.0;
  if (peakAmplitudes.length > 2) {
    const smoothedAmps = [];
    for (let i = 0; i < peakAmplitudes.length; i++) {
      if (i === 0 || i === peakAmplitudes.length - 1) {
        smoothedAmps.push(peakAmplitudes[i]);
      } else {
        const window = [peakAmplitudes[i-1], peakAmplitudes[i], peakAmplitudes[i+1]].sort((a,b)=>a-b);
        smoothedAmps.push(window[1]);
      }
    }
    let shimmerSum = 0;
    for (let i = 1; i < smoothedAmps.length; i++) {
      const ratio = smoothedAmps[i] / (smoothedAmps[i - 1] || 0.0001);
      shimmerSum += Math.abs(20.0 * Math.log10(ratio));
    }
    shimmer_db = shimmerSum / (smoothedAmps.length - 1);
  }
  
  // Limit values to reasonable bounds for display scaling
  jitter_percent = Math.min(jitter_percent, 15.0);
  shimmer_db = Math.min(shimmer_db, 10.0);
  
  return {
    f0_mean: f0_mean,
    jitter_percent: jitter_percent,
    shimmer_db: shimmer_db,
    speaking_rate_proxy: zcr,
    voice_intensity: rms * 2.0, // Scale for display range
    jitter_reliable: rms > 0.005
  };
}

export default function WaveformRecorder({ continuous, chunkIntervalMs = 2000, onChunk, voiceScore = null, onIndicatorsUpdate = null }) {
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
        
        // Compute indicators on current chunk (256ms) and emit for local zero-latency UI updates
        if (onIndicatorsUpdate) {
          const indicators = computeAcousticIndicators(inputData, 16000);
          onIndicatorsUpdate(indicators);
        }
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
