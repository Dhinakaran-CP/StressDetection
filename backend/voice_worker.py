import numpy as np
import librosa
import io
import tempfile
import os

def _extract_period_with_interpolation(ac, min_lag, max_lag):
    """
    Extract sub-sample precise period from autocorrelation via parabolic interpolation.
    Solves the flat-line jitter problem caused by integer lag binning.
    """
    if max_lag >= len(ac):
        max_lag = len(ac) - 1

    # Find discrete peak
    km = np.argmax(ac[min_lag:max_lag]) + min_lag

    # Parabolic interpolation around the peak
    # Requires one sample on each side of the peak
    if min_lag < km < max_lag - 1:
        y1 = ac[km - 1]
        y2 = ac[km]
        y3 = ac[km + 1]
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-8:
            delta = 0.5 * (y1 - y3) / denom
            # Clamp delta to ±0.5 samples (parabola valid only near peak)
            delta = np.clip(delta, -0.5, 0.5)
            return float(km) + delta, float(y2)
    return float(km), float(ac[km])
def extract_f0_yin(y, sr, f0_min=75, f0_max=400, frame_len=512, hop_len=160):
    """
    YIN algorithm for F0 extraction.
    Faster than pyin, more accurate than autocorrelation.
    """
    try:
        f0_yin = librosa.yin(
            y,
            fmin=f0_min,
            fmax=f0_max,
            sr=sr,
            frame_length=frame_len,
            hop_length=hop_len,
            trough_threshold=0.1
        )
        voiced_flag = (f0_yin >= f0_min) & (f0_yin <= f0_max)
        f0_clean = f0_yin.copy()
        f0_clean[~voiced_flag] = np.nan
        return f0_clean, voiced_flag
    except Exception:
        return np.array([np.nan]), np.array([False])

def extract_voice_stress_indicators(audio_bytes, sr_target=16000, f0_min=75, f0_max=400):
    """
    Extract 12 acoustic stress biomarkers from a raw audio chunk.
    Designed for 1-3 second chunks. Fast, lightweight, generalizable.
    
    Accepts raw audio bytes (wav, webm, ogg, mp3) from Flask request.
    Returns: dict with 12 named indicators + numpy array for model input
    """
    y, sr = None, None
    EPS = 1e-10

    # Try loading directly via BytesIO
    try:
        audio_buf = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buf, sr=sr_target, mono=True, duration=3.0)
    except Exception as e:
        print(f"librosa.load direct BytesIO failed, trying scipy.io.wavfile fallback: {e}")
        try:
            from scipy.io import wavfile
            from scipy import signal
            audio_buf = io.BytesIO(audio_bytes)
            sr_orig, y_orig = wavfile.read(audio_buf)
            
            # Convert to float32 normalized to [-1.0, 1.0]
            if y_orig.dtype == np.int16:
                y_float = y_orig.astype(np.float32) / 32768.0
            elif y_orig.dtype == np.int32:
                y_float = y_orig.astype(np.float32) / 2147483648.0
            elif y_orig.dtype == np.uint8:
                y_float = (y_orig.astype(np.float32) - 128.0) / 128.0
            else:
                y_float = y_orig.astype(np.float32)
            
            # Convert stereo to mono
            if len(y_float.shape) > 1:
                y_float = np.mean(y_float, axis=1)
                
            # Resample if sample rate doesn't match target
            if sr_orig != sr_target:
                num_samples = int(len(y_float) * sr_target / sr_orig)
                y = signal.resample(y_float, num_samples)
                sr = sr_target
            else:
                y = y_float
                sr = sr_target
                
            # Truncate to maximum 3.0 seconds duration
            max_samples = int(sr * 3.0)
            if len(y) > max_samples:
                y = y[:max_samples]
                
            print("Successfully loaded WAV audio using scipy.io.wavfile fallback")
        except Exception as e_scipy:
            print(f"scipy.io.wavfile fallback failed: {e_scipy}. Trying temp WAV file fallback...")
            # Fallback 2: write to a temp file with correct .wav extension and load
            try:
                fd, temp_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                y, sr = librosa.load(temp_path, sr=sr_target, mono=True, duration=3.0)
                try:
                    os.remove(temp_path)
                except:
                    pass
            except Exception as e_inner:
                print(f"Error loading audio via tempfile: {e_inner}")
                return None

    if y is None or len(y) < sr_target * 0.5 or np.max(np.abs(y)) < 0.005:
        return None

    # Ensure loaded audio is normalized to float [-1.0, 1.0] and clamped
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / 32768.0
    y = np.clip(y, -1.0, 1.0)

    indicators = {}

    frame_len = int(sr * 0.025)  # 25ms frames
    hop_len   = int(sr * 0.010)  # 10ms hop
    
    try:
        f0_track, voiced_flag = extract_f0_yin(y, sr, f0_min, f0_max, 512, hop_len)
        f0_voiced = f0_track[~np.isnan(f0_track)]
        
        indicators['f0_mean']  = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        indicators['f0_std']   = float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 0.0
        indicators['f0_range'] = float(np.ptp(f0_voiced))  if len(f0_voiced) > 0 else 0.0
        
        if len(f0_voiced) >= 3:
            periods = sr / (f0_voiced + 1e-10)
            period_diffs = np.abs(np.diff(periods))
            jitter_rap = float(np.mean(period_diffs) / (np.mean(periods) + 1e-10)) * 100
            indicators['jitter_percent']  = float(np.clip(jitter_rap, 0.0, 5.0))
            indicators['jitter_reliable'] = bool(jitter_rap < 3.0)
        else:
            indicators['jitter_percent']  = 0.0
            indicators['jitter_reliable'] = False
            
        # Shimmer: amplitude variation between consecutive voiced frames
        rms_all = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        voiced_rms = rms_all[:len(voiced_flag)][voiced_flag]
        voiced_rms = voiced_rms[voiced_rms > 0.005]
        
        if len(voiced_rms) >= 3:
            shimmer_raw = float(np.mean(np.abs(np.diff(voiced_rms))) / (np.mean(voiced_rms) + 1e-10))
            indicators['shimmer_db'] = float(np.clip(shimmer_raw * 20, 0.0, 3.0))
        else:
            indicators['shimmer_db'] = 0.0
            
        voiced_frac = float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.5
        indicators['voiced_fraction'] = voiced_frac
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        indicators['f0_mean'] = indicators['f0_std'] = indicators['f0_range'] = 0.0
        indicators['jitter_percent'] = 0.0
        indicators['jitter_reliable'] = False
        indicators['shimmer_db'] = 0.0
        indicators['voiced_fraction'] = 0.5
        
    # 6: HNR approximation via autocorrelation
    try:
        ac_full = np.correlate(y, y, mode='full')[len(y) - 1:]
        ac_norm = ac_full / (ac_full[0] + EPS)
        min_period = int(sr / 400)
        max_period = int(sr / 80)
        if max_period < len(ac_norm):
            peak_val = np.max(ac_norm[min_period:max_period])
            hnr = 10 * np.log10(peak_val / (1 - peak_val + EPS) + EPS)
        else:
            hnr = 0.0
    except Exception:
        hnr = 0.0
    indicators['hnr'] = float(np.clip(hnr, -20, 30))

    # 7: Speaking rate proxy (ZCR)
    try:
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
        indicators['speaking_rate_proxy'] = float(np.mean(zcr))
    except Exception:
        indicators['speaking_rate_proxy'] = 0.0

    # 8: Voice intensity
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        indicators['voice_intensity'] = float(np.mean(rms))
    except Exception:
        indicators['voice_intensity'] = 0.0
        rms = np.array([0.0])

    # 9: High frequency ratio (stress elevates high-freq content)
    try:
        stft = np.abs(librosa.stft(y, n_fft=512, hop_length=hop_len))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        high_mask = freqs >= 3000
        total_energy = np.sum(stft) + EPS
        indicators['high_freq_ratio'] = float(np.sum(stft[high_mask]) / total_energy)
    except Exception:
        indicators['high_freq_ratio'] = 0.0
        stft = np.zeros((257, 1))

    # 10: Spectral flux
    try:
        spectral_flux = np.mean(np.diff(stft, axis=1) ** 2) if stft.shape[1] > 1 else 0.0
        indicators['spectral_flux'] = float(np.clip(spectral_flux, 0, 1))
    except Exception:
        indicators['spectral_flux'] = 0.0

    # 11: Pause ratio (near-silent frames)
    try:
        silence_thresh = 0.01 * np.max(np.abs(y))
        pause_frames = np.sum(rms < silence_thresh)
        indicators['pause_ratio'] = float(pause_frames / (len(rms) + EPS))
    except Exception:
        indicators['pause_ratio'] = 0.0

    # Feature vector for model (fixed order, 12 features)
    feature_vec = np.array([
        indicators['f0_mean'],
        indicators['f0_std'],
        indicators['f0_range'],
        indicators['jitter_percent'],
        indicators['shimmer_db'],
        indicators['hnr'],
        indicators['speaking_rate_proxy'],
        indicators['voice_intensity'],
        indicators['high_freq_ratio'],
        indicators['spectral_flux'],
        indicators['pause_ratio'],
        indicators['voiced_fraction'],
    ], dtype=np.float32)

    return {'indicators': indicators, 'features': feature_vec}
