"""
session_state.py
Phase 7: Per-session state container for the RuntimeEngine.

Replaces the bare `self.sessions = {}` dict in realtime_core.py with a
proper, testable class that owns audio buffering, staleness tracking, and
score-buffer handles.
"""

import time
import numpy as np


class SessionState:
    """
    Encapsulates all mutable state owned by a single user streaming session.

    Responsibilities:
    - Audio ring-buffer (last N seconds of audio)
    - Staleness tracking (how long since last activity)
    - Reference to shared ScoreBuffer for this session

    Not responsible for prediction or feature extraction.
    """

    # ── Defaults (overridable per instance) ──────────────────────────────────
    DEFAULT_MAX_AUDIO_SECONDS   = 3.0   # rolling window kept in memory
    DEFAULT_MIN_AUDIO_SECONDS   = 0.5   # minimum before audio predict is attempted
    DEFAULT_SAMPLE_RATE         = 44100  # Hz

    def __init__(
        self,
        session_id: str,
        score_buffer=None,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
        min_audio_seconds: float = DEFAULT_MIN_AUDIO_SECONDS,
        sample_rate: int          = DEFAULT_SAMPLE_RATE,
    ):
        self.session_id    = session_id
        self.score_buffer  = score_buffer  # shared ScoreBuffer singleton or None
        self.max_audio_s   = max_audio_seconds
        self.min_audio_s   = min_audio_seconds
        self.sample_rate   = sample_rate

        self._audio: np.ndarray = np.array([], dtype=np.float32)
        self._created_at: float = time.time()
        self._last_activity: float = time.time()

    # ── Audio buffering ───────────────────────────────────────────────────────

    def buffer_audio(self, chunk: np.ndarray, sample_rate: int = None) -> np.ndarray | None:
        """
        Append `chunk` to the rolling audio buffer.

        Returns the buffered audio if >= MIN threshold, else None
        (signalling the caller to wait for more data).
        """
        sr = sample_rate or self.sample_rate
        chunk = np.array(chunk, dtype=np.float32).flatten()

        self._audio = np.concatenate((self._audio, chunk))
        self._last_activity = time.time()

        max_samples = int(self.max_audio_s * sr)
        if len(self._audio) > max_samples:
            self._audio = self._audio[-max_samples:]

        min_samples = int(self.min_audio_s * sr)
        if len(self._audio) < min_samples:
            return None  # still buffering

        return self._audio.copy()

    @property
    def audio_duration_seconds(self) -> float:
        return len(self._audio) / self.sample_rate

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self):
        """Clear audio buffer and reset activity timer. Keeps session alive."""
        self._audio = np.array([], dtype=np.float32)
        self._last_activity = time.time()
        if self.score_buffer is not None:
            self.score_buffer.clear()

    def age_seconds(self) -> float:
        """Total lifetime of this session in seconds."""
        return time.time() - self._created_at

    def idle_seconds(self) -> float:
        """Seconds since last audio or video activity."""
        return time.time() - self._last_activity

    def touch(self):
        """Update last-activity timestamp (call on video frames, etc.)."""
        self._last_activity = time.time()

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"SessionState(id={self.session_id!r}, "
            f"audio={self.audio_duration_seconds:.2f}s, "
            f"idle={self.idle_seconds():.1f}s)"
        )
