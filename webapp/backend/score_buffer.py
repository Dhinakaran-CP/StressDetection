import threading
import time

class ScoreBuffer:
    """
    Thread-safe buffer holding the latest stress score per modality.
    Each modality writes independently. Fusion reads from all simultaneously.
    Scores older than STALE_THRESHOLD_S are excluded from fusion.
    """
    STALE_THRESHOLD_S = 15  # seconds before a score is considered stale
    EMA_ALPHA = 0.3  # weight given to new reading vs history

    def __init__(self):
        self._lock  = threading.Lock()
        self._store = {}  # modality -> {score, ema_score, indicators, timestamp}
        self._ema   = {}  # modality -> current EMA state

    def write(self, modality: str, score: float, indicators: dict = None):
        with self._lock:
            # Compute EMA
            prev_ema = self._ema.get(modality, score)
            new_ema  = self.EMA_ALPHA * score + (1 - self.EMA_ALPHA) * prev_ema
            self._ema[modality] = new_ema

            self._store[modality] = {
                'score':      score,
                'ema_score':  new_ema,
                'indicators': indicators or {},
                'timestamp':  time.time(),
            }

    def read(self, modality: str):
        with self._lock:
            entry = self._store.get(modality)
            if entry is None:
                return None
            if time.time() - entry['timestamp'] > self.STALE_THRESHOLD_S:
                return None  # stale
            return entry

    def read_all(self):
        now = time.time()
        with self._lock:
            return {
                k: v for k, v in self._store.items()
                if now - v['timestamp'] <= self.STALE_THRESHOLD_S
            }

    def clear(self, modality: str = None):
        with self._lock:
            if modality:
                self._store.pop(modality, None)
                self._ema.pop(modality, None)
            else:
                self._store.clear()
                self._ema.clear()

# Singleton instance shared across all Flask routes
score_buffer = ScoreBuffer()
