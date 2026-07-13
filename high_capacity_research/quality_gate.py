"""
quality_gate.py
---------------
Implements §5.2 ModalityQualityGate and §5.3 VerifiedBaselineModule
as defined in high_arc5.md.
"""

import numpy as np
import pandas as pd


# =========================================================
# §5.2 — Modality Quality Gate
# =========================================================
class ModalityQualityGate:
    """
    Computes a per-window quality score [0, 1] for each modality.
    Scores below their thresholds are flagged as rejected.

    Face rules:
      - Low-visibility (frozen): eye-aspect-ratio std < EYE_STD_THRESH  → score 0.0
      - Good motion present                                               → score 1.0

    Voice rules:
      - Clipped (RMS energy > CLIP_THRESH)                               → score 0.0
      - Silent  (RMS energy < SILENT_THRESH)                             → score 0.0
      - Otherwise                                                         → score 1.0

    Physio rules:
      - NaN fraction > NAN_FRAC_THRESH                                   → score 0.0
      - Flat HR (std < HR_STD_THRESH)                                    → score 0.0
      - Otherwise                                                         → score 1.0
    """

    # Quality thresholds (tunable)
    EYE_STD_THRESH   = 0.005   # below → frozen face / occluded
    CLIP_THRESH      = 0.98    # above → audio clipped
    SILENT_THRESH    = 0.01    # below → silent window
    NAN_FRAC_THRESH  = 0.20    # above → too many missing physio values
    HR_STD_THRESH    = 0.001   # below → flat HR signal

    # Feature column names expected in input DataFrames
    EYE_COLS    = ['eye_aspect_ratio_mean']           # representative eye feature
    RMS_COLS    = ['rms_energy_mean']                 # voice energy
    HR_COLS     = ['hr_mean']                         # physio heart-rate

    def score_face(self, face_window: pd.DataFrame) -> float:
        """
        Parameters
        ----------
        face_window : pd.DataFrame, shape (seq_len, n_face_features)

        Returns
        -------
        quality : float in [0, 1]
        """
        available = [c for c in self.EYE_COLS if c in face_window.columns]
        if not available:
            return 1.0  # cannot assess, pass through

        eye_std = face_window[available].std().mean()
        if eye_std < self.EYE_STD_THRESH:
            return 0.0
        return 1.0

    def score_voice(self, voice_window: pd.DataFrame) -> float:
        """
        Parameters
        ----------
        voice_window : pd.DataFrame, shape (seq_len, n_voice_features)
        """
        available = [c for c in self.RMS_COLS if c in voice_window.columns]
        if not available:
            return 1.0

        rms = voice_window[available].mean().mean()
        if rms > self.CLIP_THRESH:
            return 0.0
        if rms < self.SILENT_THRESH:
            return 0.0
        return 1.0

    def score_physio(self, physio_window: pd.DataFrame) -> float:
        """
        Parameters
        ----------
        physio_window : pd.DataFrame, shape (seq_len, n_physio_features)
        """
        # NaN check
        nan_frac = physio_window.isna().mean().mean()
        if nan_frac > self.NAN_FRAC_THRESH:
            return 0.0

        available_hr = [c for c in self.HR_COLS if c in physio_window.columns]
        if available_hr:
            hr_std = physio_window[available_hr].std().mean()
            if hr_std < self.HR_STD_THRESH:
                return 0.0

        return 1.0

    def score_all(
        self,
        face_window: pd.DataFrame,
        voice_window: pd.DataFrame,
        physio_window: pd.DataFrame,
    ) -> dict:
        """
        Returns quality scores for all three modalities.

        Returns
        -------
        dict with keys: 'face', 'voice', 'physio'
            Each value is a float in [0, 1].
        """
        return {
            "face":   self.score_face(face_window),
            "voice":  self.score_voice(voice_window),
            "physio": self.score_physio(physio_window),
        }

    def quality_tensor(self, quality_scores: dict, device=None):
        """
        Convert quality_scores dict to a float tensor [3] suitable for
        being used as a modality mask in the attention fusion layer.

        Parameters
        ----------
        quality_scores : dict   {"face": f, "voice": v, "physio": p}
        device         : torch.device or None

        Returns
        -------
        torch.FloatTensor shape [3] — [face_q, voice_q, physio_q]
        """
        import torch
        t = torch.FloatTensor([
            quality_scores["face"],
            quality_scores["voice"],
            quality_scores["physio"],
        ])
        if device is not None:
            t = t.to(device)
        return t


# =========================================================
# §5.3 — Verified Baseline Module
# =========================================================
class VerifiedBaselineModule:
    """
    Validates that a user's collected "calm" baseline is actually calm
    before accepting it as a reference for baseline-relative normalization.

    Uses a lightweight threshold rule on a pre-fitted fold-level
    probability estimator (or heuristic if no model is available).

    Stores per-user baseline statistics:
        mean, std, min, max, baseline_confidence_label

    Usage
    -----
        vbm = VerifiedBaselineModule()
        report = vbm.validate_and_store(subject_id, calm_df, feature_cols)
        if report["contaminated"]:
            # request re-calibration
    """

    CONTAMINATION_THRESH = 0.40   # if >40% of baseline windows look stressed → contaminated
    MIN_CALM_WINDOWS     = 3      # minimum windows required to build a valid baseline

    def __init__(self):
        self.baselines: dict = {}   # subject_id → baseline stats dict

    def validate_and_store(
        self,
        subject_id: str,
        calm_df: pd.DataFrame,
        feature_cols: list,
        stress_proba_fn=None,
    ) -> dict:
        """
        Validate the calm baseline for one subject.

        Parameters
        ----------
        subject_id       : str
        calm_df          : pd.DataFrame  — windows labelled 0 (calm) for this subject
        feature_cols     : list[str]     — feature columns to compute stats on
        stress_proba_fn  : callable(df) -> np.ndarray of shape (n,)
                           Optional. If provided, used to estimate stress probability
                           in the supposedly-calm windows. When None, heuristic
                           variance-based check is used instead.

        Returns
        -------
        dict with keys:
            subject_id, mean, std, min, max,
            baseline_confidence_label, contaminated, n_windows
        """
        avail_cols = [c for c in feature_cols if c in calm_df.columns]
        if len(calm_df) < self.MIN_CALM_WINDOWS or not avail_cols:
            report = {
                "subject_id": subject_id,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "baseline_confidence_label": "insufficient_data",
                "contaminated": True,
                "n_windows": len(calm_df),
            }
            self.baselines[subject_id] = report
            return report

        data = calm_df[avail_cols].fillna(0).values   # (n_windows, n_feats)

        # --- Baseline statistics ---
        mean_vals = data.mean(axis=0)
        std_vals  = data.std(axis=0)
        min_vals  = data.min(axis=0)
        max_vals  = data.max(axis=0)

        # --- Contamination check ---
        if stress_proba_fn is not None:
            # Use provided classifier to estimate stress prob in calm windows
            stress_prob = stress_proba_fn(calm_df[avail_cols].fillna(0))
            contamination_rate = float((stress_prob >= 0.5).mean())
        else:
            # Heuristic: high z-score variance across features suggests stress
            # Compare each window to the mean; flag if deviation is very high
            deviations = np.abs(data - mean_vals) / (std_vals + 1e-8)
            contamination_rate = float((deviations.mean(axis=1) > 2.5).mean())

        contaminated = contamination_rate > self.CONTAMINATION_THRESH

        if contaminated:
            confidence_label = "contaminated"
        elif contamination_rate > 0.20:
            confidence_label = "uncertain"
        else:
            confidence_label = "verified_calm"

        report = {
            "subject_id":                 subject_id,
            "mean":                       mean_vals.tolist(),
            "std":                        std_vals.tolist(),
            "min":                        min_vals.tolist(),
            "max":                        max_vals.tolist(),
            "baseline_confidence_label":  confidence_label,
            "contaminated":               contaminated,
            "contamination_rate":         contamination_rate,
            "n_windows":                  len(calm_df),
        }
        self.baselines[subject_id] = report
        return report

    def get_baseline(self, subject_id: str) -> dict:
        """Retrieve stored baseline stats for a subject."""
        return self.baselines.get(subject_id, None)

    def summarize(self) -> pd.DataFrame:
        """Return a DataFrame summarising all stored baselines."""
        rows = []
        for sid, b in self.baselines.items():
            rows.append({
                "subject_id":                sid,
                "n_windows":                 b["n_windows"],
                "baseline_confidence_label": b["baseline_confidence_label"],
                "contaminated":              b["contaminated"],
                "contamination_rate":        b.get("contamination_rate", None),
            })
        return pd.DataFrame(rows)
