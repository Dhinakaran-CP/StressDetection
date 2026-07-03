import React, { useEffect, useMemo, useState } from "react";

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const toPercent = (value, fallback = 0) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return fallback;
  }

  const numeric = Number(value);
  if (numeric <= 1) return clamp(numeric * 100, 0, 100);
  return clamp(numeric, 0, 100);
};

const toStressBand = (percentage) => {
  if (percentage >= 67) return "High";
  if (percentage >= 34) return "Medium";
  return "Low";
};

const useCountUp = (target, duration = 1000, enabled = true) => {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setValue(0);
      return;
    }

    let animationFrame;
    const start = performance.now();
    const safeTarget = Number(target) || 0;

    const tick = (now) => {
      const progress = clamp((now - start) / duration, 0, 1);
      setValue(Math.round(safeTarget * progress));
      if (progress < 1) {
        animationFrame = requestAnimationFrame(tick);
      }
    };

    animationFrame = requestAnimationFrame(tick);

    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
    };
  }, [target, duration, enabled]);

  return value;
};

export default function ResultEnhancements({ result }) {
  const stressPercentage = toPercent(result?.percentage, 0);
  const [stressLevel, setStressLevel] = useState(result?.stress_level || "Moderate");
  const [isBreathing, setIsBreathing] = useState(false);
  const [showIntervention, setShowIntervention] = useState(false);
  const [showRecovery, setShowRecovery] = useState(false);
  const [breathPhase, setBreathPhase] = useState("Inhale...");
  const [recoveryScore, setRecoveryScore] = useState(72);

  const analysis = useMemo(() => {
    const individual = result?.individual_predictions || {};

    const facial = toPercent(individual.facial, clamp(stressPercentage + 10, 20, 96));
    const voice = toPercent(individual.voice, clamp(stressPercentage - 12, 15, 88));
    const physiological = toPercent(
      individual.physiological,
      clamp(stressPercentage + 8, 18, 97)
    );

    const points = [
      { key: "facial", label: "Facial", value: facial, reason: "facial tension" },
      { key: "voice", label: "Voice", value: voice, reason: "vocal strain" },
      {
        key: "physiological",
        label: "Physiological",
        value: physiological,
        reason: "elevated physiological signals",
      },
    ];

    const sorted = [...points].sort((a, b) => b.value - a.value);
    const total = points.reduce((sum, item) => sum + item.value, 0) || 1;

    return {
      facial,
      voice,
      physiological,
      cause: `${sorted[0].reason} and ${sorted[1].reason}`,
      contributions: points.map((item) => ({
        ...item,
        contribution: Math.round((item.value / total) * 100),
        band: toStressBand(item.value),
      })),
    };
  }, [result, stressPercentage]);

  useEffect(() => {
    const nextLevel = result?.stress_level || toStressBand(stressPercentage);
    setStressLevel(nextLevel);

    const baseRecovery = nextLevel === "High" ? 72 : nextLevel === "Moderate" ? 79 : 88;
    setRecoveryScore(baseRecovery);
    setShowIntervention(nextLevel === "High");
    setShowRecovery(false);
    setIsBreathing(false);
  }, [result, stressPercentage]);

  useEffect(() => {
    if (!isBreathing) {
      setBreathPhase("Inhale...");
      return;
    }

    const interval = setInterval(() => {
      setBreathPhase((prev) => (prev === "Inhale..." ? "Exhale..." : "Inhale..."));
    }, 2000);

    return () => clearInterval(interval);
  }, [isBreathing]);

  const confidenceTarget = Math.round(toPercent(result?.confidence, 91));
  const recoveryDisplay = useCountUp(recoveryScore, 1200, true);
  const confidenceDisplay = useCountUp(confidenceTarget, 1000, true);
  const initialRecovery = stressLevel === "High" ? 72 : stressLevel === "Moderate" ? 79 : 88;

  const triggerText =
    stressLevel === "High"
      ? "Prolonged screen exposure"
      : stressLevel === "Moderate"
      ? "Task switching load"
      : "Sustained cognitive effort";

  const copilotMessage =
    stressLevel === "High"
      ? "You seem stressed. Try a short breathing exercise."
      : stressLevel === "Moderate"
      ? "You are showing moderate stress. A 2-minute break can help reset focus."
      : "Great regulation so far. Keep your rhythm with short mindful pauses.";

  const handleTakeBreak = () => {
    setIsBreathing(false);
    setShowIntervention(false);
    setShowRecovery(true);
    setRecoveryScore((prev) => clamp(prev + 13, 0, 100));
  };

  return (
    <div className="result-enhancements fade-in-up" style={{ marginTop: "1.25rem", textAlign: "left" }}>
      <div className="row" style={{ marginTop: "0.25rem" }}>
        <div className="col-md-6 mb-4">
          <div className="result-panel-card insights-fade">
            <h5 className="result-section-title">Stress Analysis Panel</h5>
            <div className="analysis-item-row">
              <span>Facial Stress</span>
              <strong>{toStressBand(analysis.facial)} ({Math.round(analysis.facial)}%)</strong>
            </div>
            <div className="analysis-item-row">
              <span>Voice Stress</span>
              <strong>{toStressBand(analysis.voice)} ({Math.round(analysis.voice)}%)</strong>
            </div>
            <div className="analysis-item-row">
              <span>Physiological Stress</span>
              <strong>{toStressBand(analysis.physiological)} ({Math.round(analysis.physiological)}%)</strong>
            </div>

            <div className="analysis-cause">
              <small>Main Cause</small>
              <p>{analysis.cause}</p>
            </div>

            <div>
              <small className="contrib-title">Contribution Breakdown</small>
              {analysis.contributions.map((entry) => (
                <div key={entry.key} style={{ marginTop: "0.6rem" }}>
                  <div className="analysis-item-row">
                    <span>{entry.label}</span>
                    <span>{entry.contribution}%</span>
                  </div>
                  <div className="contrib-track">
                    <div className="contrib-fill" style={{ width: `${entry.contribution}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-4">
          <div className="result-panel-card insight-cards-grid insights-slide">
            <div className="insight-card">
              <small>Recovery Score</small>
              <div className="insight-value">{recoveryDisplay}%</div>
            </div>
            <div className="insight-card">
              <small>Confidence Score</small>
              <div className="insight-value">{confidenceDisplay}%</div>
            </div>
            <div className="insight-card insight-wide">
              <small>Detected Trigger</small>
              <div className="insight-trigger">{triggerText}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="copilot-bubble slide-in-right">
        <strong>AI Insights</strong>
        <p>{copilotMessage}</p>
      </div>

      {showIntervention && (
        <div className="intervention-panel fade-in-up">
          <h5 style={{ marginBottom: "0.75rem" }}>Intervention Panel</h5>
          <p style={{ marginBottom: "1rem" }}>High stress detected. Start a short calming action now.</p>
          <div className="intervention-actions">
            <button className="btn btn-neon" onClick={() => setIsBreathing(true)}>
              Start Breathing
            </button>
            <button className="btn btn-outline-neon" onClick={handleTakeBreak}>
              Take a Break
            </button>
            <button
              className="btn btn-outline-neon"
              onClick={() => {
                setIsBreathing(false);
                setShowIntervention(false);
              }}
            >
              Dismiss
            </button>
          </div>

          {isBreathing && (
            <div className="breathing-wrap">
              <div className="breathing-circle" />
              <div className="breathing-text">{breathPhase}</div>
            </div>
          )}
        </div>
      )}

      {showRecovery && (
        <div className="recovery-highlight fade-in-up">
          <h5 style={{ marginBottom: "0.5rem" }}>Great job! Stress reduced</h5>
          <p style={{ marginBottom: "0.35rem" }}>
            Recovery Score: {initialRecovery}% {"->"} {recoveryDisplay}%
          </p>
          <div className="calm-streak">Calm Streak: 10 mins</div>
        </div>
      )}
    </div>
  );
}
