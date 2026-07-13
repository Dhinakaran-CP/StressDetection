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

const useCountUp = (target, duration = 1000) => {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let frame;
    const start = performance.now();
    const safeTarget = Number(target) || 0;

    const tick = (now) => {
      const progress = clamp((now - start) / duration, 0, 1);
      setValue(Math.round(safeTarget * progress));
      if (progress < 1) {
        frame = requestAnimationFrame(tick);
      }
    };

    frame = requestAnimationFrame(tick);
    return () => {
      if (frame) cancelAnimationFrame(frame);
    };
  }, [target, duration]);

  return value;
};

export default function InsightCards({ result, recoveryScore, stressLevel }) {
  const confidenceTarget = Math.round(toPercent(result?.confidence, 91));
  const animatedRecovery = useCountUp(recoveryScore, 1200);
  const animatedConfidence = useCountUp(confidenceTarget, 900);

  const triggerText = useMemo(() => {
    if (stressLevel === "High") return "Screen fatigue";
    if (stressLevel === "Moderate") return "Task switching";
    return "Sustained focus load";
  }, [stressLevel]);

  return (
    <div className="result-panel-card insight-cards-grid insights-slide">
      <div className="insight-card">
        <small>Recovery Score</small>
        <div className="insight-value">{animatedRecovery}%</div>
      </div>

      <div className="insight-card">
        <small>Confidence Score</small>
        <div className="insight-value">{animatedConfidence}%</div>
      </div>

      <div className="insight-card insight-wide">
        <small>Detected Trigger</small>
        <div className="insight-trigger">{triggerText}</div>
      </div>
    </div>
  );
}
