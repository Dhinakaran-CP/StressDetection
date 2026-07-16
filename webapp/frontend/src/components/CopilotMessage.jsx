import React from "react";

export default function CopilotMessage({ stressLevel, explainability }) {
  const formatDriverLabel = (modality, featureIndex) => {
    const idx = Number(featureIndex);

    if (modality === "voice") {
      if (idx < 40) return "voice tone pattern";
      if (idx < 70) return "spectral balance";
      if (idx < 95) return "rhythm and tempo";
      return "voice energy consistency";
    }

    if (modality === "facial") {
      if (idx < 20) return "facial intensity distribution";
      if (idx < 44) return "facial region tension";
      if (idx < 64) return "micro-expression edges";
      return "facial texture stability";
    }

    if (modality === "physiological") {
      if (idx < 66) return "EEG arousal pattern";
      if (idx < 110) return "GSR variability";
      return "autonomic response trend";
    }

    return "stress signal feature";
  };

  const formatModalityName = (modality) => {
    if (modality === "physiological") return "Physiology";
    if (modality === "facial") return "Facial";
    if (modality === "voice") return "Voice";
    return modality;
  };

  const message =
    stressLevel === "High"
      ? "You seem a bit overwhelmed. Let's bring it down together."
      : stressLevel === "Moderate"
      ? "You are carrying moderate stress. A quick reset can help."
      : "You are in a stable state. Keep this calm rhythm going.";

  const rawDrivers = Array.isArray(explainability?.top_drivers)
    ? explainability.top_drivers
    : [];

  const topDrivers = [];
  const seenKeys = new Set();

  for (const driver of rawDrivers) {
    const prettyLabel = formatDriverLabel(driver.modality, driver.feature_index);
    const prettyModality = formatModalityName(driver.modality);
    const trendText = driver.direction === "increase" ? "raising stress" : "reducing stress";
    
    const uniqueKey = `${prettyModality}-${prettyLabel}`;
    
    if (!seenKeys.has(uniqueKey)) {
      seenKeys.add(uniqueKey);
      const absImpact = Math.min(100, Math.abs(Number(driver.shap_value || 0)) * 1000);
      topDrivers.push({
        ...driver,
        prettyLabel,
        prettyModality,
        trendText,
        impactScore: absImpact,
      });
      
      if (topDrivers.length >= 3) break;
    }
  }

  return (
    <div className="copilot-bubble slide-in-right">
      <strong>AI Insights</strong>
      <p>{message}</p>
      {explainability?.engine === "shap" && (
        <div style={{ marginTop: "0.35rem" }}>
          <small style={{ opacity: 0.9 }}>
            Explainability: SHAP {explainability?.available ? "enabled" : "unavailable"}
          </small>
        </div>
      )}

      {topDrivers.length > 0 && (
        <div style={{ marginTop: "0.5rem" }}>
          <small style={{ display: "block", marginBottom: "0.35rem" }}>Top stress drivers (interpreted)</small>
          {topDrivers.map((driver, idx) => (
            <div key={`${driver.modality}-${driver.feature_index}-${idx}`} style={{ marginBottom: "0.35rem" }}>
              <div style={{ fontSize: "0.87rem" }}>
                <strong>{driver.prettyModality}</strong>: {driver.prettyLabel} is <strong>{driver.trendText}</strong>
              </div>
              <div style={{ height: "6px", borderRadius: "999px", background: "var(--accent-light-bg)", marginTop: "0.2rem" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${driver.impactScore.toFixed(1)}%`,
                    borderRadius: "999px",
                    background: driver.direction === "increase" ? "#c74545" : "var(--primary-color)",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {explainability?.message && (
        <div style={{ marginTop: "0.35rem", fontSize: "0.85rem" }}>
          {explainability.message}
        </div>
      )}
    </div>
  );
}
