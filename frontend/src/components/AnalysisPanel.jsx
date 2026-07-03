import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';


const MODALITY_CONFIG = {
  facial:   { icon: '👁',  label: 'Facial Expression', color: '#3182ce' },
  face:     { icon: '👁',  label: 'Facial Expression', color: '#3182ce' },
  voice:    { icon: '🎙',  label: 'Vocal Strain',      color: '#805ad5' },
  physiological: { icon: '📈',  label: 'Physiological',     color: '#319795' },
  physio:   { icon: '📈',  label: 'Physiological',     color: '#319795' },
};

const FEATURE_LABELS = {
  avg_ear:           'Eye Openness',
  brow_descent_left: 'Brow Tension',
  lip_compression:   'Lip Compression',
  jaw_displacement:  'Jaw Tension',
  jitter_percent:    'Vocal Jitter',
  f0_mean:           'Pitch Level',
  hnr:               'Voice Clarity',
  shimmer_db:        'Amplitude Stability',
  alpha_power:       'Alpha Brainwave',
  beta_power:        'Beta Brainwave',
  scr_rate:          'Skin Conductance',
};

const FACIAL_FEATURE_NAMES = [
  "Left Eye Openness",          // 0
  "Right Eye Openness",         // 1
  "Average Eye Openness",       // 2
  "Blink Velocity",             // 3
  "Brow Tension (Left)",        // 4
  "Brow Tension (Right)",       // 5
  "Brow Asymmetry",             // 6
  "Lip Compression",            // 7
  "Jaw Tension",                // 8
  "Mouth Corner Pull",          // 9
  "Forehead Tension",           // 10
  "Normalized Face Height",     // 11
  "Head Tilt",                  // 12
  "Temporal X Variation",       // 13
  "Temporal Y Variation",       // 14
  "Eye Openness Ratio",         // 15
  "Landmark Confidence",        // 16
  "Nose Wrinkle"                // 17
];

const VOICE_FEATURE_NAMES = [
  "Mean Pitch",                 // 0
  "Pitch Standard Deviation",   // 1
  "Pitch Range",                // 2
  "Vocal Jitter",               // 3
  "Amplitude Shimmer",          // 4
  "Voice Harmonics-to-Noise Ratio", // 5
  "Speaking Rate",              // 6
  "Voice Intensity",            // 7
  "High Frequency Energy Ratio", // 8
  "Spectral Flux",              // 9
  "Pause Ratio",                // 10
  "Voiced Fraction"             // 11
];

function featureLabel(key) {
  if (!key) return "Unknown Feature";
  const normalizedKey = String(key).toLowerCase();
  
  if (normalizedKey.startsWith("facial_")) {
    const idx = parseInt(normalizedKey.split("_")[1]);
    if (idx >= 0 && idx < FACIAL_FEATURE_NAMES.length) {
      return FACIAL_FEATURE_NAMES[idx];
    }
  }
  
  if (normalizedKey.startsWith("voice_")) {
    const idx = parseInt(normalizedKey.split("_")[1]);
    if (idx >= 0 && idx < VOICE_FEATURE_NAMES.length) {
      return VOICE_FEATURE_NAMES[idx];
    }
  }

  if (normalizedKey.startsWith("phys_") || normalizedKey.startsWith("physiological_")) {
    const parts = normalizedKey.split("_");
    const idx = parseInt(parts[parts.length - 1]);
    if (idx >= 42) {
      return `Skin Conductance / GSR Response (Feature ${idx - 41})`;
    }
    return `EEG Brainwave Band Power (Feature ${idx})`;
  }

  return FEATURE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

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

// ── Sub-components ────────────────────────────────────────────────────────────

function SHAPDrivers({ explainability }) {
  if (!explainability?.available || !explainability.top_drivers?.length) return null;

  // Calculate total absolute SHAP value for normalization
  const totalShap = explainability.top_drivers.reduce((sum, d) => sum + Math.abs(d.shap_value), 0) || 1.0;

  return (
    <div style={{
      background: 'var(--card-inner-bg)',
      border: 'var(--glass-border)',
      borderRadius: 10, padding: '14px 16px', marginTop: 16,
    }}>
      <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem',
                    letterSpacing: '0.1em', marginBottom: 12, fontWeight: 700 }}>
        🧠 WHY THIS PREDICTION (TOP DRIVERS - RELATIVE IMPACT)
      </div>
      {explainability.top_drivers.slice(0, 3).map((d, i) => {
        const modCfg = MODALITY_CONFIG[d.modality] || { icon: '🧠', color: 'var(--primary-color)' };
        const isIncrease = d.direction === 'increases_stress';
        
        // Calculate relative contribution percentage
        const displayValue = (Math.abs(d.shap_value) / totalShap) * 100;

        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center',
                                 gap: 10, marginBottom: 10 }}>
            <span style={{ fontSize: '0.8rem', color: modCfg.color || 'var(--text-color)',
                            minWidth: 20 }}>
              {modCfg.icon}
            </span>
            <span style={{ flex: 1, color: 'var(--text-color)',
                            fontSize: '0.82rem', fontWeight: 500 }}>
              {featureLabel(d.feature)}
            </span>
            <span style={{
              color:      isIncrease ? '#F44336' : '#4CAF50',
              fontSize:   '0.78rem',
              fontWeight: 700,
              background: isIncrease ? 'rgba(244, 67, 54, 0.1)' : 'rgba(76, 175, 80, 0.1)',
              borderRadius: 4, padding: '2px 8px',
            }}>
              {isIncrease ? '▲' : '▼'} {displayValue.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function AnalysisPanel({ result, onRequestGame, previousResult, theme }) {
  const stress_level = result?.stress_level || "Low";
  // Clamp fused score to [0, 1] bounds
  const fused_score = clamp(result?.stress_probability || (result?.percentage ? result.percentage / 100 : 0), 0, 1);
  const confidence_score = clamp(result?.confidence || 0.9, 0, 1);
  const explainability = result?.explainability;
  const individual = result?.individual_predictions || {};
  const face_score = individual.facial ?? individual.face;
  const voice_score = individual.voice;
  const physio_score = individual.physiological ?? individual.physio;


  // Before vs After comparison
  const hasPrevious = previousResult != null;
  const delta       = hasPrevious
    ? ((previousResult.fused_score - fused_score) * 100).toFixed(1)
    : null;
  const improved    = hasPrevious && fused_score < previousResult.fused_score;

  // Resolve primary color dynamically based on theme to prevent SVG CSS var caching issues
  const themePrimaryColor = theme === 'earthy' ? '#8d9740' : '#00f2ff';

  // Modality percentages & levels calculations for the bars and charts
  const analysis = useMemo(() => {
    const points = [];
    if (face_score != null) {
      points.push({ key: "facial", label: "Facial", value: toPercent(face_score), reason: "facial expression stress" });
    }
    if (voice_score != null) {
      points.push({ key: "voice", label: "Voice", value: toPercent(voice_score), reason: "vocal strain" });
    }
    if (physio_score != null) {
      points.push({ key: "physiological", label: "Physiological", value: toPercent(physio_score), reason: "physiological arousal" });
    }

    if (points.length === 0) {
      points.push({ key: "overall", label: "Overall", value: toPercent(fused_score), reason: "overall indicators" });
    }

    const sorted = [...points].sort((a, b) => b.value - a.value);
    const total = points.reduce((sum, item) => sum + item.value, 0) || 1;

    // Formulate a specific, meaningful main cause description using specific features if possible
    const drivers = explainability?.top_drivers || [];
    const topFeatureLabels = drivers.slice(0, 2).map(d => featureLabel(d.feature));

    let cause = "";
    if (topFeatureLabels.length >= 2) {
      cause = `Fused output is primarily driven by elevated ${topFeatureLabels[0]} and increased ${topFeatureLabels[1]}.`;
    } else if (topFeatureLabels.length === 1) {
      cause = `Fused output is primarily driven by elevated ${topFeatureLabels[0]}.`;
    } else {
      cause = sorted.length > 1
        ? `Primary drivers are ${sorted[0].reason} and ${sorted[1].reason}.`
        : `Primary driver is ${sorted[0].reason}.`;
    }

    return {
      points,
      cause,
      contributions: points.map((item) => ({
        ...item,
        contribution: Math.round((item.value / total) * 100),
        band: toStressBand(item.value),
      })),
    };
  }, [face_score, voice_score, physio_score, fused_score, explainability]);

  // Construct chart data to show both Stress and Non-Stress percentages side-by-side
  const chartData = useMemo(() => {
    return analysis.points.map(p => {
      const val = p.value;
      return {
        label: p.label,
        Stress: val,
        "Non-Stress": clamp(100 - val, 0, 100),
      };
    });
  }, [analysis.points]);

  // Agreement, Coverage, Risk, Resilience calculation
  const activeValues = useMemo(() => {
    return analysis.points.map(p => p.value);
  }, [analysis.points]);

  const mean = useMemo(() => {
    return activeValues.length
      ? activeValues.reduce((sum, value) => sum + value, 0) / activeValues.length
      : 0;
  }, [activeValues]);

  const variance = useMemo(() => {
    return activeValues.length
      ? activeValues.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / activeValues.length
      : 0;
  }, [activeValues, mean]);

  const stdDev = useMemo(() => Math.sqrt(variance), [variance]);

  const maxPossibleStdDev = useMemo(() => {
    if (activeValues.length <= 1) return 1; // Avoid division by zero, though stdDev will be 0 anyway
    return activeValues.length === 3 ? 47.14045 : 50.0;
  }, [activeValues]);

  const agreement = useMemo(() => {
    if (activeValues.length <= 1) return 100;
    return clamp(100 - (stdDev / maxPossibleStdDev) * 100, 0, 100);
  }, [stdDev, activeValues.length, maxPossibleStdDev]);

  const completeness = useMemo(() => (activeValues.length / 3) * 100, [activeValues]);
  const riskIndex = useMemo(() => clamp(fused_score * 100, 0, 100), [fused_score]);
  const resilienceIndex = useMemo(() => clamp((100 - riskIndex) * 0.8 + (agreement * 0.2), 0, 100), [riskIndex, agreement]);

  const confidenceTarget = clamp(confidence_score * 100, 0, 100);

  const radarData = useMemo(() => {
    return [
      { subject: 'Risk', A: riskIndex },
      { subject: 'Agreement', A: agreement },
      { subject: 'Coverage', A: completeness },
      { subject: 'Resilience', A: resilienceIndex },
    ];
  }, [riskIndex, agreement, completeness, resilienceIndex]);

  const assessmentText = useMemo(() => {
    const isStress = riskIndex >= 50;
    const levelText = stress_level + " Stress";
    const dominantState = isStress ? "Stress Detected" : "No Stress";
    return `${levelText} - ${dominantState}`;
  }, [stress_level, riskIndex]);

  const getRecommendation = (level) => {
    switch(level) {
      case "Low":
        return "You're doing well! Maintain your current stress management practices.";
      case "Moderate":
        return "Consider taking short breaks and practicing deep breathing exercises.";
      case "High":
      case "Extreme":
        return "High stress detected. Consider speaking with a wellness professional and taking immediate breaks.";
      default:
        return "Continue monitoring your stress levels regularly.";
    }
  };

  if (!result) return null;

  return (
    <div style={{
      background: 'var(--card-bg)',
      border:     'var(--glass-border)',
      borderRadius: 16,
      padding:    24,
      boxShadow:  'var(--glass-shadow)',
      backdropFilter: 'blur(12px)',
      color:      'var(--text-color)',
    }}>

      {/* ── Before/After Banner ── */}
      {hasPrevious && (
        <div style={{
          background: improved ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)',
          border:     `1px solid ${improved ? '#4CAF50' : '#F44336'}`,
          borderRadius: 10, padding: '10px 16px',
          marginBottom: 24,
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <span style={{ fontSize: '1.2rem' }}>{improved ? '📉' : '📈'}</span>
          <div>
            <div style={{
              color:      improved ? '#4CAF50' : '#F44336',
              fontWeight: 700, fontSize: '0.9rem',
            }}>
              {improved
                ? `Stress reduced by ${delta}% after recovery`
                : `Stress increased by ${Math.abs(delta)}% — try another activity`}
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem', fontWeight: 600 }}>
              Before: {Math.round(previousResult.fused_score * 100)}% →
              After: {Math.round(fused_score * 100)}%
            </div>
          </div>
        </div>
      )}

      {/* ── Section 1: Title & Main Percentage ── */}
      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <h3 style={{ fontFamily: 'var(--font-headings)', fontWeight: 700, letterSpacing: '0.05em', color: 'var(--text-color)', fontSize: '1.45rem', textTransform: 'uppercase' }}>
          Overall Stress Assessment
        </h3>
        <div style={{ fontSize: '3.75rem', fontWeight: 800, color: 'var(--primary-color)', marginTop: 10, fontFamily: 'monospace' }}>
          {riskIndex.toFixed(1)}%
        </div>
        <div style={{ fontSize: '1.15rem', color: 'var(--primary-color)', fontWeight: 700, marginTop: 8, letterSpacing: '0.04em' }}>
          {assessmentText}
        </div>
      </div>

      {/* ── Key Metrics & Indicators Cohesive Panel ── */}
      <div style={{
        background: 'var(--accent-light-bg)',
        border: 'var(--glass-border)',
        borderRadius: 12,
        padding: 20,
        marginBottom: 32
      }}>
        <div style={{
          color: 'var(--text-muted)',
          fontSize: '0.75rem',
          fontWeight: 700,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          marginBottom: 16,
          textAlign: 'center'
        }}>
          📊 KEY METRICS & INDICATORS
        </div>

        {/* ── Section 2: Three Status Cards ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 16 }}>
          <div style={{
            background: 'var(--card-inner-bg)',
            border: 'var(--glass-border)',
            borderRadius: 10,
            padding: '12px 24px',
            textAlign: 'center',
            boxShadow: 'var(--glass-shadow)'
          }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Confidence</div>
            <div style={{ color: 'var(--text-color)', fontSize: '1.5rem', fontWeight: 700, marginTop: 4 }}>{confidenceTarget.toFixed(1)}%</div>
          </div>
          <div style={{
            background: 'var(--card-inner-bg)',
            border: 'var(--glass-border)',
            borderRadius: 10,
            padding: '12px 24px',
            textAlign: 'center',
            boxShadow: 'var(--glass-shadow)'
          }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Stress Probability</div>
            <div style={{ color: 'var(--text-color)', fontSize: '1.5rem', fontWeight: 700, marginTop: 4 }}>{riskIndex.toFixed(1)}%</div>
          </div>
          <div style={{
            background: 'var(--card-inner-bg)',
            border: 'var(--glass-border)',
            borderRadius: 10,
            padding: '12px 24px',
            textAlign: 'center',
            boxShadow: 'var(--glass-shadow)'
          }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>No Stress Probability</div>
            <div style={{ color: 'var(--text-color)', fontSize: '1.5rem', fontWeight: 700, marginTop: 4 }}>{(100 - riskIndex).toFixed(1)}%</div>
          </div>
        </div>

        {/* ── Section 3: Four Metric Cards ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16 }}>
          <div style={{ background: 'var(--card-inner-bg)', border: 'var(--glass-border)', borderRadius: 10, padding: 16, textAlign: 'center', boxShadow: 'var(--glass-shadow)' }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Agreement Score</div>
            <div style={{ color: 'var(--text-color)', fontSize: '1.45rem', fontWeight: 700, marginTop: 6 }}>{agreement.toFixed(1)}%</div>
          </div>
          <div style={{ background: 'var(--card-inner-bg)', border: 'var(--glass-border)', borderRadius: 10, padding: 16, textAlign: 'center', boxShadow: 'var(--glass-shadow)' }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Modality Coverage</div>
            <div style={{ color: 'var(--text-color)', fontSize: '1.45rem', fontWeight: 700, marginTop: 6 }}>{completeness.toFixed(1)}%</div>
          </div>
          <div style={{ background: 'var(--card-inner-bg)', border: 'var(--glass-border)', borderRadius: 10, padding: 16, textAlign: 'center', boxShadow: 'var(--glass-shadow)' }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Risk Index</div>
            <div style={{ color: 'var(--text-color)', fontSize: '1.45rem', fontWeight: 700, marginTop: 6 }}>{riskIndex.toFixed(1)}%</div>
          </div>
          <div style={{ background: 'var(--card-inner-bg)', border: 'var(--glass-border)', borderRadius: 10, padding: 16, textAlign: 'center', boxShadow: 'var(--glass-shadow)' }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600 }}>Resilience Index</div>
            <div style={{ color: 'var(--text-color)', fontSize: '1.45rem', fontWeight: 700, marginTop: 6 }}>{resilienceIndex.toFixed(1)}%</div>
          </div>
        </div>
      </div>

      {/* ── Section 4: Two Side-by-Side Plots ── */}
      <div className="row" style={{ marginBottom: 32 }}>
        {/* Left Column: Modality Stress Graph */}
        <div className="col-md-6">
          <div style={{ background: 'var(--accent-light-bg)', border: 'var(--glass-border)', borderRadius: 12, padding: 20, height: '100%' }}>
            <h4 style={{ color: themePrimaryColor, fontSize: '1.15rem', fontWeight: 700, marginBottom: 16, textAlign: 'center' }}>
              Modality Stress Graph
            </h4>
            <div style={{ width: '100%', height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(120, 120, 120, 0.15)" />
                  <XAxis dataKey="label" tick={{ fill: 'var(--text-color)', fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-color)', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--card-bg)',
                      border: 'var(--glass-border)',
                      borderRadius: 8,
                      color: 'var(--text-color)',
                      fontSize: 12
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-color)' }} />
                  <Bar dataKey="Stress" fill="#c74545" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Non-Stress" fill={themePrimaryColor} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Column: Health Radar */}
        <div className="col-md-6">
          <div style={{ background: 'var(--accent-light-bg)', border: 'var(--glass-border)', borderRadius: 12, padding: 20, height: '100%' }}>
            <h4 style={{ color: themePrimaryColor, fontSize: '1.15rem', fontWeight: 700, marginBottom: 16, textAlign: 'center' }}>
              Health Radar
            </h4>
            <div style={{ width: '100%', height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" radius="70%" data={radarData}>
                  <PolarGrid stroke="rgba(120, 120, 120, 0.2)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-color)', fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: 'var(--text-color)', fontSize: 9 }} />
                  <Radar name="Metrics" dataKey="A" stroke={themePrimaryColor} fill={themePrimaryColor} fillOpacity={0.3} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* ── Section 5: Specific Drivers & Breakdown (SHAP Details) ── */}
      <div style={{ background: 'var(--accent-light-bg)', border: 'var(--glass-border)', borderRadius: 12, padding: 20, marginBottom: 32 }}>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', letterSpacing: '0.1em', marginBottom: 12, fontWeight: 700 }}>
          MODALITY CONTRIBUTION BREAKDOWN
        </div>
        {analysis.contributions.map((entry) => {
          const mCfg = MODALITY_CONFIG[entry.key] || { color: 'var(--primary-color)' };
          return (
            <div key={entry.key} style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 500, marginBottom: 4 }}>
                <span>{entry.label}</span>
                <span>{entry.contribution}%</span>
              </div>
              <div className="contrib-track" style={{ height: 8, background: 'var(--bar-bg)', borderRadius: 4, overflow: 'hidden' }}>
                <div className="contrib-fill" style={{
                  height: '100%',
                  width: `${entry.contribution}%`,
                  background: `linear-gradient(90deg, ${mCfg.color}aa, ${mCfg.color})`,
                  borderRadius: 4,
                  transition: 'width 1s ease'
                }} />
              </div>
            </div>
          );
        })}

        <div className="analysis-cause" style={{
          marginTop: '1.25rem',
          background: 'var(--card-inner-bg)',
          border: 'var(--glass-border)',
          borderRadius: 10,
          padding: '12px 16px',
          boxShadow: 'var(--glass-shadow)'
        }}>
          <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem', letterSpacing: '0.1em', fontWeight: 700 }}>MAIN CONTRIBUTING DRIVER</small>
          <p style={{ margin: '4px 0 0', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-color)' }}>
            {analysis.cause}
          </p>
        </div>

        <SHAPDrivers explainability={explainability} />
      </div>

      {/* ── Section 6: Action Buttons & Recommendation ── */}
      <div style={{
        textAlign: 'center',
        padding: '14px 16px',
        background: 'var(--accent-light-bg)',
        border: 'var(--glass-border)',
        borderRadius: 8,
        fontWeight: 600,
        fontSize: '0.95rem',
        color: 'var(--text-color)',
        marginBottom: 20
      }}>
        Recommendation: {getRecommendation(stress_level)}
      </div>

      <div style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={onRequestGame}
          className="btn btn-primary"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '10px 24px',
            fontSize: '0.95rem',
            borderRadius: '30px',
            border: 'none',
            cursor: 'pointer',
            fontWeight: 700
          }}
        >
          🎮 Play Relaxation Game
        </button>

        <button
          onClick={() => {
            window.location.reload();
          }}
          className="btn btn-secondary"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '10px 24px',
            fontSize: '0.95rem',
            borderRadius: '30px',
            cursor: 'pointer',
            fontWeight: 700
          }}
        >
          📂 New Analysis
        </button>
      </div>
    </div>
  );
}
