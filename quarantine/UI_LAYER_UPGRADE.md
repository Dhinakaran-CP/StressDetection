# UI Output Layer — Complete Implementation Guide
## StressDetectionUsingML · Post-Analysis · Games · Re-Evaluation Loop
**Scope:** Frontend-only. No backend ML changes. Focuses on structured result display,  
interactive recovery activities, and the before/after re-evaluation closed loop.

---

## WHAT THIS GUIDE BUILDS

```
[Analyze Stress]
      ↓
┌─────────────────────────────────────────┐
│  RESULT PANEL  (structured, validated)  │
│  • Overall stress score + level badge   │
│  • Per-modality contribution bars       │
│  • SHAP trigger explanation             │
│  • Confidence + agreement indicators   │
└──────────────┬──────────────────────────┘
               │  if stress ≥ Moderate
               ↓
┌─────────────────────────────────────────┐
│  GAME PANEL  (5 activities)             │
│  • Each game has a minimum duration     │
│  • Timer enforces engagement            │
│  • Completion unlocks Re-Check button   │
└──────────────┬──────────────────────────┘
               │  after game completes
               ↓
┌─────────────────────────────────────────┐
│  RE-EVALUATION FLOW                     │
│  • Prompt to re-analyze                 │
│  • Run same modalities again            │
│  • Show Before vs After comparison      │
│  • Show improvement % + message         │
└─────────────────────────────────────────┘
```

---

## PART 1 — BACKEND RESPONSE CONTRACT

The UI must consume the `/api/multimodal/analyze` response correctly.  
Every UI component in this guide is built against this exact response shape.

```javascript
// Expected response shape from /api/multimodal/analyze
// The UI must handle every field being null gracefully.

const RESPONSE_CONTRACT = {
  // Core result
  stress_label:    "stressed" | "not_stressed",
  stress_level:    "Low" | "Moderate" | "High",
  fused_score:     0.0,   // float 0.0–1.0
  confidence_score: 0.0,  // float 0.0–1.0

  // Per-modality (null if not provided by user)
  face_score:   null | 0.0,   // float 0.0–1.0
  voice_score:  null | 0.0,
  physio_score: null | 0.0,

  // SHAP explainability
  explainability: {
    available: true | false,
    dominant_modality: "face" | "voice" | "physio" | null,
    top_drivers: [
      {
        modality: "voice",
        feature:  "jitter_percent",
        shap_value: 0.142,
        direction: "increases_stress"
      }
    ],
    modalities: {
      face:   { top_features: [...] },
      voice:  { top_features: [...] },
      physio: { top_features: [...] },
    }
  },

  // Metadata
  session_id:  123,
  inference_ms: 240,
  fusion_mode:  "reliability",
  modality_weights: { face: 0.38, voice: 0.41, physio: 0.21 }
};
```

---

## PART 2 — RESULT PANEL REBUILD (`AnalysisPanel.jsx`)

### 2.1 The Main Score Card

```jsx
// frontend/src/components/AnalysisPanel.jsx

import React, { useMemo } from 'react';

// ── Helpers ──────────────────────────────────────────────────────────────────

const LEVEL_CONFIG = {
  Low:      { color: '#4CAF50', glow: '#4CAF5040', icon: '😌', label: 'Low Stress',      bg: '#0a2a0e' },
  Moderate: { color: '#FF9800', glow: '#FF980040', icon: '😐', label: 'Moderate Stress', bg: '#2a1e00' },
  High:     { color: '#F44336', glow: '#F4433640', icon: '😰', label: 'High Stress',     bg: '#2a0a0a' },
};

const MODALITY_CONFIG = {
  face:   { icon: '👁',  label: 'Facial',        color: '#64B5F6' },
  voice:  { icon: '🎙',  label: 'Vocal',         color: '#CE93D8' },
  physio: { icon: '📈',  label: 'Physiological', color: '#80CBC4' },
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

function featureLabel(key) {
  return FEATURE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function confidenceLabel(score) {
  if (score >= 0.80) return { text: 'High Confidence',   color: '#4CAF50' };
  if (score >= 0.55) return { text: 'Medium Confidence', color: '#FF9800' };
  return               { text: 'Low Confidence',    color: '#F44336' };
}

function generateContextualSummary(result) {
  const { stress_level, explainability, face_score, voice_score, physio_score } = result;
  const dominant = explainability?.dominant_modality;
  const drivers  = explainability?.top_drivers || [];

  const modalities_used = [
    face_score   != null && 'facial',
    voice_score  != null && 'vocal',
    physio_score != null && 'physiological',
  ].filter(Boolean);

  if (stress_level === 'Low') {
    return `All ${modalities_used.length} modalities indicate a calm, relaxed state. No significant stress markers detected.`;
  }

  const dominantLabel = dominant
    ? MODALITY_CONFIG[dominant]?.label?.toLowerCase() || dominant
    : modalities_used[0];

  const topFeature = drivers[0]
    ? featureLabel(drivers[0].feature)
    : null;

  if (stress_level === 'High') {
    return topFeature
      ? `Elevated stress detected primarily through ${dominantLabel} analysis. ${topFeature} is the strongest contributing indicator.`
      : `Elevated stress detected across ${modalities_used.join(', ')} indicators. Consider taking a short recovery break.`;
  }

  return topFeature
    ? `Moderate stress detected. ${topFeature} in ${dominantLabel} data is the main driver. Monitor and manage as needed.`
    : `Moderate stress detected across ${modalities_used.join(' and ')} indicators.`;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ScoreRing({ score, level }) {
  const cfg    = LEVEL_CONFIG[level] || LEVEL_CONFIG.Low;
  const radius = 54;
  const circ   = 2 * Math.PI * radius;
  const offset = circ * (1 - score);
  const pct    = Math.round(score * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <svg width={140} height={140} viewBox="0 0 140 140">
        {/* Track */}
        <circle cx={70} cy={70} r={radius} fill="none"
          stroke="rgba(255,255,255,0.06)" strokeWidth={10} />
        {/* Progress */}
        <circle cx={70} cy={70} r={radius} fill="none"
          stroke={cfg.color} strokeWidth={10}
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
          style={{
            transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)',
            filter: `drop-shadow(0 0 8px ${cfg.color})`,
          }}
        />
        {/* Center text */}
        <text x={70} y={62} textAnchor="middle"
          fill={cfg.color} fontSize={28} fontWeight={700}
          fontFamily="'Segoe UI', sans-serif">
          {pct}%
        </text>
        <text x={70} y={82} textAnchor="middle"
          fill="rgba(255,255,255,0.5)" fontSize={11}
          fontFamily="'Segoe UI', sans-serif">
          STRESS
        </text>
      </svg>
      {/* Level badge */}
      <div style={{
        background: cfg.bg,
        border:     `1px solid ${cfg.color}`,
        borderRadius: 20,
        padding:    '6px 18px',
        color:      cfg.color,
        fontSize:   '0.88rem',
        fontWeight: 700,
        letterSpacing: '0.04em',
        boxShadow:  `0 0 12px ${cfg.glow}`,
      }}>
        {cfg.icon} {cfg.label.toUpperCase()}
      </div>
    </div>
  );
}

function ModalityBar({ modality, score, weight }) {
  if (score == null) return null;
  const cfg = MODALITY_CONFIG[modality] || {};
  const pct = Math.round(score * 100);
  const wPct = weight != null ? Math.round(weight * 100) : null;

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginBottom: 6 }}>
        <span style={{ color: 'rgba(255,255,255,0.75)', fontSize: '0.85rem' }}>
          {cfg.icon} {cfg.label}
          {wPct != null && (
            <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.75rem',
                           marginLeft: 8 }}>
              {wPct}% weight
            </span>
          )}
        </span>
        <span style={{ color: cfg.color, fontWeight: 700, fontSize: '0.9rem' }}>
          {pct}%
        </span>
      </div>
      <div style={{ height: 7, borderRadius: 4,
                    background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width:  `${pct}%`,
          background: `linear-gradient(90deg, ${cfg.color}88, ${cfg.color})`,
          borderRadius: 4,
          boxShadow: `0 0 8px ${cfg.color}66`,
          transition: 'width 1.0s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>
    </div>
  );
}

function SHAPDrivers({ explainability }) {
  if (!explainability?.available || !explainability.top_drivers?.length) return null;

  return (
    <div style={{
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 10, padding: '14px 16px', marginTop: 16,
    }}>
      <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.72rem',
                    letterSpacing: '0.1em', marginBottom: 10 }}>
        🧠 WHY THIS PREDICTION
      </div>
      {explainability.top_drivers.slice(0, 3).map((d, i) => {
        const modCfg = MODALITY_CONFIG[d.modality] || {};
        const isIncrease = d.direction === 'increases_stress';
        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center',
                                 gap: 10, marginBottom: 8 }}>
            <span style={{ fontSize: '0.8rem', color: modCfg.color || '#fff',
                            minWidth: 20 }}>
              {modCfg.icon}
            </span>
            <span style={{ flex: 1, color: 'rgba(255,255,255,0.7)',
                            fontSize: '0.82rem' }}>
              {featureLabel(d.feature)}
            </span>
            <span style={{
              color:      isIncrease ? '#F44336' : '#4CAF50',
              fontSize:   '0.78rem',
              fontWeight: 600,
              background: isIncrease ? '#F4433618' : '#4CAF5018',
              borderRadius: 4, padding: '2px 8px',
            }}>
              {isIncrease ? '▲' : '▼'} {Math.abs(d.shap_value * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ConfidenceRow({ confidence, inferenceMs }) {
  const conf = confidenceLabel(confidence || 0);
  return (
    <div style={{ display: 'flex', gap: 12, marginTop: 16, flexWrap: 'wrap' }}>
      <div style={{
        flex: 1, background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 8, padding: '10px 14px', textAlign: 'center',
      }}>
        <div style={{ color: conf.color, fontWeight: 700, fontSize: '0.95rem' }}>
          {Math.round((confidence || 0) * 100)}%
        </div>
        <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.72rem',
                      marginTop: 2 }}>
          {conf.text}
        </div>
      </div>
      {inferenceMs && (
        <div style={{
          flex: 1, background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: 8, padding: '10px 14px', textAlign: 'center',
        }}>
          <div style={{ color: 'rgba(255,255,255,0.8)', fontWeight: 700,
                         fontSize: '0.95rem' }}>
            {inferenceMs}ms
          </div>
          <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.72rem',
                         marginTop: 2 }}>
            Analysis Time
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function AnalysisPanel({ result, onRequestGame, previousResult }) {
  const summary = useMemo(() => generateContextualSummary(result), [result]);

  if (!result) return null;

  const {
    stress_level, fused_score, confidence_score, inference_ms,
    face_score, voice_score, physio_score,
    explainability, modality_weights,
  } = result;

  const cfg          = LEVEL_CONFIG[stress_level] || LEVEL_CONFIG.Low;
  const showGamePrompt = stress_level === 'Moderate' || stress_level === 'High';

  // Before vs After comparison
  const hasPrevious = previousResult != null;
  const delta       = hasPrevious
    ? ((previousResult.fused_score - fused_score) * 100).toFixed(1)
    : null;
  const improved    = hasPrevious && fused_score < previousResult.fused_score;

  return (
    <div style={{
      background: `linear-gradient(135deg, var(--card-bg, #0f1923) 0%, ${cfg.bg} 100%)`,
      border:     `1px solid ${cfg.color}44`,
      borderRadius: 16,
      padding:    24,
      boxShadow:  `0 8px 32px ${cfg.glow}, inset 0 1px 0 rgba(255,255,255,0.05)`,
      backdropFilter: 'blur(12px)',
    }}>

      {/* ── Before/After Banner ── */}
      {hasPrevious && (
        <div style={{
          background: improved ? '#0a2a0e' : '#2a0a0a',
          border:     `1px solid ${improved ? '#4CAF50' : '#F44336'}`,
          borderRadius: 10, padding: '10px 16px',
          marginBottom: 20,
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
            <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.78rem' }}>
              Before: {Math.round(previousResult.fused_score * 100)}% →
              After: {Math.round(fused_score * 100)}%
            </div>
          </div>
        </div>
      )}

      {/* ── Header row: ring + summary ── */}
      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start',
                    flexWrap: 'wrap' }}>
        <ScoreRing score={fused_score || 0} level={stress_level} />

        <div style={{ flex: 1, minWidth: 180 }}>
          {/* Contextual summary */}
          <p style={{
            color: 'rgba(255,255,255,0.75)', fontSize: '0.9rem',
            lineHeight: 1.6, margin: '0 0 16px',
          }}>
            {summary}
          </p>

          {/* Modality contribution bars */}
          <div style={{ marginTop: 4 }}>
            <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.72rem',
                           letterSpacing: '0.1em', marginBottom: 10 }}>
              MODALITY CONTRIBUTIONS
            </div>
            <ModalityBar modality="face"   score={face_score}
              weight={modality_weights?.face} />
            <ModalityBar modality="voice"  score={voice_score}
              weight={modality_weights?.voice} />
            <ModalityBar modality="physio" score={physio_score}
              weight={modality_weights?.physio} />
          </div>
        </div>
      </div>

      {/* ── SHAP drivers ── */}
      <SHAPDrivers explainability={explainability} />

      {/* ── Confidence + timing ── */}
      <ConfidenceRow confidence={confidence_score} inferenceMs={inference_ms} />

      {/* ── Game prompt ── */}
      {showGamePrompt && !hasPrevious && (
        <div style={{
          marginTop: 20,
          background: 'rgba(255,255,255,0.03)',
          border:     `1px solid ${cfg.color}44`,
          borderRadius: 10, padding: '14px 16px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          gap: 12, flexWrap: 'wrap',
        }}>
          <div>
            <div style={{ color: cfg.color, fontWeight: 600, fontSize: '0.88rem' }}>
              Recovery recommended
            </div>
            <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.78rem',
                           marginTop: 2 }}>
              Try a 2–5 minute activity, then re-check your stress level.
            </div>
          </div>
          <button
            onClick={onRequestGame}
            style={{
              background:    `linear-gradient(135deg, ${cfg.color}22, ${cfg.color}44)`,
              border:        `1px solid ${cfg.color}`,
              color:         cfg.color,
              borderRadius:  8,
              padding:       '10px 20px',
              cursor:        'pointer',
              fontWeight:    700,
              fontSize:      '0.85rem',
              whiteSpace:    'nowrap',
              transition:    'all 0.2s',
            }}
          >
            Start Recovery →
          </button>
        </div>
      )}
    </div>
  );
}
```

---

## PART 3 — GAME PANEL (`GamePanel.jsx`)

Each game has a **minimum engagement time** before the "Done" button activates. This prevents users from instantly clicking through. After any game completes, the panel shows a "Re-Check Stress" prompt.

```jsx
// frontend/src/components/GamePanel.jsx

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ── Game 1: Breathing ─────────────────────────────────────────────────────────
function BreathingGame({ onComplete }) {
  const [phase,  setPhase]  = useState('inhale');
  const [count,  setCount]  = useState(4);
  const [cycles, setCycles] = useState(0);
  const TARGET_CYCLES = 5;  // minimum 5 full cycles ≈ 70 seconds

  const PHASES = {
    inhale:  { duration: 4, next: 'hold',   label: 'Breathe In',  color: '#64B5F6' },
    hold:    { duration: 4, next: 'exhale',  label: 'Hold',        color: '#CE93D8' },
    exhale:  { duration: 6, next: 'inhale',  label: 'Breathe Out', color: '#80CBC4' },
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setCount(c => {
        if (c <= 1) {
          setPhase(p => {
            const next = PHASES[p].next;
            if (next === 'inhale') setCycles(cy => cy + 1);
            return next;
          });
          return PHASES[PHASES[phase].next].duration;
        }
        return c - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [phase]);

  const currentPhase = PHASES[phase];
  const radius  = 60;
  const circ    = 2 * Math.PI * radius;
  const progress = count / currentPhase.duration;
  const done     = cycles >= TARGET_CYCLES;

  return (
    <div style={{ textAlign: 'center', padding: 24 }}>
      <h3 style={{ color: '#64B5F6', marginBottom: 4 }}>Guided Breathing</h3>
      <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.8rem', marginBottom: 20 }}>
        5 complete cycles · {Math.max(0, TARGET_CYCLES - cycles)} remaining
      </p>

      {/* Animated ring */}
      <div style={{ position: 'relative', display: 'inline-block', marginBottom: 20 }}>
        <svg width={160} height={160} viewBox="0 0 160 160">
          <circle cx={80} cy={80} r={radius} fill="none"
            stroke="rgba(255,255,255,0.06)" strokeWidth={10} />
          <circle cx={80} cy={80} r={radius} fill="none"
            stroke={currentPhase.color} strokeWidth={10}
            strokeDasharray={circ}
            strokeDashoffset={circ * (1 - progress)}
            strokeLinecap="round"
            transform="rotate(-90 80 80)"
            style={{
              transition: 'stroke-dashoffset 1s linear, stroke 0.5s',
              filter: `drop-shadow(0 0 10px ${currentPhase.color})`,
            }}
          />
          <text x={80} y={74} textAnchor="middle"
            fill={currentPhase.color} fontSize={24} fontWeight={700}
            fontFamily="'Segoe UI', sans-serif">{count}</text>
          <text x={80} y={96} textAnchor="middle"
            fill="rgba(255,255,255,0.6)" fontSize={12}
            fontFamily="'Segoe UI', sans-serif">{currentPhase.label}</text>
        </svg>
      </div>

      <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.78rem', marginBottom: 20 }}>
        Cycle {Math.min(cycles + 1, TARGET_CYCLES)} of {TARGET_CYCLES}
      </div>

      {done && (
        <button onClick={onComplete} style={DONE_BTN_STYLE('#4CAF50')}>
          ✓ Breathing Complete — Re-Check Stress
        </button>
      )}
    </div>
  );
}

// ── Game 2: Focus Tap ─────────────────────────────────────────────────────────
function FocusTapGame({ onComplete }) {
  const TARGET   = 30;
  const [count, setCount] = useState(0);
  const [streak, setStreak] = useState(0);
  const [maxStreak, setMaxStreak] = useState(0);
  const [lastTap, setLastTap] = useState(null);
  const [feedback, setFeedback] = useState('');

  const tap = useCallback(() => {
    const now = Date.now();
    const elapsed = lastTap ? now - lastTap : 999;
    setLastTap(now);

    let newStreak = streak;
    if (elapsed < 800) {
      newStreak += 1;
      setFeedback(newStreak >= 5 ? '🔥 On fire!' : '⚡ Good rhythm!');
    } else {
      newStreak = 0;
      setFeedback('');
    }
    setStreak(newStreak);
    setMaxStreak(m => Math.max(m, newStreak));
    setCount(c => c + 1);
  }, [streak, lastTap]);

  const pct  = Math.min(count / TARGET, 1);
  const done = count >= TARGET;

  return (
    <div style={{ textAlign: 'center', padding: 24 }}>
      <h3 style={{ color: '#CE93D8', marginBottom: 4 }}>Focus Tap</h3>
      <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.8rem', marginBottom: 16 }}>
        Tap the button {TARGET} times in a steady rhythm
      </p>

      {/* Progress bar */}
      <div style={{ height: 6, borderRadius: 3,
                    background: 'rgba(255,255,255,0.06)', margin: '0 auto 20px',
                    maxWidth: 280 }}>
        <div style={{
          height: '100%', borderRadius: 3,
          width: `${pct * 100}%`,
          background: 'linear-gradient(90deg, #CE93D888, #CE93D8)',
          boxShadow: '0 0 8px #CE93D866',
          transition: 'width 0.3s ease',
        }} />
      </div>

      {/* Tap button */}
      {!done && (
        <button onClick={tap} style={{
          width: 110, height: 110, borderRadius: '50%',
          background: 'radial-gradient(circle, #CE93D822, #CE93D811)',
          border: '2px solid #CE93D8',
          color: '#CE93D8',
          fontSize: '1.8rem',
          fontWeight: 700,
          cursor: 'pointer',
          boxShadow: `0 0 20px #CE93D833`,
          transition: 'transform 0.08s, box-shadow 0.08s',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto',
        }}
          onMouseDown={e => e.currentTarget.style.transform = 'scale(0.92)'}
          onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}
        >
          {count}
        </button>
      )}

      {feedback && !done && (
        <div style={{ color: '#FFD54F', fontSize: '0.85rem',
                      marginTop: 12, height: 20 }}>
          {feedback}
        </div>
      )}
      {maxStreak >= 3 && (
        <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.75rem',
                      marginTop: done ? 12 : 8 }}>
          Best streak: {maxStreak} taps
        </div>
      )}

      {done && (
        <button onClick={onComplete} style={DONE_BTN_STYLE('#CE93D8')}>
          ✓ Focus Restored — Re-Check Stress
        </button>
      )}
    </div>
  );
}

// ── Game 3: Calm Timer ────────────────────────────────────────────────────────
function CalmTimer({ onComplete }) {
  const DURATION     = 120;  // 2 minutes
  const [remaining, setRemaining] = useState(DURATION);
  const [started,   setStarted]   = useState(false);

  useEffect(() => {
    if (!started) return;
    if (remaining <= 0) { onComplete(); return; }
    const t = setTimeout(() => setRemaining(r => r - 1), 1000);
    return () => clearTimeout(t);
  }, [remaining, started]);

  const pct  = ((DURATION - remaining) / DURATION) * 100;
  const mins = Math.floor(remaining / 60);
  const secs = remaining % 60;

  return (
    <div style={{ textAlign: 'center', padding: 24 }}>
      <h3 style={{ color: '#80CBC4', marginBottom: 4 }}>Calm Mode</h3>
      <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.8rem', marginBottom: 20 }}>
        2 minutes of quiet focus. Close your eyes when ready.
      </p>

      <div style={{
        fontSize: '3.5rem', fontWeight: 700,
        color: remaining < 30 ? '#4CAF50' : '#80CBC4',
        letterSpacing: '-0.02em', marginBottom: 8,
        fontVariantNumeric: 'tabular-nums',
        filter: `drop-shadow(0 0 16px ${remaining < 30 ? '#4CAF5066' : '#80CBC466'})`,
      }}>
        {String(mins).padStart(2,'0')}:{String(secs).padStart(2,'0')}
      </div>

      <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.06)',
                    maxWidth: 240, margin: '0 auto 24px' }}>
        <div style={{
          height: '100%', borderRadius: 2,
          width:  `${pct}%`,
          background: 'linear-gradient(90deg, #80CBC488, #80CBC4)',
          transition: 'width 1s linear',
        }} />
      </div>

      {!started ? (
        <button onClick={() => setStarted(true)} style={DONE_BTN_STYLE('#80CBC4')}>
          Begin Calm Mode
        </button>
      ) : remaining > 0 ? (
        <p style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.8rem' }}>
          Breathe naturally. The timer will complete automatically.
        </p>
      ) : (
        <button onClick={onComplete} style={DONE_BTN_STYLE('#4CAF50')}>
          ✓ Calm Session Complete — Re-Check Stress
        </button>
      )}
    </div>
  );
}

// ── Game 4: Gratitude ─────────────────────────────────────────────────────────
function GratitudeGame({ onComplete }) {
  const prompts = [
    'Something or someone that made you smile recently',
    'A small win you had today, however minor',
    'Something in your environment right now that brings comfort',
  ];
  const [items,   setItems]   = useState(['', '', '']);
  const [focused, setFocused] = useState(null);

  const allFilled  = items.every(s => s.trim().length >= 3);
  const totalChars = items.reduce((a, s) => a + s.trim().length, 0);

  return (
    <div style={{ padding: 24 }}>
      <h3 style={{ color: '#FFD54F', marginBottom: 4 }}>Gratitude Reflection</h3>
      <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.8rem', marginBottom: 20 }}>
        Write three things. Research shows this reframes the amygdala's threat response.
      </p>

      {prompts.map((prompt, i) => (
        <div key={i} style={{ marginBottom: 14 }}>
          <div style={{
            color: 'rgba(255,255,255,0.4)', fontSize: '0.74rem',
            marginBottom: 5, letterSpacing: '0.03em',
          }}>
            {i + 1}. {prompt}
          </div>
          <textarea
            value={items[i]}
            onChange={e => {
              const n = [...items];
              n[i] = e.target.value;
              setItems(n);
            }}
            onFocus={() => setFocused(i)}
            onBlur={() => setFocused(null)}
            placeholder="Type here..."
            rows={2}
            style={{
              width: '100%', boxSizing: 'border-box',
              background: 'rgba(255,255,255,0.04)',
              border: `1px solid ${focused === i ? '#FFD54F66' : 'rgba(255,255,255,0.1)'}`,
              borderRadius: 8, padding: '10px 12px',
              color: 'rgba(255,255,255,0.85)', fontSize: '0.88rem',
              resize: 'none', outline: 'none',
              transition: 'border-color 0.2s',
              fontFamily: 'inherit',
            }}
          />
        </div>
      ))}

      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginTop: 4 }}>
        <span style={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.72rem' }}>
          {totalChars} characters
        </span>
        <button
          onClick={() => allFilled && onComplete()}
          disabled={!allFilled}
          style={allFilled ? DONE_BTN_STYLE('#FFD54F') : DISABLED_BTN_STYLE}
        >
          ✓ Reflected — Re-Check Stress
        </button>
      </div>
    </div>
  );
}

// ── Game 5: Posture Reset ─────────────────────────────────────────────────────
function PostureReset({ onComplete }) {
  const steps = [
    { icon: '💺', text: 'Sit fully back — back flat against the chair',     tip: 'Slumping compresses the diaphragm, raising cortisol.' },
    { icon: '🦶', text: 'Both feet flat on the floor',                      tip: 'Grounding your feet activates the parasympathetic system.' },
    { icon: '🫁', text: 'Take one slow, full breath from your diaphragm',   tip: 'One deep breath lowers heart rate within 30 seconds.' },
    { icon: '💆', text: 'Drop your shoulders — release all tension',        tip: 'Trapezius tension is a direct stress indicator.' },
    { icon: '😶', text: 'Unclench your jaw — tongue off the roof of mouth', tip: 'Masseter tension mirrors emotional stress levels.' },
    { icon: '👁',  text: 'Look away from the screen for 10 seconds',        tip: '20-20-20 rule: every 20 min, look 20 ft away for 20s.' },
  ];
  const [checked, setChecked] = useState(Array(steps.length).fill(false));
  const allDone = checked.every(Boolean);

  const toggle = i => {
    const n = [...checked];
    n[i] = !n[i];
    setChecked(n);
  };

  return (
    <div style={{ padding: 24 }}>
      <h3 style={{ color: '#A5D6A7', marginBottom: 4 }}>Posture Reset</h3>
      <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.8rem', marginBottom: 18 }}>
        Check each step as you complete it. Each has a physiological basis.
      </p>

      {steps.map((step, i) => (
        <div
          key={i}
          onClick={() => toggle(i)}
          style={{
            display: 'flex', alignItems: 'flex-start', gap: 12,
            padding: '10px 12px', borderRadius: 8, marginBottom: 6,
            background: checked[i] ? 'rgba(165,214,167,0.08)' : 'rgba(255,255,255,0.03)',
            border: `1px solid ${checked[i] ? '#A5D6A744' : 'rgba(255,255,255,0.06)'}`,
            cursor: 'pointer', transition: 'all 0.2s',
          }}
        >
          <div style={{
            width: 20, height: 20, borderRadius: 4, flexShrink: 0, marginTop: 1,
            background: checked[i] ? '#A5D6A7' : 'rgba(255,255,255,0.08)',
            border: `1px solid ${checked[i] ? '#A5D6A7' : 'rgba(255,255,255,0.2)'}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '0.75rem', color: '#1a1a1a', fontWeight: 700,
            transition: 'all 0.2s',
          }}>
            {checked[i] ? '✓' : ''}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{
              color: checked[i] ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.8)',
              fontSize: '0.85rem',
              textDecoration: checked[i] ? 'line-through' : 'none',
              marginBottom: 2,
            }}>
              {step.icon} {step.text}
            </div>
            <div style={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.72rem' }}>
              {step.tip}
            </div>
          </div>
        </div>
      ))}

      <div style={{ marginTop: 16, textAlign: 'right' }}>
        {allDone ? (
          <button onClick={onComplete} style={DONE_BTN_STYLE('#A5D6A7')}>
            ✓ Reset Complete — Re-Check Stress
          </button>
        ) : (
          <span style={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.78rem' }}>
            {checked.filter(Boolean).length} / {steps.length} completed
          </span>
        )}
      </div>
    </div>
  );
}

// ── Shared button styles ──────────────────────────────────────────────────────
const DONE_BTN_STYLE = (color) => ({
  background: `linear-gradient(135deg, ${color}22, ${color}44)`,
  border: `1px solid ${color}`,
  color,
  borderRadius: 8, padding: '11px 22px',
  cursor: 'pointer', fontWeight: 700,
  fontSize: '0.88rem', marginTop: 8,
  transition: 'all 0.2s',
});

const DISABLED_BTN_STYLE = {
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.12)',
  color: 'rgba(255,255,255,0.25)',
  borderRadius: 8, padding: '11px 22px',
  cursor: 'not-allowed', fontWeight: 700,
  fontSize: '0.88rem', marginTop: 8,
};

// ── Game Selector + Container ─────────────────────────────────────────────────

const GAMES = [
  { key: 'breathing', label: 'Breathing',   icon: '🫁', color: '#64B5F6',
    desc: '5 cycles · ~70 seconds',
    component: BreathingGame,
    recommended: ['High'] },
  { key: 'focus',     label: 'Focus Tap',   icon: '👆', color: '#CE93D8',
    desc: '30 taps · ~1 minute',
    component: FocusTapGame,
    recommended: ['Moderate', 'High'] },
  { key: 'calm',      label: 'Calm Timer',  icon: '⏱', color: '#80CBC4',
    desc: '2 minutes silence',
    component: CalmTimer,
    recommended: ['High'] },
  { key: 'gratitude', label: 'Gratitude',   icon: '🙏', color: '#FFD54F',
    desc: '3 reflections',
    component: GratitudeGame,
    recommended: ['Moderate'] },
  { key: 'posture',   label: 'Posture',     icon: '💺', color: '#A5D6A7',
    desc: '6-step checklist',
    component: PostureReset,
    recommended: ['Moderate', 'High'] },
];

export default function GamePanel({ stressLevel, onGameComplete, onDismiss }) {
  const [selected,    setSelected]    = useState(null);
  const [completed,   setCompleted]   = useState([]);
  const [showRecheck, setShowRecheck] = useState(false);

  const handleComplete = useCallback((gameKey) => {
    setCompleted(c => [...c, gameKey]);
    setSelected(null);
    setShowRecheck(true);
  }, []);

  const GameComponent = selected
    ? GAMES.find(g => g.key === selected)?.component
    : null;

  if (showRecheck) {
    return (
      <div style={PANEL_WRAPPER}>
        <div style={{ textAlign: 'center', padding: 32 }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>✅</div>
          <h3 style={{ color: '#4CAF50', marginBottom: 8 }}>
            Activity Complete
          </h3>
          <p style={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.88rem',
                      maxWidth: 320, margin: '0 auto 24px', lineHeight: 1.6 }}>
            You spent time on recovery. Let's measure whether your stress has changed.
            This takes the same amount of time as the original analysis.
          </p>
          <button onClick={onGameComplete} style={DONE_BTN_STYLE('#4CAF50')}>
            🔄 Re-Check My Stress Now
          </button>
          <div style={{ marginTop: 12 }}>
            <button onClick={() => setShowRecheck(false)}
              style={{
                background: 'none', border: 'none',
                color: 'rgba(255,255,255,0.3)', cursor: 'pointer',
                fontSize: '0.8rem',
              }}>
              Try another activity first
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (GameComponent) {
    const game = GAMES.find(g => g.key === selected);
    return (
      <div style={PANEL_WRAPPER}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10,
                      marginBottom: 4, padding: '4px 24px 0' }}>
          <button onClick={() => setSelected(null)}
            style={{ background: 'none', border: 'none',
                     color: 'rgba(255,255,255,0.4)', cursor: 'pointer',
                     fontSize: '1.1rem' }}>
            ← Back
          </button>
          <span style={{ color: game.color, fontWeight: 600, fontSize: '0.85rem' }}>
            {game.icon} {game.label}
          </span>
        </div>
        <GameComponent onComplete={() => handleComplete(selected)} />
      </div>
    );
  }

  return (
    <div style={PANEL_WRAPPER}>
      <div style={{ padding: '20px 24px 0' }}>
        <h3 style={{ color: 'rgba(255,255,255,0.85)', margin: '0 0 4px' }}>
          Recovery Activities
        </h3>
        <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.8rem', margin: '0 0 18px' }}>
          {stressLevel === 'High'
            ? 'High stress detected. Any of these activities will help right now.'
            : 'Moderate stress detected. A short reset will improve clarity.'}
        </p>
      </div>

      <div style={{ padding: '0 16px 16px',
                    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        {GAMES.map(game => {
          const isRecommended = game.recommended.includes(stressLevel);
          const isDone        = completed.includes(game.key);
          return (
            <button key={game.key}
              onClick={() => setSelected(game.key)}
              style={{
                background: isDone
                  ? 'rgba(76,175,80,0.08)'
                  : isRecommended
                    ? `rgba(255,255,255,0.05) linear-gradient(135deg, ${game.color}11, transparent)`
                    : 'rgba(255,255,255,0.03)',
                border: `1px solid ${isDone ? '#4CAF5044' : isRecommended ? game.color + '55' : 'rgba(255,255,255,0.08)'}`,
                borderRadius: 10, padding: '14px 12px',
                cursor: 'pointer', textAlign: 'left',
                transition: 'all 0.2s',
              }}>
              <div style={{ fontSize: '1.4rem', marginBottom: 6 }}>
                {isDone ? '✅' : game.icon}
              </div>
              <div style={{ color: isDone ? '#4CAF50' : game.color,
                             fontWeight: 700, fontSize: '0.88rem',
                             marginBottom: 3 }}>
                {game.label}
                {isRecommended && !isDone && (
                  <span style={{ background: `${game.color}22`, color: game.color,
                                  fontSize: '0.65rem', padding: '1px 6px',
                                  borderRadius: 10, marginLeft: 6 }}>
                    Recommended
                  </span>
                )}
              </div>
              <div style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.72rem' }}>
                {isDone ? 'Completed' : game.desc}
              </div>
            </button>
          );
        })}
      </div>

      {completed.length > 0 && (
        <div style={{ padding: '0 16px 16px' }}>
          <button onClick={() => setShowRecheck(true)}
            style={{ width: '100%', ...DONE_BTN_STYLE('#4CAF50') }}>
            🔄 Re-Check My Stress Now
          </button>
        </div>
      )}
    </div>
  );
}

const PANEL_WRAPPER = {
  background: 'rgba(255,255,255,0.02)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
  backdropFilter: 'blur(12px)',
  overflow: 'hidden',
};
```

---

## PART 4 — RE-EVALUATION LOOP (`Dashboard.js`)

Wire everything together in Dashboard.js. This manages the complete state machine:  
`idle → analyzing → result → game → re-analyzing → comparison`.

```jsx
// In Dashboard.js — add these state variables and handlers

import { useState, useCallback } from 'react';
import AnalysisPanel from '../components/AnalysisPanel';
import GamePanel     from '../components/GamePanel';

// State machine
const [phase,           setPhase]           = useState('idle');
// phases: 'idle' | 'analyzing' | 'result' | 'game' | 'reanalyzing' | 'comparison'

const [currentResult,   setCurrentResult]   = useState(null);
const [previousResult,  setPreviousResult]  = useState(null);
const [analysisPayload, setAnalysisPayload] = useState(null);
// analysisPayload stores the form data so re-analysis uses identical inputs

const handleAnalyze = useCallback(async (formData) => {
  setPhase('analyzing');
  setAnalysisPayload(formData);  // save for re-analysis
  setPreviousResult(null);
  setCurrentResult(null);

  try {
    const response = await fetch('/api/multimodal/analyze', {
      method:  'POST',
      body:    formData,
    });
    const data = await response.json();

    // Validate response has minimum required fields
    if (!data.stress_level || data.fused_score == null) {
      throw new Error('Invalid response from server');
    }

    setCurrentResult(data);
    setPhase('result');
  } catch (err) {
    console.error('Analysis failed:', err);
    setPhase('idle');
    // show error toast
  }
}, []);

const handleRequestGame = useCallback(() => {
  setPhase('game');
}, []);

const handleGameComplete = useCallback(async () => {
  // User completed a game and wants to re-analyze
  if (!analysisPayload) {
    setPhase('idle');
    return;
  }

  setPhase('reanalyzing');
  setPreviousResult(currentResult);  // save current as "before"

  try {
    const response = await fetch('/api/multimodal/analyze', {
      method: 'POST',
      body:   analysisPayload,  // same inputs as before
    });
    const data = await response.json();

    if (!data.stress_level || data.fused_score == null) {
      throw new Error('Invalid response from server');
    }

    setCurrentResult(data);
    setPhase('comparison');  // shows before/after in AnalysisPanel
  } catch (err) {
    console.error('Re-analysis failed:', err);
    setPhase('result');  // fall back to result without comparison
  }
}, [analysisPayload, currentResult]);

// In the render:
return (
  <div>
    {/* ... input controls ... */}

    {/* Analysis trigger */}
    {phase === 'idle' && (
      <button onClick={() => handleAnalyze(buildFormData())}>
        Analyze Stress
      </button>
    )}

    {/* Loading states */}
    {(phase === 'analyzing' || phase === 'reanalyzing') && (
      <div style={{ textAlign: 'center', padding: 40 }}>
        <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.9rem' }}>
          {phase === 'reanalyzing'
            ? '🔄 Re-analyzing after recovery...'
            : '🔍 Analyzing stress indicators...'}
        </div>
      </div>
    )}

    {/* Result panel */}
    {(phase === 'result' || phase === 'comparison') && currentResult && (
      <AnalysisPanel
        result={currentResult}
        previousResult={phase === 'comparison' ? previousResult : null}
        onRequestGame={handleRequestGame}
      />
    )}

    {/* Game panel */}
    {phase === 'game' && (
      <GamePanel
        stressLevel={currentResult?.stress_level}
        onGameComplete={handleGameComplete}
        onDismiss={() => setPhase('result')}
      />
    )}

    {/* Re-analyze button after comparison */}
    {phase === 'comparison' && (
      <div style={{ marginTop: 16, textAlign: 'center' }}>
        <button onClick={() => {
          setPhase('idle');
          setCurrentResult(null);
          setPreviousResult(null);
        }}
          style={{ background: 'none', border: '1px solid rgba(255,255,255,0.15)',
                   color: 'rgba(255,255,255,0.5)', borderRadius: 8,
                   padding: '8px 20px', cursor: 'pointer', fontSize: '0.82rem' }}>
          Start New Analysis
        </button>
      </div>
    )}
  </div>
);
```

---

## PART 5 — VALIDATION RULES

Add these to the frontend before submitting to the backend. Show clear errors instead of silent failures.

```javascript
// frontend/src/utils/validateInputs.js

export function validateAnalysisInputs({ faceFile, voiceFile, eegData, gsrData }) {
  const errors = [];

  // Must have at least one modality
  const hasModality = faceFile || voiceFile || eegData?.trim() || gsrData?.trim();
  if (!hasModality) {
    errors.push('Provide at least one input: photo, voice recording, or EEG/GSR data.');
  }

  // Face file validation
  if (faceFile) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(faceFile.type)) {
      errors.push('Face image must be JPG or PNG format.');
    }
    if (faceFile.size > 10 * 1024 * 1024) {
      errors.push('Face image must be under 10MB.');
    }
  }

  // Voice file validation
  if (voiceFile) {
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg',
                        'audio/webm', 'audio/m4a'];
    if (!validTypes.some(t => voiceFile.type.includes(t.split('/')[1]))) {
      errors.push('Voice recording must be WAV, MP3, OGG, WebM, or M4A format.');
    }
    if (voiceFile.size > 50 * 1024 * 1024) {
      errors.push('Voice file must be under 50MB.');
    }
  }

  // EEG validation: must be comma-separated numbers
  if (eegData?.trim()) {
    const nums = eegData.trim().split(',').map(Number);
    if (nums.some(isNaN)) {
      errors.push('EEG data must be comma-separated numbers only (e.g., 0.52, 0.61, 0.58).');
    }
    if (nums.length < 10) {
      errors.push(`EEG data needs at least 10 values for reliable analysis (you provided ${nums.length}).`);
    }
  }

  // GSR validation
  if (gsrData?.trim()) {
    const nums = gsrData.trim().split(',').map(Number);
    if (nums.some(isNaN)) {
      errors.push('GSR data must be comma-separated numbers only.');
    }
    if (nums.some(n => n < 0 || n > 100)) {
      errors.push('GSR values should be in the 0–100 µS range.');
    }
  }

  return errors;
}

// Response validation after fetch
export function validateAnalysisResponse(data) {
  const errors = [];

  if (!data) {
    errors.push('Empty response from server.');
    return errors;
  }
  if (!['Low','Moderate','High'].includes(data.stress_level)) {
    errors.push(`Invalid stress_level: ${data.stress_level}`);
  }
  if (typeof data.fused_score !== 'number' || data.fused_score < 0 || data.fused_score > 1) {
    errors.push(`Invalid fused_score: ${data.fused_score}`);
  }
  if (data.confidence_score != null &&
      (typeof data.confidence_score !== 'number' ||
       data.confidence_score < 0 || data.confidence_score > 1)) {
    errors.push(`Invalid confidence_score: ${data.confidence_score}`);
  }
  // At least one modality score must be present
  const hasScore = data.face_score != null ||
                   data.voice_score != null ||
                   data.physio_score != null;
  if (!hasScore) {
    errors.push('No modality scores in response. The model may not have processed the inputs.');
  }

  return errors;
}
```

---

## PART 6 — WIRING CHECKLIST FOR AGENT

```
[ ] AnalysisPanel.jsx rebuilt with ScoreRing, ModalityBar, SHAPDrivers, ConfidenceRow
[ ] AnalysisPanel accepts `previousResult` prop — renders Before/After banner when present
[ ] AnalysisPanel `onRequestGame` prop triggers phase transition to 'game'
[ ] GamePanel.jsx has all 5 games: Breathing, FocusTap, CalmTimer, Gratitude, PostureReset
[ ] Each game has minimum engagement requirement before Done button activates
[ ] GamePanel shows Recommended badge based on stress level
[ ] GamePanel shows Re-Check prompt after any game completes
[ ] Dashboard.js state machine: idle → analyzing → result → game → reanalyzing → comparison
[ ] Re-analysis uses the same form data (analysisPayload) as original analysis
[ ] validateInputs.js imported in Dashboard.js — errors shown before fetch is called
[ ] validateAnalysisResponse called after every fetch — errors surfaced to user
[ ] CSS variables used throughout: var(--card-bg), var(--glass-border) — no hardcoded colors
[ ] Both Cyber and Earthy themes render correctly (test by switching theme mid-session)
[ ] Loading state shows different message for 'analyzing' vs 'reanalyzing'
[ ] "Start New Analysis" button resets all state cleanly
```

---

## PART 7 — HOW THE FULL USER FLOW WORKS

```
User opens dashboard
│
├── Uploads photo + voice recording
├── Clicks "Analyze Stress"
│
├── Loading: "🔍 Analyzing stress indicators..."
│
├── Result panel appears:
│   ├── Score ring (animated, 0–100%)
│   ├── Level badge (Low / Moderate / High)
│   ├── Context summary ("Elevated stress detected via vocal analysis...")
│   ├── Modality bars (Face 72%, Voice 84%, Physio 61%)
│   ├── SHAP drivers ("▲ Jitter +14.2%", "▲ Brow Descent +9.1%")
│   ├── Confidence + timing row
│   └── "Start Recovery →" button (if Moderate or High)
│
├── User clicks "Start Recovery →"
│
├── Game panel appears:
│   ├── Recommended game highlighted (e.g. Breathing for High stress)
│   ├── User picks Breathing
│   │   ├── 5 breathing cycles animation
│   │   └── "✓ Breathing Complete — Re-Check Stress" button appears
│   └── User clicks done
│
├── Re-Check prompt:
│   └── "✅ Activity Complete. Let's measure if your stress changed."
│       └── User clicks "🔄 Re-Check My Stress Now"
│
├── Loading: "🔄 Re-analyzing after recovery..."
│
└── Comparison panel:
    ├── Before/After banner:
    │   "📉 Stress reduced by 23% after recovery"
    │   "Before: 78% → After: 60%"
    ├── Updated score ring (now showing 60%)
    ├── Updated modality bars
    └── "Start New Analysis" button
```
