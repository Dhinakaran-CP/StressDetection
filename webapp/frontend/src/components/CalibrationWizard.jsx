import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../config';

const PHASES = [
  {
    key: 'silence',
    title: 'Step 1 of 3 — Silence Baseline',
    instruction: 'Please stay silent. Do not speak or make noise.',
    duration: 15,
    icon: 'notifications_off',
    tip: 'We are measuring your room noise level so voice features are calibrated to your environment.',
  },
  {
    key: 'voice',
    title: 'Step 2 of 3 — Voice Baseline',
    instruction: 'Read this aloud in your natural calm voice: "Today is a calm day. I am sitting comfortably. The weather is pleasant. I feel relaxed and at ease. My breathing is slow and steady."',
    duration: 40,
    icon: 'settings_voice',
    tip: 'Speak naturally. This calibrates your personal pitch, tone, and speaking rhythm.',
  },
  {
    key: 'face',
    title: 'Step 3 of 3 — Face Baseline',
    instruction: 'Look at the camera with a relaxed, neutral expression. You can blink normally.',
    duration: 45,
    icon: 'face',
    tip: 'This calibrates your personal eye openness, brow position, and jaw resting position.',
  },
];

export default function CalibrationWizard({ userId = 'default', onComplete, silenceRmsRef, onPhaseChange }) {
  const [phase, setPhase] = useState(0);
  const [countdown, setCountdown] = useState(PHASES[0].duration);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [verification, setVerification] = useState(null);
  const [notes, setNotes] = useState('');
  const timerRef = useRef(null);

  const currentPhase = PHASES[phase];

  useEffect(() => {
    if (onPhaseChange) {
      onPhaseChange(running ? currentPhase.key : 'idle');
    }
  }, [phase, running, onPhaseChange, currentPhase]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startPhase = () => {
    setRunning(true);
    setCountdown(currentPhase.duration);

    if (currentPhase.key === 'silence' && silenceRmsRef) {
      silenceRmsRef.current = [];
    }

    timerRef.current = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) {
          clearInterval(timerRef.current);
          handlePhaseComplete();
          return 0;
        }
        return c - 1;
      });
    }, 1000);
  };

  const handlePhaseComplete = async () => {
    setRunning(false);

    if (currentPhase.key === 'silence' && silenceRmsRef && silenceRmsRef.current.length > 0) {
      const noiseRms = silenceRmsRef.current.reduce((a, b) => a + b, 0) / silenceRmsRef.current.length;
      try {
        await fetch(`${API_BASE}/api/calibrate/silence`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, noise_rms: noiseRms }),
        });
      } catch (err) {
        console.error("Failed to post silence calibration:", err);
      }
    }

    if (phase < PHASES.length - 1) {
      setPhase((p) => p + 1);
      setCountdown(PHASES[phase + 1].duration);
    } else {
      // Finalize calibration
      try {
        const res = await fetch(`${API_BASE}/api/calibrate/finalize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId }),
        });
        const data = await res.json();

        if (data.verification && data.verification.recommendation === 'NEEDS_CONFIRMATION') {
          setVerification(data.verification);
        } else {
          setDone(true);
          if (onComplete) onComplete(data.calibration);
        }
      } catch (err) {
        console.error("Failed to finalize calibration:", err);
        setDone(true);
        if (onComplete) onComplete(null);
      }
    }
  };

  const pct = Math.round(((currentPhase.duration - countdown) / currentPhase.duration) * 100);

  if (verification) {
    return (
      <div className="w-full max-w-[520px] mx-auto p-8 bg-surface rounded-[24px] border border-outline-variant/30 shadow-[0_8px_32px_0_rgba(0,0,0,0.05)] text-on-surface select-none">
        <div className="text-error flex justify-center mb-6">
          <span className="material-symbols-outlined text-[64px]">warning</span>
        </div>
        <h3 className="font-headline-sm text-headline-sm text-center text-primary mb-2">Baseline Quality Check</h3>
        <p className="text-on-surface-variant text-sm text-center leading-relaxed mb-6">
          The system detected elevated stress markers or high baseline deviation during your calibration.
        </p>

        <div className="bg-surface-container-low p-5 rounded-2xl border border-outline-variant/20 mb-6 space-y-3.5">
          <div className="flex justify-between items-center text-sm font-semibold">
            <span className="text-on-surface-variant">Overall Stress Level:</span>
            <strong className={verification.stress_probability > 0.6 ? 'text-error' : 'text-[#854D0E]'}>
              {Math.round(verification.stress_probability * 100)}%
            </strong>
          </div>
          
          <div className="h-[1px] bg-outline-variant/20 w-full"></div>

          <div className="space-y-2 text-xs font-semibold text-on-surface-variant">
            <div className="flex justify-between">
              <span>Face Indicator Score:</span>
              <span>{Math.round(verification.biomarker_scores?.face * 100 || 0)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Voice Indicator Score:</span>
              <span>{Math.round(verification.biomarker_scores?.voice * 100 || 0)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Physiological Score:</span>
              <span>{Math.round(verification.biomarker_scores?.physio * 100 || 0)}%</span>
            </div>
          </div>
          
          <div className="pt-3 border-t border-outline-variant/20 text-[11px] font-medium italic text-primary">
            <strong>Explanation:</strong> {verification.explanation_summary}
          </div>
        </div>

        <div className="mb-6 space-y-2">
          <label className="block text-xs font-bold uppercase tracking-wider text-outline px-1">
            Is this really your normal, neutral state?
          </label>
          <textarea
            className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all placeholder:text-outline-variant text-sm text-on-surface"
            rows="2"
            placeholder="Optional notes (e.g. just had coffee, felt slightly rushed)"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>

        <div className="flex flex-col gap-3">
          <button
            onClick={async () => {
              try {
                const res = await fetch(`${API_BASE}/api/calibrate/confirm`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ user_id: userId, action: 'accept_low_confidence', notes }),
                });
                const data = await res.json();
                setDone(true);
                setVerification(null);
                if (onComplete) onComplete(data.calibration);
              } catch (e) {
                console.error(e);
              }
            }}
            className="w-full bg-primary text-on-primary py-3.5 rounded-xl font-semibold shadow-sm hover:opacity-90 active:scale-[0.98] transition-all"
          >
            Confirm as Normal State
          </button>
          
          <div className="flex gap-3">
            <button
              onClick={async () => {
                try {
                  await fetch(`${API_BASE}/api/calibrate/confirm`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, action: 'recalibrate' }),
                  });
                  setPhase(0);
                  setCountdown(PHASES[0].duration);
                  setRunning(false);
                  setVerification(null);
                  setNotes('');
                } catch (e) {
                  console.error(e);
                }
              }}
              className="flex-1 border border-outline-variant text-on-surface-variant hover:bg-surface-container-low py-3 rounded-xl font-semibold transition-colors flex items-center justify-center gap-1.5"
            >
              <span className="material-symbols-outlined text-[16px]">autorenew</span>
              Recalibrate
            </button>
            <button
              onClick={async () => {
                try {
                  await fetch(`${API_BASE}/api/calibrate/confirm`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, action: 'discard' }),
                  });
                  setDone(true);
                  setVerification(null);
                  if (onComplete) onComplete(null);
                } catch (e) {
                  console.error(e);
                }
              }}
              className="flex-1 border border-error text-error hover:bg-error-container/10 py-3 rounded-xl font-semibold transition-colors flex items-center justify-center gap-1.5"
            >
              <span className="material-symbols-outlined text-[16px]">close</span>
              Discard
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (done) {
    return (
      <div className="w-full max-w-[480px] mx-auto text-center p-8 bg-surface rounded-[24px] border border-outline-variant/30 shadow-[0_8px_32px_0_rgba(0,0,0,0.05)] text-on-surface select-none">
        <div className="text-primary flex justify-center mb-6">
          <span className="material-symbols-outlined text-[72px]" style={{ fontVariationSettings: "'FILL' 1" }}>verified</span>
        </div>
        <h2 className="font-headline-sm text-headline-sm text-primary mb-3">Calibration Complete</h2>
        <p className="text-on-surface-variant text-sm leading-relaxed max-w-sm mx-auto mb-8">
          Your stress monitoring is now tuned to your personal baseline and environment,
          ensuring accurate metrics relative to your own calm state.
        </p>
        <button
          onClick={() => onComplete && onComplete()}
          className="w-full bg-primary text-on-primary py-3.5 rounded-xl font-semibold shadow-sm hover:opacity-90 active:scale-[0.98] transition-all"
        >
          Start Monitoring
        </button>
      </div>
    );
  }

  return (
    <div className="w-full max-w-[520px] mx-auto p-8 bg-surface rounded-[24px] border border-outline-variant/30 shadow-[0_8px_32px_0_rgba(0,0,0,0.05)] text-on-surface select-none">
      {/* Progress timeline */}
      <div className="flex gap-2 mb-8">
        {PHASES.map((p, i) => (
          <div
            key={p.key}
            className={`h-1.5 flex-1 rounded-full transition-all duration-500 ${
              i < phase
                ? 'bg-primary'
                : i === phase && running
                ? 'bg-gradient-to-r from-primary to-primary-container animate-pulse shadow-sm shadow-primary/45'
                : 'bg-surface-container-high'
            }`}
          />
        ))}
      </div>

      <div className="text-center mb-8">
        <div className="flex justify-center mb-4 text-primary">
          <span className="material-symbols-outlined text-[56px]">{currentPhase.icon}</span>
        </div>
        <h3 className="font-headline-sm text-headline-sm text-primary mb-3">{currentPhase.title}</h3>
        <p className="text-primary-container font-semibold text-sm leading-relaxed mb-4 max-w-xs mx-auto">
          {currentPhase.instruction}
        </p>
        <p className="text-on-surface-variant text-xs italic max-w-xs mx-auto">
          {currentPhase.tip}
        </p>
      </div>

      {/* Progress ring countdown */}
      {running && (
        <div className="flex justify-center mb-6">
          <div className="relative w-28 h-28 flex items-center justify-center">
            <svg className="w-full h-full -rotate-90">
              <circle cx="56" cy="56" fill="transparent" r="48" stroke="var(--outline-variant)" strokeOpacity="0.2" strokeWidth="6"></circle>
              <circle cx="56" cy="56" fill="transparent" r="48" stroke="#0e3b69" strokeDasharray="301.6" strokeDashoffset={`${301.6 * (1 - pct / 100)}`} strokeLinecap="round" strokeWidth="6" className="transition-all duration-100 linear"></circle>
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="font-data-metric text-2xl text-primary font-bold">{countdown}s</span>
            </div>
          </div>
        </div>
      )}

      {!running && (
        <button
          onClick={startPhase}
          className="w-full bg-primary text-on-primary py-4 rounded-xl font-bold text-sm tracking-wider uppercase shadow-md hover:opacity-90 active:scale-[0.98] transition-all"
        >
          {phase === 0 ? 'Begin Calibration' : `Start Step ${phase + 1}`}
        </button>
      )}
    </div>
  );
}
