import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../config';

const PHASES = [
    {
        key:         'silence',
        title:       'Step 1 of 3 — Silence Baseline',
        instruction: 'Please stay silent. Do not speak or make noise.',
        duration:    15,
        icon:        '🔇',
        tip:         'We are measuring your room noise level so voice features are calibrated to your environment.',
    },
    {
        key:         'voice',
        title:       'Step 2 of 3 — Voice Baseline',
        instruction: 'Read this aloud in your natural calm voice: "Today is a calm day. I am sitting comfortably. The weather is pleasant. I feel relaxed and at ease. My breathing is slow and steady."',
        duration:    40,
        icon:        '🗣️',
        tip:         'Speak naturally. This calibrates your personal pitch, tone, and speaking rhythm.',
    },
    {
        key:         'face',
        title:       'Step 3 of 3 — Face Baseline',
        instruction: 'Look at the camera with a relaxed, neutral expression. You can blink normally.',
        duration:    45,
        icon:        '😐',
        tip:         'This calibrates your personal eye openness, brow position, and jaw resting position.',
    },
];

export default function CalibrationWizard({ userId = 'default', onComplete, silenceRmsRef, onPhaseChange }) {
    const [phase,     setPhase]     = useState(0);
    const [countdown, setCountdown] = useState(PHASES[0].duration);
    const [running,   setRunning]   = useState(false);
    const [done,      setDone]      = useState(false);
    const [verification, setVerification] = useState(null);
    const [notes,     setNotes]     = useState('');
    const timerRef   = useRef(null);

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
            setCountdown(c => {
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
            setPhase(p => p + 1);
            setCountdown(PHASES[phase + 1].duration);
        } else {
            // All phases done — finalize
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
            <div style={{ maxWidth: 520, margin: '20px auto', padding: 28, background: 'var(--card-bg)', borderRadius: 16, border: 'var(--glass-border)', boxShadow: 'var(--glass-shadow)', backdropFilter: 'blur(8px)' }}>
                <div style={{ fontSize: '3rem', marginBottom: 12, textAlign: 'center' }}>⚠️</div>
                <h3 style={{ color: 'var(--text-color)', margin: '0 0 12px', fontSize: '1.4rem', textAlign: 'center' }}>Baseline Quality Check</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5', marginBottom: 20, textAlign: 'center' }}>
                    The system detected elevated stress markers or high baseline deviation during your calibration.
                </p>

                <div style={{ background: 'rgba(0,0,0,0.2)', padding: 16, borderRadius: 8, marginBottom: 20 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <span>Overall Stress Level:</span>
                        <strong style={{ color: verification.stress_probability > 0.6 ? '#ff4d4d' : '#ff9900' }}>
                            {Math.round(verification.stress_probability * 100)}%
                        </strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '0.85rem' }}>
                        <span>Face Indicator Score:</span>
                        <span>{Math.round(verification.biomarker_scores.face * 100)}%</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '0.85rem' }}>
                        <span>Voice Indicator Score:</span>
                        <span>{Math.round(verification.biomarker_scores.voice * 100)}%</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10, fontSize: '0.85rem' }}>
                        <span>Physiological Score:</span>
                        <span>{Math.round(verification.biomarker_scores.physio * 100)}%</span>
                    </div>
                    
                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 10, fontSize: '0.85rem', fontStyle: 'italic', color: 'var(--primary-color)' }}>
                        <strong>Explanation:</strong> {verification.explanation_summary}
                    </div>
                </div>

                <div style={{ marginBottom: 20 }}>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 'bold', fontSize: '0.85rem' }}>
                        Is this window really your normal, neutral state?
                    </label>
                    <textarea
                        className="form-control"
                        rows="2"
                        placeholder="Optional notes (e.g. just had coffee, felt slightly rushed)"
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        style={{ width: '100%', borderRadius: 8, padding: 8, background: 'rgba(0,0,0,0.15)', color: 'var(--text-color)', border: '1px solid rgba(255,255,255,0.15)' }}
                    />
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
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
                        className="btn btn-primary"
                        style={{ padding: 12, fontWeight: 'bold', width: '100%' }}
                    >
                        Confirm as Normal State (Use Baseline)
                    </button>
                    
                    <div style={{ display: 'flex', gap: 12 }}>
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
                            className="btn btn-secondary"
                            style={{ flex: 1, padding: 10 }}
                        >
                            🔄 Recalibrate
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
                            className="btn btn-secondary"
                            style={{ flex: 1, padding: 10, border: '1px solid #ff4d4d', color: '#ff4d4d' }}
                        >
                            ❌ Discard
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    if (done) {
        return (
            <div style={{ textAlign: 'center', padding: 40 }}>
                <div style={{ fontSize: '3rem', marginBottom: 16 }}>✅</div>
                <h2 style={{ color: 'var(--primary-color)', margin: '0 0 12px', textShadow: '0 0 10px var(--neon-glow)' }}>Calibration Complete</h2>
                <p style={{ color: 'var(--text-color)', fontSize: '0.95rem', lineHeight: '1.6', maxWidth: 440, margin: '0 auto 24px' }}>
                    Your stress monitoring is now tuned to your personal baseline and environment,
                    ensuring accurate metrics relative to your own calm state.
                </p>
                <button
                    onClick={() => onComplete && onComplete()}
                    className="btn btn-primary"
                    style={{ padding: '12px 36px', fontSize: '1.05rem' }}
                >
                    Start Monitoring
                </button>
            </div>
        );
    }

    return (
        <div style={{ maxWidth: 520, margin: '20px auto', padding: 28, background: 'var(--card-bg)', borderRadius: 16, border: 'var(--glass-border)', boxShadow: 'var(--glass-shadow)', backdropFilter: 'blur(8px)' }}>
            {/* Phase indicator */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 28 }}>
                {PHASES.map((p, i) => (
                    <div key={p.key} style={{
                        flex: 1, height: 4, borderRadius: 2,
                        background: i < phase ? 'var(--primary-color)' : i === phase && running ? 'linear-gradient(90deg, var(--primary-color), var(--accent-light-bg))' : 'var(--bar-bg)',
                        boxShadow: i <= phase && running ? '0 0 8px var(--neon-glow)' : 'none',
                        transition: 'background 0.3s ease'
                    }} />
                ))}
            </div>

            <div style={{ textAlign: 'center', marginBottom: 28 }}>
                <div style={{ fontSize: '3rem', marginBottom: 12, filter: 'drop-shadow(0 0 10px var(--neon-glow))' }}>{currentPhase.icon}</div>
                <h3 style={{ color: 'var(--text-color)', margin: '0 0 12px', fontSize: '1.4rem' }}>{currentPhase.title}</h3>
                <p style={{ color: 'var(--primary-color)', fontSize: '1.05rem', lineHeight: '1.5', margin: '0 0 16px', fontWeight: 500 }}>
                    {currentPhase.instruction}
                </p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontStyle: 'italic', maxWidth: 400, margin: '0 auto' }}>
                    {currentPhase.tip}
                </p>
            </div>

            {/* Progress ring */}
            {running && (
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 24 }}>
                    <svg width={110} height={110} viewBox="0 0 100 100">
                        <circle cx={50} cy={50} r={42} fill="none" stroke="var(--bar-bg)" strokeWidth={6} />
                        <circle cx={50} cy={50} r={42} fill="none" stroke="var(--primary-color)" strokeWidth={6}
                            strokeDasharray={`${2 * Math.PI * 42}`}
                            strokeDashoffset={`${2 * Math.PI * 42 * (1 - pct / 100)}`}
                            strokeLinecap="round"
                            transform="rotate(-90 50 50)"
                            style={{ transition: 'stroke-dashoffset 0.1s linear', filter: 'drop-shadow(0 0 6px var(--neon-glow))' }}
                        />
                        <text x={50} y={56} textAnchor="middle" fill="var(--primary-color)"
                               fontSize={22} fontWeight="bold" fontFamily="monospace">{countdown}s</text>
                    </svg>
                </div>
            )}

            {!running && (
                <button
                    onClick={startPhase}
                    className="btn btn-primary"
                    style={{
                        display: 'block', width: '100%',
                        padding: '14px', borderRadius: 8,
                        fontSize: '1.1rem', fontWeight: 'bold'
                    }}
                >
                    {phase === 0 ? 'Begin Calibration' : `Start Step ${phase + 1}`}
                </button>
            )}
        </div>
    );
}
