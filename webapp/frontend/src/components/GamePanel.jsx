import React, { useState, useEffect, useCallback } from 'react';

const PHASES = {
  inhale:  { duration: 4, next: 'hold',   label: 'Breathe In',  color: '#64B5F6' },
  hold:    { duration: 4, next: 'exhale',  label: 'Hold',        color: '#CE93D8' },
  exhale:  { duration: 6, next: 'inhale',  label: 'Breathe Out', color: '#80CBC4' },
};

// ── Game 1: Breathing ─────────────────────────────────────────────────────────
function BreathingGame({ onComplete }) {
  const [phase,  setPhase]  = useState('inhale');
  const [count,  setCount]  = useState(4);
  const [cycles, setCycles] = useState(0);
  const TARGET_CYCLES = 5;  // minimum 5 full cycles ≈ 70 seconds

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
    <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-color)' }}>
      <h3 style={{ color: '#64B5F6', marginBottom: 4 }}>Guided Breathing</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 20 }}>
        5 complete cycles · {Math.max(0, TARGET_CYCLES - cycles)} remaining
      </p>

      {/* Animated ring */}
      <div style={{ position: 'relative', display: 'inline-block', marginBottom: 20 }}>
        <svg width={160} height={160} viewBox="0 0 160 160">
          <circle cx={80} cy={80} r={radius} fill="none"
            stroke="var(--bar-bg)" strokeWidth={10} />
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
            fontFamily="inherit">{count}</text>
          <text x={80} y={96} textAnchor="middle"
            fill="var(--text-muted)" fontSize={12}
            fontFamily="inherit">{currentPhase.label}</text>
        </svg>
      </div>

      <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginBottom: 20 }}>
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
    <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-color)' }}>
      <h3 style={{ color: '#CE93D8', marginBottom: 4 }}>Focus Tap</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 16 }}>
        Tap the button {TARGET} times in a steady rhythm
      </p>

      {/* Progress bar */}
      <div style={{ height: 6, borderRadius: 3,
                    background: 'var(--bar-bg)', margin: '0 auto 20px',
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
        <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem',
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
  }, [remaining, started, onComplete]);

  const pct  = ((DURATION - remaining) / DURATION) * 100;
  const mins = Math.floor(remaining / 60);
  const secs = remaining % 60;

  return (
    <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-color)' }}>
      <h3 style={{ color: '#80CBC4', marginBottom: 4 }}>Calm Mode</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 20 }}>
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

      <div style={{ height: 4, borderRadius: 2, background: 'var(--bar-bg)',
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
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
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
    <div style={{ padding: 24, color: 'var(--text-color)' }}>
      <h3 style={{ color: '#FFD54F', marginBottom: 4 }}>Gratitude Reflection</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 20 }}>
        Write three things. Research shows this reframes the amygdala's threat response.
      </p>

      {prompts.map((prompt, i) => (
        <div key={i} style={{ marginBottom: 14 }}>
          <div style={{
            color: 'var(--text-muted)', fontSize: '0.74rem',
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
              background: 'var(--bar-bg)',
              border: `1px solid ${focused === i ? '#FFD54F66' : 'var(--glass-border)'}`,
              borderRadius: 8, padding: '10px 12px',
              color: 'var(--text-color)', fontSize: '0.88rem',
              resize: 'none', outline: 'none',
              transition: 'border-color 0.2s',
              fontFamily: 'inherit',
            }}
          />
        </div>
      ))}

      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginTop: 4 }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
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
    <div style={{ padding: 24, color: 'var(--text-color)' }}>
      <h3 style={{ color: '#A5D6A7', marginBottom: 4 }}>Posture Reset</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 18 }}>
        Check each step as you complete it. Each has a physiological basis.
      </p>

      {steps.map((step, i) => (
        <div
          key={i}
          onClick={() => toggle(i)}
          style={{
            display: 'flex', alignItems: 'flex-start', gap: 12,
            padding: '10px 12px', borderRadius: 8, marginBottom: 6,
            background: checked[i] ? 'rgba(165,214,167,0.08)' : 'var(--bar-bg)',
            border: `1px solid ${checked[i] ? '#A5D6A744' : 'var(--glass-border)'}`,
            cursor: 'pointer', transition: 'all 0.2s',
          }}
        >
          <div style={{
            width: 20, height: 20, borderRadius: 4, flexShrink: 0, marginTop: 1,
            background: checked[i] ? '#A5D6A7' : 'var(--bar-bg)',
            border: `1px solid ${checked[i] ? '#A5D6A7' : 'var(--glass-border)'}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '0.75rem', color: '#1a1a1a', fontWeight: 700,
            transition: 'all 0.2s',
          }}>
            {checked[i] ? '✓' : ''}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{
              color: checked[i] ? 'var(--text-muted)' : 'var(--text-color)',
              fontSize: '0.85rem',
              textDecoration: checked[i] ? 'line-through' : 'none',
              marginBottom: 2,
            }}>
              {step.icon} {step.text}
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
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
          <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
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
  background: 'var(--bar-bg)',
  border: 'var(--glass-border)',
  color: 'var(--text-muted)',
  borderRadius: 8, padding: '11px 22px',
  cursor: 'not-allowed', fontWeight: 700,
  fontSize: '0.88rem', marginTop: 8,
};

// ── Game Selector + Container ─────────────────────────────────────────────────

const GAMES = [
  { key: 'breathing', label: 'Breathing',   icon: '🫁', color: '#64B5F6',
    desc: '5 cycles · ~70 seconds',
    component: BreathingGame,
    recommended: ['High', 'Extreme'] },
  { key: 'focus',     label: 'Focus Tap',   icon: '👆', color: '#CE93D8',
    desc: '30 taps · ~1 minute',
    component: FocusTapGame,
    recommended: ['Moderate', 'High', 'Extreme'] },
  { key: 'calm',      label: 'Calm Timer',  icon: '⏱', color: '#80CBC4',
    desc: '2 minutes silence',
    component: CalmTimer,
    recommended: ['High', 'Extreme'] },
  { key: 'gratitude', label: 'Gratitude',   icon: '🙏', color: '#FFD54F',
    desc: '3 reflections',
    component: GratitudeGame,
    recommended: ['Moderate'] },
  { key: 'posture',   label: 'Posture',     icon: '💺', color: '#A5D6A7',
    desc: '6-step checklist',
    component: PostureReset,
    recommended: ['Moderate', 'High', 'Extreme'] },
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
        <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-color)' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>✅</div>
          <h3 style={{ color: '#4CAF50', marginBottom: 8 }}>
            Activity Complete
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem',
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
                color: 'var(--text-muted)', cursor: 'pointer',
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
      <>
        {/* Navigation Bar outside the main box */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12,
          padding: '0 8px'
        }}>
          <button
            onClick={() => setSelected(null)}
            className="btn btn-secondary"
            style={{
              padding: '6px 16px',
              fontSize: '0.82rem',
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            ← Back to Activities
          </button>
          
          <span style={{
            color: 'var(--text-color)',
            fontWeight: 700,
            fontSize: '0.9rem',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
            display: 'flex',
            alignItems: 'center',
            gap: 6
          }}>
            <span style={{ color: game.color }}>{game.icon}</span>
            <span>{game.label}</span>
          </span>

          <button
            onClick={onDismiss}
            className="btn btn-secondary"
            style={{
              padding: '6px 16px',
              fontSize: '0.82rem',
              borderRadius: '20px',
              fontWeight: 600,
              cursor: 'pointer',
              borderColor: '#c74545',
            }}
          >
            Close Game Panel
          </button>
        </div>

        <div style={PANEL_WRAPPER}>
          <GameComponent onComplete={() => handleComplete(selected)} />
        </div>
      </>
    );
  }

  return (
    <>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
        padding: '0 8px'
      }}>
        <h3 style={{ color: 'var(--text-color)', margin: 0, fontSize: '1.25rem', fontWeight: 700 }}>
          🧘 Recovery Activities
        </h3>
        <button onClick={onDismiss}
          className="btn btn-secondary"
          style={{
            padding: '6px 16px',
            fontSize: '0.82rem',
            borderRadius: '20px',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          Close Panel
        </button>
      </div>

      <div style={PANEL_WRAPPER}>
        <div style={{ padding: '16px 20px 12px' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', margin: 0 }}>
            {stressLevel === 'High' || stressLevel === 'Extreme'
              ? `${stressLevel} stress detected. Any of these activities will help right now.`
              : 'Take a short reset to improve clarity and calm your nervous system.'}
          </p>
        </div>

      <div style={{ padding: '0 16px 16px',
                    display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
        {GAMES.map(game => {
          const isRecommended = stressLevel ? game.recommended.includes(stressLevel) : true;
          const isDone        = completed.includes(game.key);
          return (
            <button key={game.key}
              onClick={() => setSelected(game.key)}
              style={{
                background: isDone
                  ? 'rgba(76,175,80,0.08)'
                  : isRecommended
                    ? `linear-gradient(135deg, ${game.color}11, transparent)`
                    : 'var(--card-inner-bg)',
                border: `1px solid ${isDone ? '#4CAF5044' : isRecommended ? game.color + '55' : 'var(--glass-border)'}`,
                borderRadius: 10, padding: '14px 12px',
                cursor: 'pointer', textAlign: 'left',
                transition: 'all 0.2s',
                boxShadow: 'var(--glass-shadow)',
                color: 'var(--text-color)'
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
              <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
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
    </>
  );
}

const PANEL_WRAPPER = {
  background: 'var(--card-bg)',
  border: 'var(--glass-border)',
  borderRadius: 16,
  boxShadow: 'var(--glass-shadow)',
  backdropFilter: 'blur(12px)',
  overflow: 'hidden',
  marginBottom: 24
};
