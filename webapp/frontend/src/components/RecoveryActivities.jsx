import React, { useState, useEffect, useRef } from 'react';

const BREATHING_RHYTHMS = [
  { id: 'box', name: 'Box Breathing (4-4-4-4)', inhale: 4, hold1: 4, exhale: 4, hold2: 4, description: 'Stabilize sympathetic nervous system' },
  { id: 'calm', name: 'Deep Calm (4-2-6-0)', inhale: 4, hold1: 2, exhale: 6, hold2: 0, description: 'Soothe acute anxiety surges' },
  { id: 'sleep', name: 'Sleep Wake (4-7-8-0)', inhale: 4, hold1: 7, exhale: 8, hold2: 0, description: 'Induce rapid parasympathetic rest' }
];

export default function RecoveryActivities() {
  const [calmStreak, setCalmStreak] = useState(12);

  // Breathing State Machine
  const [selectedRhythm, setSelectedRhythm] = useState(BREATHING_RHYTHMS[2]); // 4-7-8 by default
  const [breathingRunning, setBreathingRunning] = useState(false);
  const [breathPhase, setBreathPhase] = useState('inhale');
  const [breathCountdown, setBreathCountdown] = useState(selectedRhythm.inhale);
  const breathTimerRef = useRef(null);

  // Focus Tap Game State
  const [tapCount, setTapCount] = useState(0);
  const [tapStreak, setTapStreak] = useState(0);
  const [lastTapTime, setLastTapTime] = useState(null);
  const [tapFeedback, setTapFeedback] = useState('');

  const handleTap = () => {
    const now = Date.now();
    const elapsed = lastTapTime ? now - lastTapTime : 999;
    setLastTapTime(now);

    let newStreak = tapStreak;
    if (elapsed > 400 && elapsed < 600) { // roughly 120 bpm = 2 beats per second = 500ms
      newStreak += 1;
      setTapFeedback('🔥 Perfect!');
    } else {
      newStreak = 0;
      setTapFeedback(elapsed < 400 ? 'Too Fast' : 'Too Slow');
    }
    setTapStreak(newStreak);
    setTapCount(c => c + 1);
    
    setTimeout(() => setTapFeedback(''), 1500);
  };

  // Calm Timer State
  const [timerRunning, setTimerRunning] = useState(false);
  const [timeLeft, setTimeLeft] = useState(120); // 2 minutes

  useEffect(() => {
    let interval;
    if (timerRunning && timeLeft > 0) {
      interval = setInterval(() => {
        setTimeLeft(t => t - 1);
      }, 1000);
    } else if (timeLeft === 0) {
      setTimerRunning(false);
    }
    return () => clearInterval(interval);
  }, [timerRunning, timeLeft]);

  const toggleTimer = () => {
    if (timeLeft === 0) setTimeLeft(120); // reset if finished
    setTimerRunning(!timerRunning);
  };

  // Posture Reset Checklist
  const [postureChecked, setPostureChecked] = useState(Array(3).fill(false));
  const postureSteps = [
    { text: 'Uncross legs' },
    { text: 'Roll shoulders back' },
    { text: 'Level eyes to screen' },
  ];

  const handlePostureCheck = (index) => {
    const updated = [...postureChecked];
    updated[index] = !updated[index];
    setPostureChecked(updated);

    if (updated.every(Boolean)) {
      setCalmStreak(s => s + 1);
      setTimeout(() => setPostureChecked(Array(3).fill(false)), 2000); // reset after celebration
    }
  };

  // Gratitude State
  const [gratitudeText, setGratitudeText] = useState("");
  const [showHeart, setShowHeart] = useState(false);

  useEffect(() => {
    if (gratitudeText.length > 0) {
      setShowHeart(true);
      const t = setTimeout(() => setShowHeart(false), 2000);
      return () => clearTimeout(t);
    }
  }, [gratitudeText]);

  // --- Breathing Timer ---
  useEffect(() => {
    if (!breathingRunning) {
      if (breathTimerRef.current) clearInterval(breathTimerRef.current);
      return;
    }

    breathTimerRef.current = setInterval(() => {
      setBreathCountdown((prev) => {
        if (prev <= 1) {
          let nextPhase = 'inhale';
          let duration = selectedRhythm.inhale;

          if (breathPhase === 'inhale') {
            if (selectedRhythm.hold1 > 0) {
              nextPhase = 'hold1';
              duration = selectedRhythm.hold1;
            } else {
              nextPhase = 'exhale';
              duration = selectedRhythm.exhale;
            }
          } else if (breathPhase === 'hold1') {
            nextPhase = 'exhale';
            duration = selectedRhythm.exhale;
          } else if (breathPhase === 'exhale') {
            if (selectedRhythm.hold2 > 0) {
              nextPhase = 'hold2';
              duration = selectedRhythm.hold2;
            } else {
              nextPhase = 'inhale';
              duration = selectedRhythm.inhale;
            }
          } else if (breathPhase === 'hold2') {
            nextPhase = 'inhale';
            duration = selectedRhythm.inhale;
          }

          setBreathPhase(nextPhase);
          return duration;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(breathTimerRef.current);
  }, [breathingRunning, breathPhase, selectedRhythm]);


  let scale = 1.0;
  if (breathingRunning) {
    if (breathPhase === 'inhale') {
      const elapsed = selectedRhythm.inhale - breathCountdown;
      scale = 1.0 + (elapsed / selectedRhythm.inhale) * 0.45;
    } else if (breathPhase === 'hold1') {
      scale = 1.45;
    } else if (breathPhase === 'exhale') {
      scale = 1.45 - ( (selectedRhythm.exhale - breathCountdown) / selectedRhythm.exhale ) * 0.45;
    }
  }

  return (
    <div className="space-y-8 max-w-5xl">
      <div className="space-y-2">
        <h1 className="text-[32px] font-headline-sm font-bold text-primary">Recovery Game Panel</h1>
        <p className="text-on-surface-variant text-[15px] font-medium leading-relaxed max-w-3xl">
          A curated selection of micro-activities designed to reset your parasympathetic nervous system and restore cognitive focus.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
        
        {/* Breathing Exercise - Span 2 cols, Span 2 rows */}
        <div className="col-span-1 md:col-span-2 md:row-span-2 bg-surface-container-lowest rounded-3xl p-8 border border-outline-variant/10 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-primary-container/20 text-primary flex items-center justify-center">
                <span className="material-symbols-outlined text-xl">air</span>
              </div>
              <span className="font-label-caps text-[10px] font-bold text-primary tracking-widest uppercase">Deep Reset</span>
            </div>
            
            <h3 className="font-headline-sm text-2xl text-primary font-semibold mb-2">Breathing Exercise</h3>
            <p className="text-on-surface-variant text-sm font-medium leading-relaxed max-w-sm">
              Synchronize your breath with the visual ring to trigger physiological calm in under 60 seconds.
            </p>
          </div>

          <div className="flex-1 flex flex-col items-center justify-center py-10">
            <div className="relative w-48 h-48 flex items-center justify-center mb-6">
              <div
                className="absolute inset-0 rounded-full border border-primary/20 bg-gradient-to-tr from-primary/10 to-transparent transition-all duration-[1s] ease-in-out"
                style={{ transform: `scale(${scale})` }}
              />
              <div
                className="absolute w-36 h-36 rounded-full border border-primary/40 bg-white/50 backdrop-blur-sm shadow-inner flex flex-col items-center justify-center transition-all duration-[1s] ease-in-out z-10"
                style={{ transform: `scale(${scale * 0.95})` }}
              >
                {breathingRunning ? (
                  <>
                    <span className="font-data-metric text-4xl text-primary font-bold">{breathCountdown}</span>
                    <span className="font-label-caps text-[10px] text-primary uppercase font-bold tracking-wider mt-1">
                      {breathPhase === 'inhale' && 'Breathe In'}
                      {breathPhase === 'hold1' && 'Hold'}
                      {breathPhase === 'exhale' && 'Breathe Out'}
                      {breathPhase === 'hold2' && 'Rest'}
                    </span>
                  </>
                ) : (
                  <span className="material-symbols-outlined text-4xl text-primary/30">self_improvement</span>
                )}
              </div>
            </div>
            <div className="font-data-metric text-lg text-primary font-bold mb-4">
              4-7-8 Rhythm
            </div>
            <button
              onClick={() => setBreathingRunning(!breathingRunning)}
              className="w-full max-w-xs py-3 rounded-xl border border-outline-variant/30 text-on-surface text-sm font-bold shadow-sm hover:bg-surface-container-low transition-all"
            >
              {breathingRunning ? 'Stop Session' : 'Start Session'}
            </button>
          </div>
        </div>

        {/* Focus Tap Game */}
        <div className="col-span-1 bg-surface-container-lowest rounded-3xl p-6 border border-outline-variant/10 shadow-sm flex flex-col relative overflow-hidden h-full">
          <div>
            <div className="flex justify-between items-start mb-4">
              <div className="w-10 h-10 rounded-full bg-surface-container-low flex items-center justify-center border border-outline-variant/10 text-on-surface-variant">
                <span className="material-symbols-outlined text-[20px]">touch_app</span>
              </div>
              <span className="font-data-metric font-bold text-outline-variant/50 uppercase tracking-widest text-xs">120 BPM</span>
            </div>
            <h3 className="font-headline-sm text-lg text-primary font-semibold mb-1">Focus Tap</h3>
            <p className="text-on-surface-variant text-[13px] font-medium leading-relaxed">
              Align your motor skills with a rhythmic pulse.
            </p>
          </div>
          <div className="flex-1 flex flex-col items-center justify-center pt-4">
            <button
              onClick={handleTap}
              className="relative w-24 h-24 rounded-full bg-primary-container/10 border-2 border-primary/20 flex flex-col items-center justify-center group active:scale-95 transition-all cursor-pointer overflow-hidden shadow-sm hover:shadow-md"
            >
              <div className="absolute inset-0 bg-primary/10 rounded-full scale-0 group-active:scale-100 transition-transform duration-150"></div>
              <span className="font-data-metric text-4xl font-bold text-primary z-10">{tapCount}</span>
              <span className="font-label-caps text-[8px] text-primary font-bold uppercase tracking-widest z-10">Taps</span>
            </button>
            <div className="h-4 mt-3 text-center">
              <span className={`text-[10px] font-bold font-label-caps uppercase tracking-widest transition-all ${tapFeedback.includes('Perfect') ? 'text-[#4ADE80] animate-pulse' : 'text-error/80'}`}>
                {tapFeedback}
              </span>
            </div>
          </div>
        </div>

        {/* Calm Timer */}
        <div className="col-span-1 bg-surface-container-lowest rounded-3xl p-6 border border-outline-variant/10 shadow-sm flex flex-col relative overflow-hidden h-full">
           <div>
             <div className="flex justify-between items-start mb-4">
              <div className="w-10 h-10 rounded-full bg-primary-container/10 flex items-center justify-center text-primary">
                <span className="material-symbols-outlined text-[20px]">{timerRunning ? 'pause' : 'timer'}</span>
              </div>
              <span className="font-data-metric font-bold text-primary/40 uppercase tracking-widest text-xs">02:00</span>
            </div>
            <h3 className="font-headline-sm text-lg text-primary font-semibold mb-1">Calm Timer</h3>
            <p className="text-on-surface-variant text-[13px] font-medium leading-relaxed">
              Two minutes of absolute digital stillness.
            </p>
          </div>
          <div className="flex-1 flex flex-col items-center justify-center pt-4">
            <div className="relative w-28 h-28 flex items-center justify-center">
              {/* Progress Ring */}
              <svg className="absolute inset-0 w-full h-full -rotate-90">
                <circle cx="56" cy="56" r="52" fill="none" stroke="currentColor" strokeWidth="4" className="text-outline-variant/10" />
                <circle 
                  cx="56" cy="56" r="52" fill="none" stroke="currentColor" strokeWidth="4" 
                  className="text-primary transition-all duration-1000 ease-linear"
                  strokeDasharray="326"
                  strokeDashoffset={326 - (326 * (timeLeft / 120))}
                />
              </svg>
              <div className="text-center">
                <div className="font-data-metric text-3xl font-bold text-primary">
                  {String(Math.floor(timeLeft / 60)).padStart(2, '0')}:{String(timeLeft % 60).padStart(2, '0')}
                </div>
              </div>
            </div>
            <button
              onClick={toggleTimer}
              className={`mt-4 px-6 py-2 rounded-xl border text-xs font-bold font-label-caps uppercase tracking-wider transition-all active:scale-95 ${timerRunning ? 'border-error/20 text-error hover:bg-error-container/10' : 'border-primary/20 text-primary hover:bg-primary-container/10'}`}
            >
              {timerRunning ? 'Pause' : timeLeft === 0 ? 'Restart' : 'Start'}
            </button>
          </div>
        </div>

        {/* Gratitude Reflection */}
        <div className="col-span-1 md:col-span-2 bg-[#F0F6FF] rounded-3xl p-6 border border-[#D9E6FA] shadow-sm flex flex-col justify-center relative overflow-hidden">
          {showHeart && (
            <span className="absolute right-12 top-8 text-4xl text-[#FF6B6B] animate-ping opacity-50" style={{ fontVariationSettings: "'FILL' 1" }}>
              ♥
            </span>
          )}
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 rounded-2xl bg-white shadow-sm text-primary flex items-center justify-center flex-shrink-0 relative">
              <span className="material-symbols-outlined text-xl transition-all" style={{ fontVariationSettings: "'FILL' 1", transform: showHeart ? 'scale(1.2)' : 'scale(1)' }}>favorite</span>
            </div>
            <div>
              <h3 className="font-headline-sm text-xl text-primary font-semibold mb-0.5">Gratitude Reflection</h3>
              <p className="text-on-surface-variant text-xs font-medium">Type one thing that went well today.</p>
            </div>
          </div>
          <input 
            type="text" 
            placeholder="Today, I am grateful for..."
            value={gratitudeText}
            onChange={(e) => setGratitudeText(e.target.value)}
            className="w-full bg-white border border-[#D9E6FA] rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 text-on-surface font-medium placeholder:italic placeholder:text-outline-variant transition-all"
          />
        </div>

        {/* Posture Reset */}
        <div className="col-span-1 bg-surface-container-lowest rounded-3xl p-6 border border-outline-variant/10 shadow-sm flex flex-col relative overflow-hidden">
          {postureChecked.every(Boolean) && (
            <div className="absolute inset-0 bg-[#4ADE80]/10 z-0 animate-pulse"></div>
          )}
          <div className="flex justify-between items-center mb-4 relative z-10">
            <h3 className="font-headline-sm text-lg text-primary font-semibold">Posture Reset</h3>
            <span className="material-symbols-outlined text-outline-variant">checklist</span>
          </div>
          <div className="space-y-3 relative z-10">
            {postureSteps.map((step, idx) => {
              const checked = postureChecked[idx];
              return (
                <div 
                  key={idx} 
                  onClick={() => handlePostureCheck(idx)}
                  className="flex items-center gap-3 cursor-pointer group"
                >
                  <div className={`w-5 h-5 rounded flex items-center justify-center border transition-all ${checked ? 'bg-[#4ADE80]/20 border-[#4ADE80] text-[#4ADE80]' : 'bg-white border-outline-variant/40 group-hover:border-primary/50'}`}>
                    {checked && <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 700" }}>check</span>}
                  </div>
                  <span className={`text-[13px] font-medium transition-colors ${checked ? 'text-on-surface line-through opacity-70' : 'text-on-surface-variant group-hover:text-primary'}`}>
                    {step.text}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Current Streak */}
        <div className="col-span-1 bg-surface-container-lowest rounded-3xl p-6 border border-outline-variant/10 shadow-sm flex flex-col justify-between">
          <div>
            <span className="font-label-caps text-[10px] font-bold text-outline uppercase tracking-widest">Current Streak</span>
            <div className="mt-2 flex items-baseline gap-2 text-primary">
              <span className="font-headline-sm text-5xl font-bold">{calmStreak}</span>
              <span className="font-headline-sm text-lg font-semibold">Days</span>
            </div>
          </div>
          <div className="mt-4">
            <div className="h-1.5 w-full bg-surface-container rounded-full overflow-hidden mb-2">
              <div className="h-full bg-primary rounded-full transition-all duration-300" style={{ width: `80%` }}></div>
            </div>
            <p className="text-[10px] text-on-surface-variant font-medium">Next milestone: 15 days (Zen Master)</p>
          </div>
        </div>

        {/* Feeling overwhelmed banner */}
        <div className="col-span-1 md:col-span-2 bg-primary text-on-primary rounded-3xl p-8 shadow-md flex items-center justify-between">
          <div className="max-w-sm space-y-2">
            <h3 className="font-headline-sm text-2xl font-semibold text-white">Feeling overwhelmed?</h3>
            <p className="text-on-primary-container text-[13px] font-medium leading-relaxed opacity-90">
              Our AI Copilot suggests a 3-minute 'Focus Tap' session based on your elevated heart rate variability.
            </p>
          </div>
          <button 
            id="start-rec-btn"
            onClick={() => {
              const btn = document.getElementById('start-rec-btn');
              if (btn) {
                const original = btn.innerText;
                btn.innerText = 'Redirecting...';
                btn.classList.add('opacity-80', 'bg-surface-container-low');
                setTimeout(() => {
                  btn.innerText = original;
                  btn.classList.remove('opacity-80', 'bg-surface-container-low');
                  window.scrollTo({ top: 0, behavior: 'smooth' }); // Scroll up to the games
                }, 1000);
              }
            }}
            className="bg-white text-primary font-bold text-sm px-6 py-3 rounded-2xl shadow hover:bg-surface-container-lowest active:scale-95 transition-all"
          >
            Start Recommended Activity
          </button>
        </div>

      </div>
    </div>
  );
}
