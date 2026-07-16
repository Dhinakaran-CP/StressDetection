import React, { useEffect, useMemo, useRef, useState } from "react";

export default function BreathingExercise({ isActive, totalCycles = 5, onComplete }) {
  const [phase, setPhase] = useState("Inhale...");
  const [cycle, setCycle] = useState(0);
  const completeRef = useRef(false);

  useEffect(() => {
    if (!isActive) {
      setPhase("Inhale...");
      setCycle(0);
      completeRef.current = false;
      return;
    }

    let exhaleTimeout;
    const cycleInterval = setInterval(() => {
      setPhase("Inhale...");
      exhaleTimeout = setTimeout(() => setPhase("Exhale..."), 2000);

      setCycle((prev) => {
        const next = prev + 1;
        if (next >= totalCycles && !completeRef.current) {
          completeRef.current = true;
          if (onComplete) onComplete();
        }
        return Math.min(next, totalCycles);
      });
    }, 4000);

    exhaleTimeout = setTimeout(() => setPhase("Exhale..."), 2000);

    return () => {
      clearInterval(cycleInterval);
      clearTimeout(exhaleTimeout);
    };
  }, [isActive, onComplete, totalCycles]);

  const ringStyle = useMemo(() => {
    const progress = (cycle / totalCycles) * 360;
    return {
      background: `conic-gradient(var(--primary-color) ${progress}deg, var(--accent-light-bg) 0deg)`,
    };
  }, [cycle, totalCycles]);

  return (
    <div className="breathing-wrap">
      <div className="breathing-ring" style={ringStyle}>
        <div className="breathing-circle" />
      </div>
      <div className="breathing-text">{phase}</div>
      <small className="breathing-cycle-text">Cycle {cycle}/{totalCycles} completed</small>
    </div>
  );
}
