import React, { useEffect, useState } from "react";

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const useTweenValue = (from, to, duration = 900) => {
  const [value, setValue] = useState(from);

  useEffect(() => {
    let frame;
    const start = performance.now();

    const tick = (now) => {
      const progress = clamp((now - start) / duration, 0, 1);
      const current = Math.round(from + (to - from) * progress);
      setValue(current);
      if (progress < 1) {
        frame = requestAnimationFrame(tick);
      }
    };

    frame = requestAnimationFrame(tick);
    return () => {
      if (frame) cancelAnimationFrame(frame);
    };
  }, [from, to, duration]);

  return value;
};

export default function RewardSystem({ reward, calmStreak }) {
  const fromScore = reward ? reward.from : 0;
  const toScore = reward ? reward.to : 0;
  const animatedScore = useTweenValue(fromScore, toScore, 1100);

  if (!reward) return null;

  return (
    <div className="recovery-highlight fade-in-up">
      <h5 style={{ marginBottom: "0.35rem" }}>Great job!</h5>
      <p style={{ marginBottom: "0.35rem" }}>Stress reduced by {reward.reducedBy}%</p>
      <p style={{ marginBottom: "0.35rem" }}>
        Recovery Score: {reward.from}% -&gt; {animatedScore}%
      </p>
      <div className="calm-streak">Calm Streak: {calmStreak} mins</div>
    </div>
  );
}
