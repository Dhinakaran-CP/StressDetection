export const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:5000";

export const STRESS_LEVELS = {
  Extreme: { label: 'Extreme', color: 'var(--accent-red)' },
  High: { label: 'High', color: 'var(--accent-orange)' },
  Moderate: { label: 'Moderate', color: 'var(--accent-yellow)' },
  Low: { label: 'Low', color: 'var(--accent-green)' }
};

export const CHATBOT_PROMPT = "You are a supportive stress-management assistant in a general stress monitoring app. Give concise, practical, non-medical advice. Do not diagnose.";
