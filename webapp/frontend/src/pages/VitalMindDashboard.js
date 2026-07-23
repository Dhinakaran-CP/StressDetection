import React from 'react';

export default function VitalMindDashboard({ onNavigate }) {
  return (
    <div className="w-full max-w-[1440px] mx-auto animate-fade-in-up">
      {/* Welcome Hero Section */}
      <section className="mb-8">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <h2 className="font-display-hero text-3xl md:text-5xl font-bold text-on-background tracking-tight">
              Welcome back, Alexander
            </h2>
            <p className="font-body-lg text-on-surface-variant mt-2 max-w-2xl text-base md:text-lg">
              Your VitalMind AI has analyzed 12,400 data points since midnight. Your overall vitality is at a peak performance level today.
            </p>
          </div>
          <div className="flex gap-2">
            <span className="px-4 py-2 bg-secondary/10 text-secondary rounded-full font-label-caps text-xs flex items-center gap-2 font-semibold">
              <span className="w-2 h-2 rounded-full bg-secondary animate-pulse"></span>
              Optimal Performance
            </span>
          </div>
        </div>
      </section>

      {/* Bento Grid Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Stress Level Gauge */}
        <div className="glass-card p-6 rounded-2xl flex flex-col justify-between shadow-sm">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="font-label-caps text-xs text-on-surface-variant/70 uppercase tracking-wider font-semibold">Stress Level</p>
              <h3 className="font-display-hero text-4xl font-bold text-primary mt-1">18%</h3>
            </div>
            <span className="p-3 bg-primary-container/10 text-primary rounded-xl material-symbols-outlined">
              sentiment_satisfied
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-emerald-600 mb-2">
              <span className="material-symbols-outlined text-sm">trending_down</span>
              <span>-4% vs yesterday</span>
            </div>
            <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
              <div className="bg-primary h-full rounded-full w-[18%] transition-all duration-500"></div>
            </div>
          </div>
        </div>

        {/* HRV Stability */}
        <div className="glass-card p-6 rounded-2xl flex flex-col justify-between shadow-sm">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="font-label-caps text-xs text-on-surface-variant/70 uppercase tracking-wider font-semibold">HRV Stability</p>
              <h3 className="font-display-hero text-4xl font-bold text-on-background mt-1">68 ms</h3>
            </div>
            <span className="p-3 bg-tertiary-container/10 text-tertiary rounded-xl material-symbols-outlined">
              ecg
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-tertiary mb-2">
              <span className="material-symbols-outlined text-sm">check_circle</span>
              <span>Parasympathetic active</span>
            </div>
            <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
              <div className="bg-tertiary h-full rounded-full w-[78%] transition-all duration-500"></div>
            </div>
          </div>
        </div>

        {/* Recovery Index */}
        <div className="glass-card p-6 rounded-2xl flex flex-col justify-between shadow-sm">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="font-label-caps text-xs text-on-surface-variant/70 uppercase tracking-wider font-semibold">Recovery Index</p>
              <h3 className="font-display-hero text-4xl font-bold text-on-background mt-1">92 / 100</h3>
            </div>
            <span className="p-3 bg-secondary-container/20 text-secondary rounded-xl material-symbols-outlined">
              battery_charging_full
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-secondary mb-2">
              <span className="material-symbols-outlined text-sm">bedtime</span>
              <span>Restorative sleep 8.2h</span>
            </div>
            <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
              <div className="bg-secondary h-full rounded-full w-[92%] transition-all duration-500"></div>
            </div>
          </div>
        </div>

        {/* Cognitive Fatigue */}
        <div className="glass-card p-6 rounded-2xl flex flex-col justify-between shadow-sm">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="font-label-caps text-xs text-on-surface-variant/70 uppercase tracking-wider font-semibold">Cognitive Fatigue</p>
              <h3 className="font-display-hero text-4xl font-bold text-on-background mt-1">14%</h3>
            </div>
            <span className="p-3 bg-primary/10 text-primary rounded-xl material-symbols-outlined">
              psychology
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-emerald-600 mb-2">
              <span className="material-symbols-outlined text-sm">psychology_alt</span>
              <span>Low neural strain</span>
            </div>
            <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
              <div className="bg-primary h-full rounded-full w-[14%] transition-all duration-500"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Analytics & Stream Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Real-time Biometric Stream Card */}
        <div className="glass-card p-6 rounded-3xl lg:col-span-2 shadow-sm flex flex-col justify-between">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="font-display-hero text-xl font-bold text-on-background">Live Bio-Telemetry Stream</h3>
              <p className="font-body-md text-xs text-on-surface-variant mt-0.5">Real-time photoplethysmography & eye aspect ratio monitoring</p>
            </div>
            <button
              onClick={() => onNavigate && onNavigate('realtime')}
              className="px-4 py-2 bg-primary text-on-primary rounded-xl font-label-caps text-xs font-semibold flex items-center gap-2 hover:bg-primary/90 transition-all"
            >
              <span className="material-symbols-outlined text-base">videocam</span>
              <span>Open Monitor</span>
            </button>
          </div>

          <div className="relative w-full h-64 bg-surface-container-lowest rounded-2xl overflow-hidden border border-primary/10 flex items-center justify-center">
            {/* Live Stream Canvas placeholder / visualizer */}
            <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-tertiary/5 to-primary/5 flex items-center justify-center">
              <svg className="w-full h-full text-primary/30" viewBox="0 0 500 150" preserveAspectRatio="none">
                <path
                  d="M0,75 Q50,20 100,75 T200,75 T300,30 T400,120 T500,75"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
              </svg>
            </div>
            <div className="relative z-10 flex flex-col items-center gap-3 text-center px-4">
              <span className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center material-symbols-outlined text-2xl animate-pulse">
                sensors
              </span>
              <p className="font-label-caps text-xs text-primary font-bold tracking-wider">
                LIVE MULTIMODAL FEED ACTIVE
              </p>
              <p className="text-xs text-on-surface-variant max-w-sm">
                468 facial landmarks & 120Hz heart rate variability tracking synchronized with neural model inference engine.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mt-6 pt-4 border-t border-primary/10 text-center">
            <div>
              <p className="font-label-caps text-[10px] text-on-surface-variant uppercase font-semibold">Pulse Rate</p>
              <p className="font-display-hero text-lg font-bold text-primary mt-0.5">64 BPM</p>
            </div>
            <div>
              <p className="font-label-caps text-[10px] text-on-surface-variant uppercase font-semibold">Respiration</p>
              <p className="font-display-hero text-lg font-bold text-secondary mt-0.5">14 RPM</p>
            </div>
            <div>
              <p className="font-label-caps text-[10px] text-on-surface-variant uppercase font-semibold">Blink Rate</p>
              <p className="font-display-hero text-lg font-bold text-tertiary mt-0.5">18 / min</p>
            </div>
          </div>
        </div>

        {/* System Diagnostics & Neural Confidence */}
        <div className="glass-card p-6 rounded-3xl shadow-sm flex flex-col justify-between space-y-6">
          <div>
            <h3 className="font-display-hero text-xl font-bold text-on-background">AI Diagnostics</h3>
            <p className="font-body-md text-xs text-on-surface-variant mt-0.5">Neural Network Confidence Engine</p>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-on-surface">Multimodal Pipeline</span>
                <span className="text-primary font-bold">99.4%</span>
              </div>
              <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                <div className="bg-primary h-full rounded-full w-[99%]"></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-on-surface">Face Mesh Feature Vector</span>
                <span className="text-secondary font-bold">98.1%</span>
              </div>
              <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                <div className="bg-secondary h-full rounded-full w-[98%]"></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-on-surface">Vocal Pitch Spectrum</span>
                <span className="text-tertiary font-bold">96.8%</span>
              </div>
              <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                <div className="bg-tertiary h-full rounded-full w-[97%]"></div>
              </div>
            </div>
          </div>

          <div className="p-4 bg-surface-container-low rounded-2xl border border-primary/10 space-y-2">
            <div className="flex items-center gap-2 text-primary font-label-caps text-xs font-bold">
              <span className="material-symbols-outlined text-base">auto_awesome</span>
              <span>AI Recommendation</span>
            </div>
            <p className="text-xs text-on-surface-variant leading-relaxed">
              Biometric indicators suggest optimal physiological equilibrium. Maintain current focus session until 16:30.
            </p>
          </div>

          <button
            onClick={() => onNavigate && onNavigate('insights')}
            className="w-full py-3 bg-surface-container text-primary rounded-xl font-label-caps text-xs font-bold hover:bg-surface-container-high transition-all flex items-center justify-center gap-2 border border-primary/10"
          >
            <span>View Detailed Insights</span>
            <span className="material-symbols-outlined text-base">arrow_forward</span>
          </button>
        </div>
      </div>

      {/* Actionable Executive Protocols */}
      <section className="mb-8">
        <h3 className="font-display-hero text-xl font-bold text-on-background mb-4">Recommended Recovery Protocols</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="glass-card p-6 rounded-2xl flex flex-col justify-between shadow-sm">
            <div className="flex items-start gap-4">
              <span className="p-3 bg-primary/10 text-primary rounded-xl material-symbols-outlined text-2xl">
                air
              </span>
              <div>
                <h4 className="font-display-hero text-base font-bold text-on-background">4-7-8 Breathing Protocol</h4>
                <p className="text-xs text-on-surface-variant mt-1">5 min parasympathetic activation</p>
              </div>
            </div>
            <button
              onClick={() => onNavigate && onNavigate('recovery')}
              className="mt-6 w-full py-2.5 bg-primary/10 text-primary rounded-xl font-label-caps text-xs font-bold hover:bg-primary/20 transition-all flex items-center justify-center gap-2"
            >
              <span>Start Session</span>
              <span className="material-symbols-outlined text-sm">play_arrow</span>
            </button>
          </div>

          <div className="glass-card p-6 rounded-2xl flex flex-col justify-between shadow-sm">
            <div className="flex items-start gap-4">
              <span className="p-3 bg-tertiary/10 text-tertiary rounded-xl material-symbols-outlined text-2xl">
                cloud_upload
              </span>
              <div>
                <h4 className="font-display-hero text-base font-bold text-on-background">Upload New Bio-Data</h4>
                <p className="text-xs text-on-surface-variant mt-1">Batch process ECG/PPG/Voice telemetry</p>
              </div>
            </div>
            <button
              onClick={() => onNavigate && onNavigate('upload')}
              className="mt-6 w-full py-2.5 bg-tertiary/10 text-tertiary rounded-xl font-label-caps text-xs font-bold hover:bg-tertiary/20 transition-all flex items-center justify-center gap-2"
            >
              <span>Upload Files</span>
              <span className="material-symbols-outlined text-sm">upload</span>
            </button>
          </div>

          <div className="glass-card p-6 rounded-2xl flex flex-col justify-between shadow-sm">
            <div className="flex items-start gap-4">
              <span className="p-3 bg-secondary/10 text-secondary rounded-xl material-symbols-outlined text-2xl">
                analytics
              </span>
              <div>
                <h4 className="font-display-hero text-base font-bold text-on-background">Multimodal Analysis</h4>
                <p className="text-xs text-on-surface-variant mt-1">Cross-reference face mesh & voice data</p>
              </div>
            </div>
            <button
              onClick={() => onNavigate && onNavigate('multimodal')}
              className="mt-6 w-full py-2.5 bg-secondary/10 text-secondary rounded-xl font-label-caps text-xs font-bold hover:bg-secondary/20 transition-all flex items-center justify-center gap-2"
            >
              <span>Open Center</span>
              <span className="material-symbols-outlined text-sm">explore</span>
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
