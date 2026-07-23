import React, { useState } from 'react';
import PersonalInsights from '../components/PersonalInsights';

export default function AIInsightsHub({ onNavigate }) {
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'historical' | 'correlations'

  return (
    <div className="w-full max-w-[1440px] mx-auto animate-fade-in-up">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
        <div>
          <h2 className="font-display-hero text-3xl font-bold text-on-background">AI Insights & Analytics</h2>
          <p className="font-body-md text-sm text-on-surface-variant mt-1">
            Deep neural network feature importance breakdown & longitudinal biometric trends
          </p>
        </div>

        {/* Tab Filters */}
        <div className="flex items-center bg-surface-container-low p-1.5 rounded-2xl border border-primary/10">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-2 rounded-xl font-label-caps text-xs font-semibold transition-all ${
              activeTab === 'overview' ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-primary'
            }`}
          >
            Overview & Importance
          </button>
          <button
            onClick={() => setActiveTab('historical')}
            className={`px-4 py-2 rounded-xl font-label-caps text-xs font-semibold transition-all ${
              activeTab === 'historical' ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-primary'
            }`}
          >
            Longitudinal History
          </button>
        </div>
      </div>

      {/* Feature Importance & Correlation Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Multimodal Feature Contribution */}
        <div className="glass-card p-6 rounded-3xl lg:col-span-2 shadow-sm space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="font-display-hero text-xl font-bold text-on-background">Feature Contribution Matrix</h3>
              <p className="text-xs text-on-surface-variant mt-0.5">SHAP-based model feature importance weights</p>
            </div>
            <span className="px-3 py-1 bg-primary/10 text-primary rounded-full font-label-caps text-[10px] font-bold">
              XGBoost + ResNet50
            </span>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs font-semibold mb-1.5">
                <span className="text-on-surface flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-base">ecg</span>
                  Heart Rate Variability (HRV SDNN)
                </span>
                <span className="text-primary font-bold">38% Impact</span>
              </div>
              <div className="w-full bg-surface-container-high h-3 rounded-full overflow-hidden">
                <div className="bg-primary h-full rounded-full w-[38%]"></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold mb-1.5">
                <span className="text-on-surface flex items-center gap-2">
                  <span className="material-symbols-outlined text-secondary text-base">face</span>
                  Facial Action Units (AU04, AU07, AU12)
                </span>
                <span className="text-secondary font-bold">26% Impact</span>
              </div>
              <div className="w-full bg-surface-container-high h-3 rounded-full overflow-hidden">
                <div className="bg-secondary h-full rounded-full w-[26%]"></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold mb-1.5">
                <span className="text-on-surface flex items-center gap-2">
                  <span className="material-symbols-outlined text-tertiary text-base">visibility</span>
                  Eye Aspect Ratio & Blink Duration
                </span>
                <span className="text-tertiary font-bold">20% Impact</span>
              </div>
              <div className="w-full bg-surface-container-high h-3 rounded-full overflow-hidden">
                <div className="bg-tertiary h-full rounded-full w-[20%]"></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold mb-1.5">
                <span className="text-on-surface flex items-center gap-2">
                  <span className="material-symbols-outlined text-amber-600 text-base">graphic_eq</span>
                  Vocal Pitch Energy & Jitter
                </span>
                <span className="text-amber-600 font-bold">16% Impact</span>
              </div>
              <div className="w-full bg-surface-container-high h-3 rounded-full overflow-hidden">
                <div className="bg-amber-500 h-full rounded-full w-[16%]"></div>
              </div>
            </div>
          </div>

          <div className="p-4 bg-surface-container-low rounded-2xl border border-primary/10 flex items-center justify-between text-xs">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-primary text-xl">psychology</span>
              <span>Model Confidence: <strong>99.4% Ensemble Precision</strong></span>
            </div>
            <span className="text-on-surface-variant">Updated 2 mins ago</span>
          </div>
        </div>

        {/* AI Insight Summary Card */}
        <div className="glass-card p-6 rounded-3xl shadow-sm flex flex-col justify-between space-y-6">
          <div>
            <h3 className="font-display-hero text-xl font-bold text-on-background">Key Findings</h3>
            <p className="text-xs text-on-surface-variant mt-0.5">Automated Clinical Intelligence Summary</p>
          </div>

          <div className="space-y-4">
            <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl space-y-1">
              <div className="flex items-center gap-2 text-emerald-700 font-label-caps text-xs font-bold">
                <span className="material-symbols-outlined text-base">trending_down</span>
                <span>Low Cortisol Signal</span>
              </div>
              <p className="text-xs text-on-surface-variant">
                HRV frequency spectrum indicates elevated vagal tone during mid-day focus hours.
              </p>
            </div>

            <div className="p-4 bg-primary/10 border border-primary/20 rounded-2xl space-y-1">
              <div className="flex items-center gap-2 text-primary font-label-caps text-xs font-bold">
                <span className="material-symbols-outlined text-base">bedtime</span>
                <span>Sleep Synchronization</span>
              </div>
              <p className="text-xs text-on-surface-variant">
                Circadian alignment is optimal with a 92% recovery score following 8.2h restorative sleep.
              </p>
            </div>
          </div>

          <button
            onClick={() => onNavigate && onNavigate('multimodal')}
            className="w-full py-3 bg-primary text-on-primary rounded-xl font-label-caps text-xs font-semibold shadow-lg hover:shadow-primary/20 transition-all flex items-center justify-center gap-2"
          >
            <span>Run Multimodal Analysis</span>
            <span className="material-symbols-outlined text-base">arrow_forward</span>
          </button>
        </div>
      </div>

      {/* Embedded Personal Insights Historical Engine */}
      <div className="glass-card p-6 rounded-3xl shadow-sm mb-8">
        <PersonalInsights />
      </div>
    </div>
  );
}
