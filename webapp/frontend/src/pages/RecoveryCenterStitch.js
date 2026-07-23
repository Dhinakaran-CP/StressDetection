import React from 'react';
import RecoveryActivities from '../components/RecoveryActivities';

export default function RecoveryCenterStitch({ onNavigate }) {
  return (
    <div className="w-full max-w-[1440px] mx-auto animate-fade-in-up">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
        <div>
          <h2 className="font-display-hero text-3xl font-bold text-on-background">Recovery & Resilience Center</h2>
          <p className="font-body-md text-sm text-on-surface-variant mt-1">
            Bio-guided breathing exercises, neural decompression sessions & parasympathetic stimulation
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-4 py-2 bg-secondary-container/20 border border-secondary/20 rounded-2xl flex items-center gap-2">
            <span className="material-symbols-outlined text-secondary text-lg">workspace_premium</span>
            <span className="font-label-caps text-xs font-bold text-secondary">840 XP • Level 4 Resilient</span>
          </div>
        </div>
      </div>

      {/* Embedded Interactive Recovery Activities Engine */}
      <div className="glass-card p-6 rounded-3xl shadow-sm mb-8">
        <RecoveryActivities />
      </div>
    </div>
  );
}
