import React, { useState } from 'react';

export default function UserProfileSettings({ user, onLogout, onCalibrateTrigger }) {
  const [activeTab, setActiveTab] = useState('profile'); // 'profile' | 'security' | 'nodes'
  const userEmail = user?.email || 'alexander@vitalmind.ai';
  const userName = userEmail.split('@')[0].replace('.', ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="w-full max-w-[1440px] mx-auto animate-fade-in-up">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
        <div>
          <h2 className="font-display-hero text-3xl font-bold text-on-background">Profile & System Settings</h2>
          <p className="font-body-md text-sm text-on-surface-variant mt-1">
            Clinical account credentials, connected biometric nodes & security standards
          </p>
        </div>

        {/* Tab Selector */}
        <div className="flex items-center bg-surface-container-low p-1.5 rounded-2xl border border-primary/10">
          <button
            onClick={() => setActiveTab('profile')}
            className={`px-4 py-2 rounded-xl font-label-caps text-xs font-semibold transition-all ${
              activeTab === 'profile' ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-primary'
            }`}
          >
            Executive Profile
          </button>
          <button
            onClick={() => setActiveTab('security')}
            className={`px-4 py-2 rounded-xl font-label-caps text-xs font-semibold transition-all ${
              activeTab === 'security' ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-primary'
            }`}
          >
            Security & Standards
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Main Profile Info Card (2 Cols) */}
        <div className="lg:col-span-2 glass-card rounded-3xl p-8 shadow-sm flex flex-col md:flex-row gap-8 items-start">
          <div className="relative group shrink-0">
            <div className="w-32 h-32 md:w-36 md:h-36 rounded-2xl bg-primary text-on-primary flex items-center justify-center font-display-hero text-5xl font-bold border-4 border-white shadow-lg">
              {userName.charAt(0)}
            </div>
            <button className="absolute -bottom-2 -right-2 bg-primary text-on-primary p-2.5 rounded-xl shadow-xl hover:scale-105 transition-all">
              <span className="material-symbols-outlined text-[18px]">edit</span>
            </button>
          </div>

          <div className="flex-1 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h3 className="font-display-hero text-2xl font-bold text-on-surface">{userName}</h3>
                <p className="font-body-lg text-sm text-primary font-semibold mt-0.5">
                  Chief Neurologist & High-Performance Consultant
                </p>
              </div>
              <span className="px-4 py-1.5 bg-secondary-container/20 text-secondary font-bold rounded-full font-label-caps text-xs border border-secondary/20 self-start md:self-auto">
                Elite Member
              </span>
            </div>

            <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed max-w-2xl">
              Specializing in cognitive optimization and stress-resilience frameworks for executive leadership. Leveraging VitalMind Pro to synchronize circadian rhythms and neuro-metabolic recovery cycles.
            </p>

            <div className="flex flex-wrap gap-2.5 pt-2 text-xs">
              <span className="px-3 py-1 bg-surface-container rounded-full font-label-caps text-on-surface-variant border border-primary/5">
                London, UK
              </span>
              <span className="px-3 py-1 bg-surface-container rounded-full font-label-caps text-on-surface-variant border border-primary/5">
                Joined Oct 2023
              </span>
              <span className="px-3 py-1 bg-primary/10 text-primary font-bold rounded-full font-label-caps border border-primary/20">
                {userEmail}
              </span>
            </div>
          </div>
        </div>

        {/* Biometric Nodes Stack (1 Col) */}
        <div className="glass-card rounded-3xl p-6 space-y-6 shadow-sm">
          <div className="flex items-center justify-between border-b border-primary/10 pb-3">
            <h4 className="font-display-hero text-lg font-bold text-on-background">Biometric Nodes</h4>
            <span className="material-symbols-outlined text-primary text-xl">sensors</span>
          </div>

          <div className="space-y-3">
            {/* Apple Watch */}
            <div className="flex items-center justify-between p-3.5 bg-surface-container-low rounded-2xl border border-primary/10">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-on-surface text-surface rounded-xl flex items-center justify-center">
                  <span className="material-symbols-outlined text-lg">watch</span>
                </div>
                <div>
                  <p className="font-bold text-xs text-on-surface">Apple Watch Ultra</p>
                  <p className="font-label-caps text-[10px] text-on-surface-variant/70">Syncing Live • 60Hz</p>
                </div>
              </div>
              <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
            </div>

            {/* Oura Ring */}
            <div className="flex items-center justify-between p-3.5 bg-surface-container-low rounded-2xl border border-primary/10">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-on-surface text-surface rounded-xl flex items-center justify-center">
                  <span className="material-symbols-outlined text-lg">radio_button_checked</span>
                </div>
                <div>
                  <p className="font-bold text-xs text-on-surface">Oura Horizon Ring</p>
                  <p className="font-label-caps text-[10px] text-on-surface-variant/70">Restorative Sleep Sync</p>
                </div>
              </div>
              <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
            </div>

            {/* Bio-GSR Sensor */}
            <div className="flex items-center justify-between p-3.5 bg-surface-container-low rounded-2xl border border-primary/10">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-primary/10 text-primary rounded-xl flex items-center justify-center">
                  <span className="material-symbols-outlined text-lg">ecg</span>
                </div>
                <div>
                  <p className="font-bold text-xs text-on-surface">Bio-GSR Chest Patch</p>
                  <p className="font-label-caps text-[10px] text-primary">Connected • Phase 1</p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded-md bg-primary/10 text-primary font-label-caps text-[9px] font-bold">
                ACTIVE
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* System Security & Account Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="glass-card p-6 rounded-3xl space-y-4 shadow-sm">
          <h4 className="font-display-hero text-lg font-bold text-on-background border-b border-primary/10 pb-3">
            Clinical Compliance Standards
          </h4>
          <div className="space-y-3">
            <div className="p-4 bg-surface-container-low rounded-2xl border border-primary/10 flex items-center justify-between">
              <div>
                <p className="font-label-caps text-[10px] text-outline font-bold uppercase tracking-wider mb-1">
                  HIPAA Encryption Protocol
                </p>
                <p className="text-primary font-bold text-xs">AES-256 GCM ACTIVE</p>
              </div>
              <span className="material-symbols-outlined text-primary text-xl">verified_user</span>
            </div>

            <div className="p-4 bg-surface-container-low rounded-2xl border border-primary/10 flex items-center justify-between">
              <div>
                <p className="font-label-caps text-[10px] text-outline font-bold uppercase tracking-wider mb-1">
                  ISO 27001 Certification
                </p>
                <p className="text-primary font-bold text-xs">CERTIFIED & VERIFIED</p>
              </div>
              <span className="material-symbols-outlined text-primary text-xl">workspace_premium</span>
            </div>
          </div>
        </div>

        <div className="glass-card p-6 rounded-3xl space-y-4 shadow-sm flex flex-col justify-between">
          <h4 className="font-display-hero text-lg font-bold text-on-background border-b border-primary/10 pb-3">
            Account Management & Actions
          </h4>
          <div className="flex gap-4">
            <button
              onClick={onCalibrateTrigger}
              className="flex-1 bg-primary text-on-primary py-3.5 rounded-xl font-label-caps text-xs font-bold hover:opacity-90 transition-all shadow-md flex items-center justify-center gap-2"
            >
              <span className="material-symbols-outlined text-base">tune</span>
              <span>Calibrate Sensors</span>
            </button>
            <button
              onClick={onLogout}
              className="flex-1 border border-error text-error hover:bg-error-container/10 py-3.5 rounded-xl font-label-caps text-xs font-bold transition-all flex items-center justify-center gap-2"
            >
              <span className="material-symbols-outlined text-base">logout</span>
              <span>Log Out</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
