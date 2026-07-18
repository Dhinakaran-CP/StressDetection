import React from 'react';

export default function TopAppBar({ title, activeView, dashboardMode, setDashboardMode, showCopilot, setShowCopilot }) {
  return (
    <header className="fixed top-0 right-0 left-64 h-16 bg-surface/85 backdrop-blur-md z-40 flex justify-between items-center px-12 border-b border-outline-variant/10">
      <div className="flex items-center gap-8">
        <span className="font-headline-sm text-[20px] font-bold text-primary">{title}</span>
        
        {activeView === 'dashboard' && (
          <nav className="flex gap-4">
            <button
              onClick={() => setDashboardMode('realtime')}
              className={`pb-1 font-label-caps text-xs tracking-wider transition-all border-b-2 ${
                dashboardMode === 'realtime'
                  ? 'text-primary border-primary font-bold'
                  : 'text-on-surface-variant border-transparent hover:text-primary'
              }`}
            >
              Realtime
            </button>
            <button
              onClick={() => setDashboardMode('upload')}
              className={`pb-1 font-label-caps text-xs tracking-wider transition-all border-b-2 ${
                dashboardMode === 'upload'
                  ? 'text-primary border-primary font-bold'
                  : 'text-on-surface-variant border-transparent hover:text-primary'
              }`}
            >
              Upload
            </button>
          </nav>
        )}
      </div>

      <div className="flex items-center gap-6">
        <div className="flex gap-2">
          <button 
            id="nav-notif-btn"
            onClick={() => {
              const btn = document.getElementById('nav-notif-btn');
              if (btn) {
                const icon = btn.querySelector('span');
                const origIcon = icon.innerText;
                icon.innerText = 'check';
                icon.classList.add('text-[#4ADE80]');
                setTimeout(() => {
                  icon.innerText = origIcon;
                  icon.classList.remove('text-[#4ADE80]');
                }, 1500);
              }
            }}
            className="p-2 text-on-surface-variant hover:text-primary transition-all flex items-center justify-center rounded-xl hover:bg-surface-container-high active:scale-95"
          >
            <span className="material-symbols-outlined text-[20px]">notifications_active</span>
          </button>
          <button 
            id="nav-tune-btn"
            onClick={() => {
              const btn = document.getElementById('nav-tune-btn');
              if (btn) {
                const icon = btn.querySelector('span');
                icon.classList.add('animate-spin');
                setTimeout(() => {
                  icon.classList.remove('animate-spin');
                }, 1000);
              }
            }}
            className="p-2 text-on-surface-variant hover:text-primary transition-all flex items-center justify-center rounded-xl hover:bg-surface-container-high active:scale-95"
          >
            <span className="material-symbols-outlined text-[20px]">tune</span>
          </button>
        </div>

        <button
          onClick={() => setShowCopilot(!showCopilot)}
          className={`px-4 py-1.5 rounded-full font-label-caps text-xs tracking-wide flex items-center gap-2 transition-all ${
            showCopilot
              ? 'bg-primary text-on-primary shadow-sm'
              : 'bg-primary-container/10 text-primary hover:bg-primary-container/20'
          }`}
        >
          <span className="material-symbols-outlined text-[16px] animate-pulse">auto_awesome</span>
          STRESS CHAT
        </button>

        <div className="w-8 h-8 rounded-full bg-secondary-container overflow-hidden border border-outline-variant/30 flex items-center justify-center font-bold text-xs text-primary">
          VM
        </div>
      </div>
    </header>
  );
}
