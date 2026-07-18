import React from 'react';

export default function SideNavBar({ activeView, setActiveView, onCalibrateTrigger, user }) {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { id: 'insights', label: 'Insights', icon: 'insights' },
    { id: 'recovery', label: 'Recovery', icon: 'spa' }, // Using spa icon for Clinical-Zen recovery
    { id: 'profile', label: 'Profile', icon: 'person' }
  ];

  const userEmail = user?.email || 'Dr. Aris Thorne';
  const userName = userEmail.split('@')[0].replace('.', ' ').replace(/\b\w/g, c => c.toUpperCase());

  return (
    <aside className="h-screen w-64 fixed left-0 top-0 bg-surface-container-low shadow-sm flex flex-col py-8 px-6 z-50 select-none border-r border-outline-variant/10">
      <div className="mb-12">
        <h1 className="font-headline-md text-headline-md font-bold text-primary">VitalMind</h1>
        <p className="font-label-caps text-label-caps text-on-surface-variant opacity-70 mt-1 uppercase tracking-widest">Clinical Stress Monitoring</p>
      </div>

      <nav className="flex-1 space-y-2">
        {menuItems.map((item) => {
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-left ${
                isActive
                  ? 'text-primary font-bold bg-surface-container border-r-4 border-primary'
                  : 'text-on-surface-variant hover:bg-surface-container-high hover:text-primary'
              }`}
            >
              <span className="material-symbols-outlined" style={{ fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0" }}>
                {item.icon}
              </span>
              <span className="font-body-md text-sm">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto space-y-6">
        <button
          onClick={onCalibrateTrigger}
          className="w-full py-4 px-4 rounded-2xl bg-primary text-on-primary font-bold text-label-caps tracking-wider transition-all hover:opacity-90 active:scale-95 shadow-md flex items-center justify-center gap-2"
        >
          <span className="material-symbols-outlined text-sm">tune</span>
          Calibrate Sensors
        </button>

        <div className="flex items-center gap-3 pt-4 border-t border-outline-variant/20">
          <div className="w-10 h-10 rounded-full bg-secondary-container overflow-hidden flex items-center justify-center text-primary font-bold text-sm border border-primary/20">
            {userName.charAt(0)}
          </div>
          <div>
            <p className="font-label-caps text-xs text-on-surface font-bold truncate max-w-[140px]">{userName}</p>
            <p className="text-[10px] text-on-surface-variant uppercase tracking-tighter">Clinical Lead</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
