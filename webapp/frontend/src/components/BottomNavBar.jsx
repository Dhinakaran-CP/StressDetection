import React from 'react';

export default function BottomNavBar({ activeView, setActiveView, onCalibrateTrigger }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { id: 'insights', label: 'Insights', icon: 'insights' },
    { id: 'recovery', label: 'Recovery', icon: 'spa' },
    { id: 'profile', label: 'Profile', icon: 'person' },
  ];

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-surface-container-low/95 backdrop-blur-md border-t border-outline-variant/20 z-[500] flex items-center justify-around px-2 select-none">
      {navItems.map((item) => {
        const isActive = activeView === item.id;
        return (
          <button
            key={item.id}
            onClick={() => setActiveView(item.id)}
            className={`flex flex-col items-center justify-center w-14 h-12 rounded-xl transition-all duration-200 ${
              isActive
                ? 'text-primary font-bold bg-primary/10'
                : 'text-on-surface-variant hover:text-primary'
            }`}
          >
            <span
              className="material-symbols-outlined text-xl"
              style={{ fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0" }}
            >
              {item.icon}
            </span>
            <span className="text-[10px] font-label-caps mt-0.5">{item.label}</span>
          </button>
        );
      })}

      <button
        onClick={onCalibrateTrigger}
        className="flex flex-col items-center justify-center w-14 h-12 rounded-xl text-primary hover:bg-primary/10 transition-all duration-200"
        title="Calibrate Sensors"
      >
        <span className="material-symbols-outlined text-xl">tune</span>
        <span className="text-[10px] font-label-caps mt-0.5">Calibrate</span>
      </button>
    </nav>
  );
}
