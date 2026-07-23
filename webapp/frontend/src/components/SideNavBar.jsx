import React from 'react';

export default function SideNavBar({ activeView, setActiveView, onCalibrateTrigger, user, isOpen = true, setIsOpen }) {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { id: 'realtime', label: 'Realtime Monitor', icon: 'videocam' },
    { id: 'insights', label: 'AI Insights', icon: 'insights' },
    { id: 'multimodal', label: 'Multimodal Center', icon: 'analytics' },
    { id: 'recovery', label: 'Recovery', icon: 'self_care' },
    { id: 'upload', label: 'Bio-Data Upload', icon: 'cloud_upload' },
    { id: 'profile', label: 'Profile', icon: 'person' }
  ];

  return (
    <aside
      className={`hidden lg:flex flex-col fixed left-0 top-0 h-full z-50 py-8 bg-surface/70 backdrop-blur-2xl border-r border-primary/10 shadow-[40px_0_40px_-10px_rgba(0,84,214,0.05)] transition-all duration-300 ${
        isOpen ? 'w-64 px-4' : 'w-20 px-3'
      }`}
    >
      {/* Header & Toggle */}
      <div className={`mb-8 px-2 flex items-center ${isOpen ? 'justify-between' : 'justify-center flex-col gap-2'}`}>
        {isOpen ? (
          <div>
            <h1 className="font-display-hero text-headline-md text-primary font-bold">VitalMind AI</h1>
            <p className="font-label-caps text-on-surface-variant/70 mt-0.5 text-xs">Executive Health</p>
          </div>
        ) : (
          <h1 className="font-display-hero text-headline-md text-primary font-bold">VM</h1>
        )}
        <button
          onClick={() => setIsOpen && setIsOpen(!isOpen)}
          className="text-on-surface-variant hover:text-primary transition-colors p-1.5 rounded-xl hover:bg-primary/5 flex items-center justify-center"
          title={isOpen ? 'Collapse Sidebar' : 'Expand Sidebar'}
          aria-label="Toggle Sidebar"
        >
          <span className="material-symbols-outlined text-xl">
            {isOpen ? 'chevron_left' : 'menu'}
          </span>
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 space-y-1.5 overflow-y-auto">
        {menuItems.map((item) => {
          const isActive = activeView === item.id || (activeView.startsWith(item.id));
          return (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center ${isOpen ? 'gap-3 px-4' : 'justify-center px-0'} py-3 rounded-xl transition-all duration-200 text-left ${
                isActive
                  ? 'text-primary bg-primary-container/20 border-r-2 border-primary translate-x-1 font-semibold'
                  : 'text-on-surface-variant/70 hover:bg-surface-variant/30 hover:text-primary'
              }`}
              title={!isOpen ? item.label : undefined}
            >
              <span
                className="material-symbols-outlined text-xl"
                style={{ fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0" }}
              >
                {item.icon}
              </span>
              {isOpen && <span className="font-label-caps text-xs tracking-wider font-semibold uppercase">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Footer Action Buttons */}
      <div className="mt-auto px-2 pt-4 border-t border-primary/10 space-y-3">
        <button
          onClick={() => setActiveView('upload')}
          className={`w-full bg-primary text-on-primary py-3.5 rounded-xl font-label-caps text-xs font-semibold shadow-lg hover:shadow-primary/20 transition-all flex items-center justify-center gap-2 ${
            !isOpen ? 'px-0' : 'px-4'
          }`}
          title={!isOpen ? 'Upload Bio-Data' : undefined}
        >
          <span className="material-symbols-outlined text-[20px]">upload_file</span>
          {isOpen && <span>Upload Bio-Data</span>}
        </button>
        {onCalibrateTrigger && isOpen && (
          <button
            onClick={onCalibrateTrigger}
            className="w-full bg-surface-container text-primary py-2.5 rounded-xl font-label-caps text-xs font-semibold hover:bg-surface-container-high transition-all flex items-center justify-center gap-2 border border-primary/10"
          >
            <span className="material-symbols-outlined text-[18px]">tune</span>
            <span>Calibrate Sensors</span>
          </button>
        )}
      </div>
    </aside>
  );
}

