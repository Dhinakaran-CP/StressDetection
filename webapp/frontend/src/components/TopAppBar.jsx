import React, { useState, useEffect } from 'react';
import { API_BASE } from '../config';

export default function TopAppBar({ title, activeView, dashboardMode, setDashboardMode, showCopilot, setShowCopilot, isSidebarOpen = true, setIsSidebarOpen }) {
  const [selectedModel, setSelectedModel] = useState('cnn_grl');
  const [isSwitching, setIsSwitching] = useState(false);

  useEffect(() => {
    // Fetch initial model state from backend
    fetch(`${API_BASE}/api/model/select`)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success' && data.active_model_key) {
          setSelectedModel(data.active_model_key);
        }
      })
      .catch(err => console.warn('Could not fetch active model:', err));
  }, []);

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    setIsSwitching(true);

    fetch(`${API_BASE}/api/model/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: newModel })
    })
      .then(res => res.json())
      .then(data => {
        setIsSwitching(false);
        if (data.status === 'success' && data.active_model_key) {
          setSelectedModel(data.active_model_key);
        }
      })
      .catch(err => {
        setIsSwitching(false);
        console.error('Failed to select model:', err);
      });
  };

  return (
    <header
      className={`fixed top-0 right-0 left-0 ${
        isSidebarOpen ? 'lg:left-64' : 'lg:left-20'
      } h-16 bg-surface/80 backdrop-blur-xl z-40 flex justify-between items-center px-4 md:px-6 border-b border-primary/10 shadow-[0_40px_40px_-10px_rgba(0,84,214,0.05)] transition-all duration-300`}
    >
      <div className="flex items-center gap-4">
        <button
          onClick={() => setIsSidebarOpen && setIsSidebarOpen(!isSidebarOpen)}
          className="lg:hidden text-primary cursor-pointer p-1 rounded-lg hover:bg-primary/5"
          aria-label="Toggle menu"
        >
          <span className="material-symbols-outlined">menu</span>
        </button>
        <span className="font-display-hero text-headline-md tracking-tighter text-primary font-bold">
          {title || 'VitalMind'}
        </span>
      </div>

      <div className="flex items-center gap-3 md:gap-4">
        {/* Model Selector Dropdown */}
        <div className="flex items-center gap-2 bg-surface-container-low px-3 py-1.5 rounded-xl border border-primary/15 shadow-sm">
          <span className="material-symbols-outlined text-primary text-[18px]">
            {selectedModel === 'cnn_grl' ? 'psychology' : 'forest'}
          </span>
          <div className="flex flex-col">
            <span className="text-[9px] uppercase font-bold tracking-wider text-on-surface-variant font-label-caps leading-none">
              Active Model
            </span>
            <select
              value={selectedModel}
              onChange={handleModelChange}
              disabled={isSwitching}
              className="bg-transparent text-xs font-bold text-primary focus:outline-none cursor-pointer border-none p-0 m-0 font-display"
              aria-label="Select AI Model"
            >
              <option value="cnn_grl">🧠 CNN + GRL (Deep)</option>
              <option value="random_forest">🌲 Random Forest (ML)</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-2 md:gap-3">
          <button
            onClick={() => setShowCopilot(!showCopilot)}
            className={`material-symbols-outlined p-2 rounded-full hover:bg-primary/5 transition-colors ${
              showCopilot ? 'text-primary bg-primary-container/20' : 'text-on-surface-variant'
            }`}
            title="Toggle Copilot AI"
          >
            smart_toy
          </button>
          
          <button
            className="material-symbols-outlined p-2 rounded-full hover:bg-primary/5 text-on-surface-variant transition-colors relative"
            title="Notifications"
          >
            notifications
            <span className="absolute top-2 right-2 w-2 h-2 bg-error rounded-full border-2 border-surface"></span>
          </button>

          <div className="w-9 h-9 rounded-full bg-primary-container overflow-hidden border-2 border-white shadow-sm flex items-center justify-center text-on-primary font-bold text-xs">
            A
          </div>
        </div>
      </div>
    </header>
  );
}

