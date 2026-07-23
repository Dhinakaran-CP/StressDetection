import React, { useState } from 'react';
import AnalysisPanel from '../components/AnalysisPanel';

export default function MultimodalAnalysisCenter({ onNavigate }) {
  const [activeStage, setActiveStage] = useState(1); // 1, 2, 3, 4

  return (
    <div className="w-full max-w-[1440px] mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
        <div>
          <h2 className="font-display-hero text-3xl font-bold text-on-background">Multimodal Analysis Center</h2>
          <p className="font-body-md text-sm text-on-surface-variant mt-1">
            Cross-reference facial mesh landmarks, vocal acoustics, and physiological bio-signals
          </p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => onNavigate && onNavigate('upload')}
            className="px-4 py-2.5 bg-primary text-on-primary rounded-xl font-label-caps text-xs font-semibold flex items-center gap-2 shadow-sm hover:bg-primary/90 transition-all"
          >
            <span className="material-symbols-outlined text-base">cloud_upload</span>
            <span>Upload Files</span>
          </button>
        </div>
      </div>

      {/* Pipeline Stage Visualizer Bar */}
      <div className="glass-card p-6 rounded-3xl mb-8 shadow-sm">
        <h3 className="font-display-hero text-sm font-bold text-on-background uppercase tracking-wider mb-4">
          5-Stage Multimodal Inference Pipeline
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div
            onClick={() => setActiveStage(1)}
            className={`p-4 rounded-2xl cursor-pointer border transition-all ${
              activeStage === 1
                ? 'bg-primary-container/20 border-primary text-primary shadow-sm'
                : 'bg-surface-container-low border-primary/10 text-on-surface-variant hover:border-primary/30'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-label-caps text-[10px] font-bold uppercase tracking-wider">Stage 01</span>
              <span className="material-symbols-outlined text-base">tune</span>
            </div>
            <p className="font-display-hero text-sm font-bold">Signal Preprocessing</p>
            <p className="text-[11px] opacity-80 mt-1">Filtering & Normalization</p>
          </div>

          <div
            onClick={() => setActiveStage(2)}
            className={`p-4 rounded-2xl cursor-pointer border transition-all ${
              activeStage === 2
                ? 'bg-primary-container/20 border-primary text-primary shadow-sm'
                : 'bg-surface-container-low border-primary/10 text-on-surface-variant hover:border-primary/30'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-label-caps text-[10px] font-bold uppercase tracking-wider">Stage 02</span>
              <span className="material-symbols-outlined text-base">face</span>
            </div>
            <p className="font-display-hero text-sm font-bold">Landmark Extraction</p>
            <p className="text-[11px] opacity-80 mt-1">468 3D Mesh Coordinates</p>
          </div>

          <div
            onClick={() => setActiveStage(3)}
            className={`p-4 rounded-2xl cursor-pointer border transition-all ${
              activeStage === 3
                ? 'bg-primary-container/20 border-primary text-primary shadow-sm'
                : 'bg-surface-container-low border-primary/10 text-on-surface-variant hover:border-primary/30'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-label-caps text-[10px] font-bold uppercase tracking-wider">Stage 03</span>
              <span className="material-symbols-outlined text-base">psychology</span>
            </div>
            <p className="font-display-hero text-sm font-bold">Model Inference</p>
            <p className="text-[11px] opacity-80 mt-1">XGBoost + ResNet50 + MLP</p>
          </div>

          <div
            onClick={() => setActiveStage(4)}
            className={`p-4 rounded-2xl cursor-pointer border transition-all ${
              activeStage === 4
                ? 'bg-primary-container/20 border-primary text-primary shadow-sm'
                : 'bg-surface-container-low border-primary/10 text-on-surface-variant hover:border-primary/30'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-label-caps text-[10px] font-bold uppercase tracking-wider">Stage 04</span>
              <span className="material-symbols-outlined text-base">insights</span>
            </div>
            <p className="font-display-hero text-sm font-bold">Late Fusion Output</p>
            <p className="text-[11px] opacity-80 mt-1">Clinical Stress Score 18/100</p>
          </div>
        </div>
      </div>

      {/* Embedded Analysis Panel */}
      <div className="glass-card p-6 rounded-3xl shadow-sm mb-8">
        <AnalysisPanel />
      </div>
    </div>
  );
}
