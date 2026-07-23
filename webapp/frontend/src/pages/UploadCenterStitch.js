import React, { useState } from 'react';

export default function UploadCenterStitch({ onNavigate }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileType, setFileType] = useState('video'); // 'video' | 'audio' | 'eeg' | 'gsr'
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [resultSummary, setResultSummary] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleStartAnalysis = () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setUploadProgress(20);

    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsProcessing(false);
          setResultSummary({
            stressLevel: 'Mild Stress (24%)',
            confidence: '98.6%',
            hrv: '62 ms',
            recommendation: 'Optimal equilibrium. Recommended 3-min breathing pause.'
          });
          return 100;
        }
        return prev + 20;
      });
    }, 600);
  };

  return (
    <div className="w-full max-w-[1440px] mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
        <div>
          <h2 className="font-display-hero text-3xl font-bold text-on-background">Bio-Data Upload Center</h2>
          <p className="font-body-md text-sm text-on-surface-variant mt-1">
            Batch process physiological telemetry files for multi-modal neural network inference
          </p>
        </div>

        <button
          onClick={() => onNavigate && onNavigate('dashboard')}
          className="px-4 py-2.5 bg-surface-container text-primary rounded-xl font-label-caps text-xs font-semibold hover:bg-surface-container-high transition-all flex items-center gap-2 border border-primary/10 self-start md:self-auto"
        >
          <span className="material-symbols-outlined text-base">dashboard</span>
          <span>Back to Dashboard</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Upload Dropzone Card */}
        <div className="glass-card p-6 rounded-3xl lg:col-span-2 shadow-sm space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="font-display-hero text-xl font-bold text-on-background">Select Telemetry Dataset</h3>
            <span className="px-3 py-1 bg-primary/10 text-primary rounded-full font-label-caps text-[10px] font-bold">
              HIPAA Encrypted
            </span>
          </div>

          {/* File Type Selector */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <button
              onClick={() => setFileType('video')}
              className={`p-3 rounded-2xl border flex flex-col items-center gap-2 transition-all ${
                fileType === 'video' ? 'bg-primary-container/20 border-primary text-primary font-bold' : 'bg-surface-container-low border-primary/10 text-on-surface-variant'
              }`}
            >
              <span className="material-symbols-outlined text-xl">videocam</span>
              <span className="font-label-caps text-xs">Video (MP4/WebM)</span>
            </button>

            <button
              onClick={() => setFileType('audio')}
              className={`p-3 rounded-2xl border flex flex-col items-center gap-2 transition-all ${
                fileType === 'audio' ? 'bg-primary-container/20 border-primary text-primary font-bold' : 'bg-surface-container-low border-primary/10 text-on-surface-variant'
              }`}
            >
              <span className="material-symbols-outlined text-xl">mic</span>
              <span className="font-label-caps text-xs">Vocal (WAV/MP3)</span>
            </button>

            <button
              onClick={() => setFileType('eeg')}
              className={`p-3 rounded-2xl border flex flex-col items-center gap-2 transition-all ${
                fileType === 'eeg' ? 'bg-primary-container/20 border-primary text-primary font-bold' : 'bg-surface-container-low border-primary/10 text-on-surface-variant'
              }`}
            >
              <span className="material-symbols-outlined text-xl">ecg</span>
              <span className="font-label-caps text-xs">EEG (CSV/EDF)</span>
            </button>

            <button
              onClick={() => setFileType('gsr')}
              className={`p-3 rounded-2xl border flex flex-col items-center gap-2 transition-all ${
                fileType === 'gsr' ? 'bg-primary-container/20 border-primary text-primary font-bold' : 'bg-surface-container-low border-primary/10 text-on-surface-variant'
              }`}
            >
              <span className="material-symbols-outlined text-xl">sensors</span>
              <span className="font-label-caps text-xs">GSR Telemetry</span>
            </button>
          </div>

          {/* Drag & Drop Area */}
          <div className="relative border-2 border-dashed border-primary/20 hover:border-primary/50 rounded-3xl p-10 text-center transition-all bg-surface-container-lowest flex flex-col items-center justify-center gap-4">
            <input
              type="file"
              onChange={handleFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            />
            <div className="w-16 h-16 rounded-full bg-primary/10 text-primary flex items-center justify-center material-symbols-outlined text-3xl">
              cloud_upload
            </div>
            <div>
              <p className="font-display-hero text-base font-bold text-on-background">
                {selectedFile ? selectedFile.name : 'Drag & drop bio-data file here'}
              </p>
              <p className="text-xs text-on-surface-variant mt-1">
                {selectedFile ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB` : 'or click to browse local filesystem'}
              </p>
            </div>
          </div>

          {/* Progress bar during analysis */}
          {isProcessing && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-bold text-primary">
                <span>Executing Pipeline Extraction...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                <div className="bg-primary h-full transition-all duration-300" style={{ width: `${uploadProgress}%` }}></div>
              </div>
            </div>
          )}

          <div className="flex justify-end gap-3 pt-2">
            <button
              onClick={() => setSelectedFile(null)}
              className="px-4 py-2.5 border border-outline-variant/30 text-on-surface-variant rounded-xl font-label-caps text-xs font-semibold hover:bg-surface-container transition-all"
            >
              Clear
            </button>
            <button
              onClick={handleStartAnalysis}
              disabled={!selectedFile || isProcessing}
              className={`px-6 py-2.5 rounded-xl font-label-caps text-xs font-semibold flex items-center gap-2 shadow-lg transition-all ${
                selectedFile && !isProcessing
                  ? 'bg-primary text-on-primary hover:shadow-primary/20'
                  : 'bg-surface-container-high text-on-surface-variant/50 cursor-not-allowed'
              }`}
            >
              <span className="material-symbols-outlined text-base">play_arrow</span>
              <span>Start Analysis</span>
            </button>
          </div>
        </div>

        {/* Upload Status & Results Sidebar */}
        <div className="glass-card p-6 rounded-3xl shadow-sm flex flex-col justify-between space-y-6">
          <div>
            <h3 className="font-display-hero text-xl font-bold text-on-background">Inference Output</h3>
            <p className="text-xs text-on-surface-variant mt-0.5">Automated Bio-Feature Results</p>
          </div>

          {resultSummary ? (
            <div className="space-y-4">
              <div className="p-4 bg-primary/10 border border-primary/20 rounded-2xl space-y-2">
                <p className="font-label-caps text-[10px] text-primary uppercase font-bold">Predicted Stress State</p>
                <p className="font-display-hero text-lg font-bold text-primary">{resultSummary.stressLevel}</p>
                <p className="text-xs text-on-surface-variant">Model Confidence: <strong>{resultSummary.confidence}</strong></p>
              </div>

              <div className="p-4 bg-surface-container-low rounded-2xl border border-primary/10 space-y-1">
                <p className="font-label-caps text-[10px] text-on-surface-variant uppercase font-semibold">Average HRV</p>
                <p className="font-display-hero text-base font-bold text-on-background">{resultSummary.hrv}</p>
              </div>

              <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl space-y-1">
                <p className="font-label-caps text-[10px] text-emerald-700 uppercase font-bold">Action Plan</p>
                <p className="text-xs text-on-surface-variant">{resultSummary.recommendation}</p>
              </div>
            </div>
          ) : (
            <div className="p-8 text-center bg-surface-container-low rounded-2xl border border-primary/10 text-on-surface-variant space-y-3">
              <span className="material-symbols-outlined text-3xl opacity-50">hourglass_empty</span>
              <p className="text-xs font-medium">Upload a file and click "Start Analysis" to generate inference scores.</p>
            </div>
          )}

          <button
            onClick={() => onNavigate && onNavigate('multimodal')}
            className="w-full py-3 bg-surface-container text-primary rounded-xl font-label-caps text-xs font-bold hover:bg-surface-container-high transition-all flex items-center justify-center gap-2 border border-primary/10"
          >
            <span>View Full Pipeline</span>
            <span className="material-symbols-outlined text-base">arrow_forward</span>
          </button>
        </div>
      </div>
    </div>
  );
}
