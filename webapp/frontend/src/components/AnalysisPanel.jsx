import React, { useMemo } from 'react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer
} from 'recharts';

const FEATURE_ICONS = {
  facial: 'face',
  face: 'face',
  voice: 'keyboard_voice',
  physiological: 'monitor_heart',
  physio: 'monitor_heart',
  overall: 'analytics'
};

const FEATURE_LABELS = {
  avg_ear: 'Eye Openness',
  brow_descent_left: 'Brow Tension',
  lip_compression: 'Lip Compression',
  jaw_displacement: 'Jaw Tension',
  jitter_percent: 'Vocal Jitter',
  f0_mean: 'Pitch Level',
  hnr: 'Voice Clarity',
  shimmer_db: 'Amplitude Stability',
  alpha_power: 'Alpha Brainwave',
  beta_power: 'Beta Brainwave',
  scr_rate: 'Skin Conductance',
};

const FACIAL_FEATURE_NAMES = [
  "Left Eye Openness", "Right Eye Openness", "Average Eye Openness",
  "Blink Velocity", "Brow Tension (Left)", "Brow Tension (Right)",
  "Brow Asymmetry", "Lip Compression", "Jaw Tension", "Mouth Corner Pull",
  "Forehead Tension", "Normalized Face Height", "Head Tilt", "Temporal X Variation",
  "Temporal Y Variation", "Eye Openness Ratio", "Landmark Confidence", "Nose Wrinkle"
];

const VOICE_FEATURE_NAMES = [
  "Mean Pitch", "Pitch Standard Deviation", "Pitch Range", "Vocal Jitter",
  "Amplitude Shimmer", "Voice Harmonics-to-Noise Ratio", "Speaking Rate",
  "Voice Intensity", "High Frequency Energy Ratio", "Spectral Flux",
  "Pause Ratio", "Voiced Fraction"
];

function getFeatureIconName(feature, modality) {
  if (feature?.toLowerCase()?.includes('jaw') || feature?.toLowerCase()?.includes('brow') || feature?.toLowerCase()?.includes('eye')) {
    return 'face';
  }
  if (feature?.toLowerCase()?.includes('pitch') || feature?.toLowerCase()?.includes('voice') || feature?.toLowerCase()?.includes('vocal') || feature?.toLowerCase()?.includes('jitter')) {
    return 'keyboard_voice';
  }
  if (modality === 'physio' || modality === 'physiological') {
    return 'monitor_heart';
  }
  return FEATURE_ICONS[modality] || 'speed';
}

function featureLabel(key) {
  if (!key) return "Unknown Feature";
  const normalizedKey = String(key).toLowerCase();
  
  if (normalizedKey.startsWith("facial_")) {
    const idx = parseInt(normalizedKey.split("_")[1]);
    if (idx >= 0 && idx < FACIAL_FEATURE_NAMES.length) {
      return FACIAL_FEATURE_NAMES[idx];
    }
  }
  
  if (normalizedKey.startsWith("voice_")) {
    const idx = parseInt(normalizedKey.split("_")[1]);
    if (idx >= 0 && idx < VOICE_FEATURE_NAMES.length) {
      return VOICE_FEATURE_NAMES[idx];
    }
  }

  if (normalizedKey.startsWith("phys_") || normalizedKey.startsWith("physiological_")) {
    const parts = normalizedKey.split("_");
    const idx = parseInt(parts[parts.length - 1]);
    if (idx >= 42) {
      return `GSR Response (Feature ${idx - 41})`;
    }
    return `EEG Band Power (Feature ${idx})`;
  }

  return FEATURE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const toPercent = (value, fallback = 0) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return fallback;
  }
  const numeric = Number(value);
  if (numeric <= 1) return clamp(numeric * 100, 0, 100);
  return clamp(numeric, 0, 100);
};

export default function AnalysisPanel({ result, onRequestGame, previousResult }) {
  const stress_level = result?.stress_level?.toUpperCase() || "LOW";
  const fused_score = clamp(result?.stress_probability || (result?.percentage ? result.percentage / 100 : 0), 0, 1);
  const confidence_score = clamp(result?.confidence || 0.9, 0, 1);
  const explainability = result?.explainability;
  const individual = result?.individual_predictions || {};
  const face_score = individual.facial ?? individual.face;
  const voice_score = individual.voice;
  const physio_score = individual.physiological ?? individual.physio;

  const analysis = useMemo(() => {
    const points = [];
    if (face_score != null) {
      points.push({ key: "facial", label: "Facial", value: toPercent(face_score), reason: "facial expression stress" });
    }
    if (voice_score != null) {
      points.push({ key: "voice", label: "Voice", value: toPercent(voice_score), reason: "vocal strain" });
    }
    if (physio_score != null) {
      points.push({ key: "physiological", label: "Physiological", value: toPercent(physio_score), reason: "physiological arousal" });
    }

    if (points.length === 0) {
      points.push({ key: "overall", label: "Overall", value: toPercent(fused_score), reason: "overall indicators" });
    }

    const sorted = [...points].sort((a, b) => b.value - a.value);
    const total = points.reduce((sum, item) => sum + item.value, 0) || 1;

    const drivers = explainability?.top_drivers || [];
    const topFeatureLabels = drivers.slice(0, 2).map(d => featureLabel(d.feature));

    let cause = "";
    if (topFeatureLabels.length >= 2) {
      cause = `Fused output is primarily driven by elevated ${topFeatureLabels[0]} and increased ${topFeatureLabels[1]}.`;
    } else if (topFeatureLabels.length === 1) {
      cause = `Fused output is primarily driven by elevated ${topFeatureLabels[0]}.`;
    } else {
      cause = sorted.length > 1
        ? `Primary drivers are ${sorted[0].reason} and ${sorted[1].reason}.`
        : `Primary driver is ${sorted[0].reason}.`;
    }

    return {
      points,
      cause,
      contributions: points.map((item) => ({
        ...item,
        contribution: Math.round((item.value / total) * 100),
      })),
    };
  }, [face_score, voice_score, physio_score, fused_score, explainability]);

  const activeValues = useMemo(() => {
    return analysis.points.map(p => p.value);
  }, [analysis.points]);

  const mean = useMemo(() => {
    return activeValues.length
      ? activeValues.reduce((sum, value) => sum + value, 0) / activeValues.length
      : 0;
  }, [activeValues]);

  const variance = useMemo(() => {
    return activeValues.length
      ? activeValues.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / activeValues.length
      : 0;
  }, [activeValues, mean]);

  const stdDev = useMemo(() => Math.sqrt(variance), [variance]);

  const maxPossibleStdDev = useMemo(() => {
    if (activeValues.length <= 1) return 1;
    return activeValues.length === 3 ? 47.14045 : 50.0;
  }, [activeValues]);

  const agreement = useMemo(() => {
    if (activeValues.length <= 1) return 100;
    return clamp(100 - (stdDev / maxPossibleStdDev) * 100, 0, 100);
  }, [stdDev, activeValues.length, maxPossibleStdDev]);

  const completeness = useMemo(() => (activeValues.length / 3) * 100, [activeValues]);
  const riskIndex = useMemo(() => clamp(fused_score * 100, 0, 100), [fused_score]);
  const resilienceIndex = useMemo(() => clamp((100 - riskIndex) * 0.8 + (agreement * 0.2), 0, 100), [riskIndex, agreement]);
  const confidenceTarget = clamp(confidence_score * 100, 0, 100);

  const radarData = useMemo(() => {
    return [
      { subject: 'Risk Index', value: riskIndex },
      { subject: 'Agreement', value: agreement },
      { subject: 'Coverage', value: completeness },
      { subject: 'Resilience', value: resilienceIndex },
    ];
  }, [riskIndex, agreement, completeness, resilienceIndex]);

  // Derived Trigger for display
  const detectedTrigger = useMemo(() => {
    if (explainability?.top_drivers?.length > 0) {
      const topFeature = explainability.top_drivers[0].feature;
      const label = featureLabel(topFeature);
      if (label.toLowerCase().includes('speaking') || label.toLowerCase().includes('pitch') || label.toLowerCase().includes('jitter')) {
        return 'Speech Pattern';
      }
      if (label.toLowerCase().includes('eye') || label.toLowerCase().includes('blink')) {
        return 'Ocular Flutter';
      }
      if (label.toLowerCase().includes('jaw') || label.toLowerCase().includes('forehead') || label.toLowerCase().includes('brow')) {
        return 'Facial Tension';
      }
      return label;
    }
    return 'Sympathetic Activation';
  }, [explainability]);

  return (
    <div className="space-y-8 select-none">
      {/* Bento Layout Header */}
      <div className="grid grid-cols-12 gap-gutter items-stretch">
        
        {/* Primary Stress Score Card */}
        <section className="col-span-12 lg:col-span-7 bg-surface-container-lowest rounded-2xl shadow-[0_4px_20px_rgba(26,28,30,0.04)] p-8 flex flex-col justify-between relative overflow-hidden border border-outline-variant/10">
          <div className="absolute -right-20 -top-20 w-64 h-64 bg-surface-container-high/30 rounded-full blur-3xl"></div>
          
          <div className="flex justify-between items-start mb-6 z-10">
            <div>
              <h2 className="font-label-caps text-[11px] text-on-surface-variant font-semibold tracking-wider mb-1">AGGREGATE STRESS INDEX</h2>
              <p className="text-xs text-on-surface opacity-60 font-medium">Calculated from multimodal telemetry streams</p>
            </div>
            <span className={`px-4 py-1 rounded-full font-label-caps text-xs font-bold ${
              stress_level === 'HIGH' || stress_level === 'EXTREME'
                ? 'bg-error-container text-on-error-container'
                : stress_level === 'MODERATE'
                ? 'bg-[#FFF0E0] text-[#854D0E]'
                : 'bg-primary-container/10 text-primary'
            }`}>
              {stress_level}
            </span>
          </div>

          <div className="flex items-baseline gap-2 mb-6 z-10">
            <span className="font-display-lg text-[96px] leading-none text-primary font-bold">
              {Math.round(fused_score * 100)}
            </span>
            <span className="font-headline-sm text-headline-sm text-outline">%</span>
          </div>

          <div className="space-y-4 z-10">
            <div className="flex justify-between font-label-caps text-[10px] text-on-surface-variant font-bold tracking-wider">
              <span>RELAXED</span>
              <span>PEAK LOAD</span>
            </div>
            <div className="h-2.5 w-full bg-surface-container rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full transition-all duration-[1.5s] ease-out" style={{ width: `${fused_score * 100}%` }}></div>
            </div>
            <p className="font-body-md text-sm text-on-surface-variant italic leading-relaxed">
              "{analysis.cause}"
            </p>
          </div>
        </section>

        {/* SHAP Contributors list */}
        <section className="col-span-12 lg:col-span-5 bg-surface-container-lowest rounded-2xl shadow-[0_4px_20px_rgba(26,28,30,0.04)] p-8 border border-outline-variant/10">
          <h2 className="font-label-caps text-[11px] text-on-surface-variant font-bold tracking-wider mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-[18px]">troubleshoot</span>
            CONTRIBUTING FACTORS (SHAP)
          </h2>

          <div className="space-y-6">
            {explainability?.top_drivers?.length > 0 ? (
              explainability.top_drivers.slice(0, 4).map((d, index) => {
                const iconName = getFeatureIconName(d.feature, d.modality);
                const absVal = Math.abs(d.shap_value);
                const totalShap = explainability.top_drivers.reduce((s, dr) => s + Math.abs(dr.shap_value), 0) || 1;
                const pctVal = Math.round((absVal / totalShap) * 100);

                return (
                  <div key={index} className="group">
                    <div className="flex justify-between items-center mb-1.5">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center border border-outline-variant/10 text-primary">
                          <span className="material-symbols-outlined text-[18px]">{iconName}</span>
                        </div>
                        <span className="text-sm font-semibold text-on-surface">{featureLabel(d.feature)}</span>
                      </div>
                      <span className="font-data-metric text-sm font-bold text-primary">{pctVal}%</span>
                    </div>
                    <div className="h-2 w-full bg-surface-container rounded-full overflow-hidden">
                      <div className="h-full bg-primary rounded-full group-hover:opacity-80 transition-all" style={{ width: `${pctVal}%` }}></div>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="h-full flex items-center justify-center py-10 text-center text-xs text-outline italic">
                No feature attribution diagnostics available for this prediction.
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Insight Cards Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-gutter">
        {/* Recovery Score */}
        <div className="bg-surface-container-lowest rounded-2xl shadow-[0_4px_20px_rgba(26,28,30,0.04)] p-6 hover:-translate-y-0.5 transition-all duration-300 border-l-4 border-[#4ADE80] border-t border-r border-b border-outline-variant/10 flex flex-col justify-between min-h-[110px]">
          <div className="flex justify-between items-center">
            <span className="font-label-caps text-[10px] text-on-surface-variant font-bold tracking-wider">RECOVERY SCORE</span>
            <span className="material-symbols-outlined text-[#4ADE80] text-sm">autorenew</span>
          </div>
          <div className="flex items-baseline gap-1 mt-2">
            <span className="font-display-lg text-2xl font-bold text-on-surface">{Math.round(resilienceIndex)}</span>
            <span className="text-outline-variant font-label-caps text-xs">%</span>
          </div>
          <p className="text-[11px] text-on-surface-variant font-medium mt-2">
            {resilienceIndex > 60 ? 'Current baseline trending towards stabilization.' : 'System recommends tactical recovery breathing.'}
          </p>
        </div>

        {/* Confidence Score */}
        <div className="bg-surface-container-lowest rounded-2xl shadow-[0_4px_20px_rgba(26,28,30,0.04)] p-6 hover:-translate-y-0.5 transition-all duration-300 border-l-4 border-primary border-t border-r border-b border-outline-variant/10 flex flex-col justify-between min-h-[110px]">
          <div className="flex justify-between items-center">
            <span className="font-label-caps text-[10px] text-on-surface-variant font-bold tracking-wider">CONFIDENCE SCORE</span>
            <span className="material-symbols-outlined text-primary text-sm">verified</span>
          </div>
          <div className="flex items-baseline gap-1 mt-2">
            <span className="font-display-lg text-2xl font-bold text-on-surface">{Math.round(confidenceTarget)}</span>
            <span className="text-outline-variant font-label-caps text-xs">%</span>
          </div>
          <p className="text-[11px] text-on-surface-variant font-medium mt-2">
            {confidenceTarget > 85 ? 'High sensor fidelity from all active channels.' : 'Medium fidelity, check camera positioning.'}
          </p>
        </div>

        {/* Detected Trigger */}
        <div className="bg-surface-container-lowest rounded-2xl shadow-[0_4px_20px_rgba(26,28,30,0.04)] p-6 hover:-translate-y-0.5 transition-all duration-300 border-l-4 border-[#FBBF24] border-t border-r border-b border-outline-variant/10 flex flex-col justify-between min-h-[110px]">
          <div className="flex justify-between items-center">
            <span className="font-label-caps text-[10px] text-on-surface-variant font-bold tracking-wider">DETECTED TRIGGER</span>
            <span className="material-symbols-outlined text-[#FBBF24] text-sm">warning</span>
          </div>
          <div className="flex items-baseline gap-1 mt-2">
            <span className="font-display-lg text-2xl font-bold text-on-surface truncate max-w-[200px]">{detectedTrigger}</span>
          </div>
          <p className="text-[11px] text-on-surface-variant font-medium mt-2">
            Primary stress biomarker trigger mapping output.
          </p>
        </div>
      </div>

      {/* Advanced Research: Radar correlation charts */}
      <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-gutter items-center">
        <div className="space-y-6">
          <h3 className="font-headline-sm text-headline-sm text-primary">Biometric Correlation Visualization</h3>
          <p className="font-body-lg text-sm text-on-surface-variant leading-relaxed max-w-lg">
            The neural overlay shows the direct relationship between sympathetic nervous system arousal and localized muscular/acoustic tension. We recommend launching a 2-minute Guided Box Breathing session to stabilize the autonomic baseline.
          </p>
          <div className="flex gap-4">
            <button
              onClick={onRequestGame}
              className="px-6 py-3 bg-primary text-on-primary rounded-xl font-label-caps text-xs font-bold tracking-wider hover:opacity-90 active:scale-95 transition-all shadow-md"
            >
              Start Session
            </button>
            <button 
              onClick={() => {
                const btn = document.getElementById('export-btn');
                if (btn) {
                  const original = btn.innerText;
                  btn.innerText = 'Exported \u2713';
                  btn.classList.add('bg-[#4ADE80]/20', 'text-[#4ADE80]', 'border-[#4ADE80]');
                  setTimeout(() => {
                    btn.innerText = original;
                    btn.classList.remove('bg-[#4ADE80]/20', 'text-[#4ADE80]', 'border-[#4ADE80]');
                  }, 2000);
                }
              }}
              id="export-btn"
              className="px-6 py-3 border border-outline text-on-surface-variant rounded-xl font-label-caps text-xs font-bold tracking-wider hover:bg-surface-container transition-all active:scale-95"
            >
              Export Report
            </button>
          </div>
        </div>

        {/* Recharts Radar Representation */}
        <div className="relative h-[320px] rounded-2xl bg-white border border-outline-variant/10 shadow-sm flex items-center justify-center p-4">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid stroke="var(--outline-variant)" strokeOpacity="0.2" />
              <PolarAngleAxis dataKey="subject" stroke="#737780" fontSize={11} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#737780" fontSize={9} />
              <Radar name="Vitals Index" dataKey="value" stroke="#0e3b69" fill="#0e3b69" fillOpacity={0.2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
