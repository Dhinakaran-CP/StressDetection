import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';

export default function PersonalInsights() {
  // Weekly stress baseline data
  const resilienceData = [
    { name: 'MON', stress: 30, calm: 70 },
    { name: 'TUE', stress: 45, calm: 55 },
    { name: 'WED', stress: 38, calm: 62 },
    { name: 'THU', stress: 55, calm: 45 },
    { name: 'FRI', stress: 28, calm: 72 },
    { name: 'SAT', stress: 18, calm: 82 },
    { name: 'SUN', stress: 12, calm: 88 },
  ];

  // Sleep vs Stress correlation data
  const correlationData = [
    { sleep: 5.0, stress: 78, size: 20 },
    { sleep: 5.5, stress: 70, size: 25 },
    { sleep: 6.0, stress: 62, size: 30 },
    { sleep: 6.5, stress: 50, size: 20 },
    { sleep: 7.0, stress: 38, size: 40 },
    { sleep: 7.5, stress: 32, size: 45 },
    { sleep: 8.0, stress: 20, size: 50 },
    { sleep: 8.5, stress: 14, size: 35 },
  ];

  // Circadian load (hour of day vs stress score)
  const hourlyLoad = [
    { hour: '08:00', load: 45, label: 'Morning Prep' },
    { hour: '10:00', load: 68, label: 'Peak Meeting load' },
    { hour: '12:00', load: 35, label: 'Midday Reset' },
    { hour: '14:00', load: 58, label: 'Afternoon Push' },
    { hour: '16:00', load: 25, label: 'Cognitive Recovery' },
    { hour: '18:00', load: 18, label: 'Wind-down' }
  ];

  return (
    <div className="space-y-8 select-none">
      {/* Upper Grid: Stress Resilience & Daily Goal */}
      <div className="grid grid-cols-12 gap-gutter">
        {/* Main Chart: Stress Resilience */}
        <section className="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-2xl p-8 shadow-[0_4px_20px_rgba(26,28,30,0.04)] hover:shadow-[0_8px_30px_rgba(26,28,30,0.08)] hover:-translate-y-0.5 transition-all duration-300">
          <div className="flex justify-between items-center mb-8">
            <div>
              <h3 className="font-headline-sm text-headline-sm text-primary">Stress Resilience</h3>
              <p className="text-on-surface-variant text-sm">Weekly performance baseline (Sympathetic Tone %)</p>
            </div>
            <span className="bg-surface-container-low px-3 py-1 rounded-full text-label-caps text-xs font-semibold text-primary">7 DAYS</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={resilienceData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorStress" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0e3b69" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#0e3b69" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="name" stroke="#737780" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#737780" fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    background: 'rgba(255,255,255,0.95)',
                    border: '1px solid rgba(14,59,105,0.1)',
                    borderRadius: '12px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
                  }}
                  labelStyle={{ fontWeight: 'bold', color: '#0e3b69' }}
                />
                <Area type="monotone" dataKey="stress" name="Stress Level" stroke="#0e3b69" strokeWidth={3} fillOpacity={1} fill="url(#colorStress)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-8 pt-8 border-t border-outline-variant/30 grid grid-cols-3 gap-4">
            <div>
              <p className="font-label-caps text-[10px] text-outline uppercase tracking-wider">Peak Stress</p>
              <p className="font-data-metric text-data-metric text-error mt-1">55%</p>
            </div>
            <div>
              <p className="font-label-caps text-[10px] text-outline uppercase tracking-wider">Min Stress</p>
              <p className="font-data-metric text-data-metric text-[#4ADE80] mt-1">12%</p>
            </div>
            <div>
              <p className="font-label-caps text-[10px] text-outline uppercase tracking-wider">Avg Variation</p>
              <p className="font-data-metric text-data-metric text-primary mt-1">±4%</p>
            </div>
          </div>
        </section>

        {/* Daily Goal Ring */}
        <section className="col-span-12 lg:col-span-4 bg-primary text-on-primary rounded-2xl p-8 flex flex-col items-center justify-center text-center shadow-md relative overflow-hidden">
          <div className="absolute -top-10 -right-10 w-32 h-32 bg-white/5 rounded-full blur-2xl"></div>
          <h3 className="font-label-caps text-[11px] uppercase tracking-widest mb-6 opacity-80">Daily Focus Goal</h3>
          <div className="relative w-40 h-40 flex items-center justify-center mb-6">
            <svg className="w-full h-full -rotate-90">
              <circle className="opacity-20" cx="80" cy="80" fill="transparent" r="70" stroke="currentColor" strokeWidth={10}></circle>
              <circle cx="80" cy="80" fill="transparent" r="70" stroke="currentColor" strokeDasharray={440} strokeDashoffset={110} strokeLinecap="round" strokeWidth={10}></circle>
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="font-data-metric text-3xl font-bold">75%</span>
              <span className="text-[10px] opacity-70 font-label-caps tracking-wider">Complete</span>
            </div>
          </div>
          <p className="font-body-md text-sm font-semibold">15 / 20 Calm Minutes</p>
          <p className="text-xs opacity-60 mt-1">Keep breathing slowly for 5 more minutes.</p>
          <button 
            id="resume-btn"
            onClick={() => {
              const btn = document.getElementById('resume-btn');
              if (btn) {
                const original = btn.innerText;
                btn.innerText = 'RESUMING...';
                btn.classList.add('bg-white/30');
                setTimeout(() => {
                  btn.innerText = original;
                  btn.classList.remove('bg-white/30');
                }, 1500);
              }
            }}
            className="mt-8 bg-white/10 hover:bg-white/20 transition-all border border-white/20 px-6 py-2.5 rounded-full font-label-caps text-xs tracking-wider active:scale-95"
          >
            RESUME SESSION
          </button>
        </section>
      </div>

      {/* Mid Grid: Averages, Triggers, Streaks */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-gutter">
        {/* Weekly Stats */}
        <div className="bg-surface-container-lowest rounded-2xl p-6 shadow-[0_4px_20px_rgba(26,28,30,0.04)] hover:shadow-[0_8px_30px_rgba(26,28,30,0.08)] hover:-translate-y-0.5 transition-all duration-300">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-surface-container rounded-lg p-2 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary">analytics</span>
            </div>
            <h4 className="font-label-caps text-[11px] text-outline uppercase tracking-wider">Weekly Averages</h4>
          </div>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-end mb-1 text-xs font-semibold">
                <span className="text-on-surface-variant">Stress Score</span>
                <span className="font-data-metric text-primary">32<span className="text-[10px] text-outline-variant">/100</span></span>
              </div>
              <div className="w-full bg-surface-container h-2 rounded-full overflow-hidden">
                <div className="bg-primary h-full rounded-full" style={{ width: '32%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-end mb-1 text-xs font-semibold">
                <span className="text-on-surface-variant">Calm Minutes</span>
                <span className="font-data-metric text-primary">124 <span className="text-[10px] text-outline-variant">min</span></span>
              </div>
              <div className="w-full bg-surface-container h-2 rounded-full overflow-hidden">
                <div className="bg-primary-container h-full rounded-full" style={{ width: '65%' }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* AI Identified Triggers */}
        <div className="bg-surface-container-lowest rounded-2xl p-6 shadow-[0_4px_20px_rgba(26,28,30,0.04)] hover:shadow-[0_8px_30px_rgba(26,28,30,0.08)] hover:-translate-y-0.5 transition-all duration-300">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-error-container/20 rounded-lg p-2 flex items-center justify-center">
              <span className="material-symbols-outlined text-error">psychology_alt</span>
            </div>
            <h4 className="font-label-caps text-[11px] text-outline uppercase tracking-wider">AI Identified Triggers</h4>
          </div>
          <ul className="space-y-4">
            <li className="flex items-center justify-between text-sm font-semibold">
              <div className="flex items-center gap-3">
                <div className="w-2.5 h-2.5 rounded-full bg-error"></div>
                <span className="text-on-surface-variant">Morning Meetings</span>
              </div>
              <span className="text-xs font-bold text-error">+22%</span>
            </li>
            <li className="flex items-center justify-between text-sm font-semibold">
              <div className="flex items-center gap-3">
                <div className="w-2.5 h-2.5 rounded-full bg-secondary"></div>
                <span className="text-on-surface-variant">Late Night Screen</span>
              </div>
              <span className="text-xs font-bold text-on-surface-variant">+14%</span>
            </li>
            <li className="flex items-center justify-between text-sm font-semibold">
              <div className="flex items-center gap-3">
                <div className="w-2.5 h-2.5 rounded-full bg-outline"></div>
                <span className="text-on-surface-variant">Caffeine Intake</span>
              </div>
              <span className="text-xs font-bold text-on-surface-variant">+8%</span>
            </li>
          </ul>
        </div>

        {/* Streaks and Badges */}
        <div className="bg-surface-container-lowest rounded-2xl p-6 shadow-[0_4px_20px_rgba(26,28,30,0.04)] hover:shadow-[0_8px_30px_rgba(26,28,30,0.08)] hover:-translate-y-0.5 transition-all duration-300">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-surface-container-high rounded-lg p-2 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary">military_tech</span>
            </div>
            <h4 className="font-label-caps text-[11px] text-outline uppercase tracking-wider">Active Streaks</h4>
          </div>
          <div className="flex justify-around items-center pt-2">
            <div className="flex flex-col items-center gap-2 group cursor-pointer">
              <div className="w-12 h-12 rounded-full bg-primary-fixed flex items-center justify-center text-primary group-hover:scale-110 transition-transform">
                <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>local_fire_department</span>
              </div>
              <span className="text-[10px] font-bold uppercase tracking-tight font-label-caps text-on-surface">14 Days</span>
            </div>
            <div className="flex flex-col items-center gap-2 group cursor-pointer opacity-50 hover:opacity-100 transition-opacity">
              <div className="w-12 h-12 rounded-full bg-surface-container-high flex items-center justify-center text-outline-variant group-hover:scale-110 transition-transform">
                <span className="material-symbols-outlined">spa</span>
              </div>
              <span className="text-[10px] font-bold uppercase tracking-tight font-label-caps text-on-surface">Zen Master</span>
            </div>
            <div className="flex flex-col items-center gap-2 group cursor-pointer">
              <div className="w-12 h-12 rounded-full bg-tertiary-fixed flex items-center justify-center text-tertiary group-hover:scale-110 transition-transform">
                <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>nights_stay</span>
              </div>
              <span className="text-[10px] font-bold uppercase tracking-tight font-label-caps text-on-surface">Deep Rest</span>
            </div>
          </div>
        </div>
      </div>

      {/* Advanced Research: Circadian Chronotype Stress Map & Sleep Correlation */}
      <div className="grid grid-cols-12 gap-gutter mt-8">
        {/* Circadian Hourly Map */}
        <section className="col-span-12 lg:col-span-6 bg-surface-container-lowest rounded-2xl p-8 shadow-[0_4px_20px_rgba(26,28,30,0.04)] hover:shadow-[0_8px_30px_rgba(26,28,30,0.08)] hover:-translate-y-0.5 transition-all duration-300">
          <div className="mb-6">
            <h3 className="font-headline-sm text-[20px] text-primary">Diurnal Stress Timeline</h3>
            <p className="text-on-surface-variant text-sm">Hourly chronotype load map</p>
          </div>

          <div className="space-y-4">
            {hourlyLoad.map((item, index) => (
              <div key={index} className="flex items-center gap-4">
                <span className="font-data-metric text-xs text-outline w-12">{item.hour}</span>
                <div className="flex-1 space-y-1">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-on-surface-variant">{item.label}</span>
                    <span className={item.load > 60 ? 'text-error' : 'text-primary'}>{item.load}%</span>
                  </div>
                  <div className="h-2 w-full bg-surface-container rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        item.load > 60 ? 'bg-error' : 'bg-primary'
                      }`}
                      style={{ width: `${item.load}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Sleep vs Stress Scatter Correlation */}
        <section className="col-span-12 lg:col-span-6 bg-surface-container-lowest rounded-2xl p-8 shadow-[0_4px_20px_rgba(26,28,30,0.04)] hover:shadow-[0_8px_30px_rgba(26,28,30,0.08)] hover:-translate-y-0.5 transition-all duration-300">
          <div className="mb-6">
            <h3 className="font-headline-sm text-[20px] text-primary">Sleep-Stress Correlation</h3>
            <p className="text-on-surface-variant text-sm">Empirical mapping of sleep duration (hours) vs stress load (%)</p>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis type="number" dataKey="sleep" name="Sleep" unit="h" stroke="#737780" fontSize={11} tickLine={false} axisLine={false} domain={[4, 9]} />
                <YAxis type="number" dataKey="stress" name="Stress" unit="%" stroke="#737780" fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} />
                <ZAxis type="number" dataKey="size" range={[50, 400]} />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{
                    background: 'rgba(255,255,255,0.95)',
                    border: '1px solid rgba(14,59,105,0.1)',
                    borderRadius: '12px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
                  }}
                />
                <Scatter name="Correlation Point" data={correlationData} fill="#0e3b69" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* Call to Action Banner */}
      <section className="mt-section-gap relative rounded-3xl overflow-hidden p-10 bg-on-background text-on-primary shadow-xl">
        <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="max-w-xl">
            <h2 className="font-headline-md text-headline-sm text-surface-bright mb-3">Sympathetic Nervous Recovery Analysis</h2>
            <p className="font-body-lg text-sm text-surface-variant/80 leading-relaxed">
              Our clinical models suggest that your resilience peaks between 2:00 PM and 4:00 PM. We recommend scheduling high-intensity tasks during this window to preserve cognitive load capacity.
            </p>
          </div>
          <button 
            id="report-btn"
            onClick={() => {
              const btn = document.getElementById('report-btn');
              if (btn) {
                const original = btn.innerHTML;
                btn.innerHTML = 'GENERATING REPORT... <span class="material-symbols-outlined text-sm animate-spin">sync</span>';
                btn.classList.add('opacity-80');
                setTimeout(() => {
                  btn.innerHTML = original;
                  btn.classList.remove('opacity-80');
                }, 2000);
              }
            }}
            className="bg-primary-container text-white px-8 py-3.5 rounded-xl font-bold text-xs tracking-wider font-label-caps hover:opacity-95 active:scale-95 transition-all flex items-center gap-2 whitespace-nowrap shadow-lg"
          >
            VIEW DETAILED REPORT
            <span className="material-symbols-outlined text-sm">arrow_forward</span>
          </button>
        </div>
        <div className="absolute -right-24 -bottom-24 w-64 h-64 bg-primary-container/10 rounded-full blur-3xl"></div>
      </section>
    </div>
  );
}
