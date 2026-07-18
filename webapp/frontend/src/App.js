import React, { useState, useEffect } from 'react';
import LoginConsent from './components/LoginConsent';
import SideNavBar from './components/SideNavBar';
import TopAppBar from './components/TopAppBar';
import Dashboard from './pages/Dashboard';
import PersonalInsights from './components/PersonalInsights';
import RecoveryActivities from './components/RecoveryActivities';
import CalibrationWizard from './components/CalibrationWizard';

function App() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('vitalmind-user');
    return saved ? JSON.parse(saved) : null;
  });

  const [activeView, setActiveView] = useState('dashboard'); // 'dashboard', 'insights', 'recovery', 'profile'
  const [dashboardMode, setDashboardMode] = useState('realtime'); // 'realtime' or 'upload'
  const [showCopilot, setShowCopilot] = useState(false);
  const [calibrationModal, setCalibrationModal] = useState(false);

  useEffect(() => {
    if (user) {
      localStorage.setItem('vitalmind-user', JSON.stringify(user));
    } else {
      localStorage.removeItem('vitalmind-user');
    }
  }, [user]);

  const handleLogin = (userData) => {
    setUser(userData);
    setActiveView('dashboard');
  };

  const handleLogout = () => {
    setUser(null);
    setActiveView('dashboard');
  };

  // If not logged in, render the Login / Consent security wall
  if (!user) {
    return <LoginConsent onLogin={handleLogin} />;
  }

  // Determine Title for TopAppBar
  let viewTitle = 'Dashboard';
  if (activeView === 'insights') viewTitle = 'Insights Summary';
  if (activeView === 'recovery') viewTitle = 'Recovery & Resilience';
  if (activeView === 'profile') viewTitle = 'User Profile';

  return (
    <div className="bg-warm-sand text-on-surface font-body-md min-h-screen flex text-sm transition-all duration-300">
      
      {/* Permanent Side Navigation bar */}
      <SideNavBar
        activeView={activeView}
        setActiveView={setActiveView}
        onCalibrateTrigger={() => setCalibrationModal(true)}
        user={user}
      />

      {/* Global Top App Bar & Main Content */}
      <div className="flex-1 flex flex-col pl-64">
        
        <TopAppBar
          title={viewTitle}
          activeView={activeView}
          dashboardMode={dashboardMode}
          setDashboardMode={setDashboardMode}
          showCopilot={showCopilot}
          setShowCopilot={setShowCopilot}
        />

        {/* Content Canvas */}
        <main className="flex-1 mt-16 p-12 bg-transparent overflow-x-hidden relative">
          
          {/* Atmospheric ambient highlights */}
          <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden select-none">
            <div className="absolute top-[-10%] right-[-10%] w-[600px] h-[600px] bg-surface-variant/20 rounded-full blur-[120px]"></div>
            <div className="absolute bottom-[-5%] left-[5%] w-[400px] h-[400px] bg-secondary-container/30 rounded-full blur-[100px]"></div>
          </div>

          <div className="relative z-10 max-w-container-max mx-auto">
            {activeView === 'dashboard' && (
              <Dashboard
                dashboardMode={dashboardMode}
                showCopilot={showCopilot}
                setShowCopilot={setShowCopilot}
                onRequestRecovery={() => setActiveView('recovery')}
              />
            )}

            {activeView === 'insights' && <PersonalInsights />}

            {activeView === 'recovery' && <RecoveryActivities />}

            {activeView === 'profile' && (
              <section className="bg-surface-container-lowest rounded-3xl p-8 border border-outline-variant/10 shadow-sm max-w-2xl mx-auto space-y-6">
                <div className="flex items-center gap-6">
                  <div className="w-16 h-16 rounded-full bg-primary-container text-white flex items-center justify-center font-bold text-2xl shadow-sm">
                    {user.email.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <h3 className="font-headline-sm text-headline-sm text-primary">
                      {user.email.split('@')[0].replace('.', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </h3>
                    <p className="text-on-surface-variant text-xs">{user.email}</p>
                  </div>
                </div>

                <div className="h-[1px] bg-outline-variant/15 w-full"></div>

                <div className="space-y-4">
                  <h4 className="font-label-caps text-xs font-bold tracking-wider text-outline uppercase">Security & Standards</h4>
                  <div className="grid grid-cols-2 gap-4 text-xs font-semibold text-on-surface-variant">
                    <div className="bg-surface-container-low p-4 rounded-xl border border-outline-variant/10">
                      <p className="text-[10px] text-outline font-bold tracking-wider uppercase mb-1">HIPAA Compliance</p>
                      <p className="text-primary font-bold">ACTIVE & ENCRYPTED</p>
                    </div>
                    <div className="bg-surface-container-low p-4 rounded-xl border border-outline-variant/10">
                      <p className="text-[10px] text-outline font-bold tracking-wider uppercase mb-1">ISO 27001</p>
                      <p className="text-primary font-bold">CERTIFIED</p>
                    </div>
                  </div>
                </div>

                <div className="flex gap-4 pt-4">
                  <button
                    onClick={() => setCalibrationModal(true)}
                    className="flex-1 bg-primary text-on-primary py-3 rounded-xl font-bold hover:opacity-90 active:scale-95 transition-all shadow-md text-xs tracking-wider font-label-caps"
                  >
                    CALIBRATE SENSORS
                  </button>
                  <button
                    onClick={handleLogout}
                    className="flex-1 border border-error text-error hover:bg-error-container/10 py-3 rounded-xl font-bold transition-all text-xs tracking-wider font-label-caps"
                  >
                    LOG OUT
                  </button>
                </div>
              </section>
            )}
          </div>
        </main>
      </div>

      {/* Global Calibration Wizard Modal */}
      {calibrationModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-6 animate-fade-in">
          <div className="relative w-full max-w-lg bg-surface rounded-[32px] overflow-hidden shadow-xl p-8 border border-outline-variant/20">
            <button
              onClick={() => setCalibrationModal(false)}
              className="absolute top-6 right-6 text-on-surface-variant hover:text-primary transition-colors"
            >
              <span className="material-symbols-outlined">close</span>
            </button>
            <CalibrationWizard
              userId="default"
              onComplete={() => setCalibrationModal(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
