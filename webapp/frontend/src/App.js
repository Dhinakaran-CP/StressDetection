import React, { useState, useEffect } from 'react';
import LoginConsent from './components/LoginConsent';
import SideNavBar from './components/SideNavBar';
import BottomNavBar from './components/BottomNavBar';
import TopAppBar from './components/TopAppBar';
import VitalMindDashboard from './pages/VitalMindDashboard';
import RealtimeMonitorStitch from './components/RealtimeMonitorStitch';
import AIInsightsHub from './pages/AIInsightsHub';
import MultimodalAnalysisCenter from './pages/MultimodalAnalysisCenter';
import RecoveryCenterStitch from './pages/RecoveryCenterStitch';
import UploadCenterStitch from './pages/UploadCenterStitch';
import UserProfileSettings from './pages/UserProfileSettings';
import StressChatbot from './components/StressChatbot';
import CalibrationWizard from './components/CalibrationWizard';

function App() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('vitalmind-user');
    return saved ? JSON.parse(saved) : null;
  });

  const [activeView, setActiveView] = useState('dashboard');
  const [dashboardMode, setDashboardMode] = useState('realtime');
  const [showCopilot, setShowCopilot] = useState(false);
  const [calibrationModal, setCalibrationModal] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

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

  if (!user) {
    return <LoginConsent onLogin={handleLogin} />;
  }

  let viewTitle = 'VitalMind Pro';
  if (activeView === 'dashboard') viewTitle = 'Executive Health Dashboard';
  if (activeView === 'realtime') viewTitle = 'Realtime Biometric Monitor';
  if (activeView === 'insights') viewTitle = 'AI Insights & Analytics';
  if (activeView === 'multimodal') viewTitle = 'Multimodal Analysis Center';
  if (activeView === 'recovery') viewTitle = 'Recovery & Resilience';
  if (activeView === 'upload') viewTitle = 'Bio-Data Upload Center';
  if (activeView === 'profile') viewTitle = 'Executive User Profile';

  return (
    <div className="bg-background text-on-background font-body-md min-h-screen flex text-sm transition-all duration-300">
      {/* Sidebar Navigation */}
      <SideNavBar
        activeView={activeView}
        setActiveView={setActiveView}
        onCalibrateTrigger={() => setCalibrationModal(true)}
        user={user}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
      />

      {/* Bottom Mobile Navigation */}
      <BottomNavBar
        activeView={activeView}
        setActiveView={setActiveView}
        onCalibrateTrigger={() => setCalibrationModal(true)}
      />

      {/* Main Content Area */}
      <div
        className={`flex-1 flex flex-col transition-all duration-300 ${
          isSidebarOpen ? 'pl-0 lg:pl-64' : 'pl-0 lg:pl-20'
        } pb-16 lg:pb-0`}
      >
        <TopAppBar
          title={viewTitle}
          activeView={activeView}
          dashboardMode={dashboardMode}
          setDashboardMode={setDashboardMode}
          showCopilot={showCopilot}
          setShowCopilot={setShowCopilot}
          isSidebarOpen={isSidebarOpen}
          setIsSidebarOpen={setIsSidebarOpen}
        />

        <main className="flex-1 mt-16 p-4 md:p-8 lg:p-10 bg-transparent overflow-x-hidden relative">
          <div className="relative z-10 w-full mx-auto">
            {activeView === 'dashboard' && (
              <VitalMindDashboard onNavigate={setActiveView} />
            )}

            {activeView === 'realtime' && (
              <RealtimeMonitorStitch variant="collapsible" onNavigate={setActiveView} />
            )}

            {activeView === 'insights' && (
              <AIInsightsHub onNavigate={setActiveView} />
            )}

            {activeView === 'multimodal' && (
              <MultimodalAnalysisCenter onNavigate={setActiveView} />
            )}

            {activeView === 'recovery' && (
              <RecoveryCenterStitch onNavigate={setActiveView} />
            )}

            {activeView === 'upload' && (
              <UploadCenterStitch onNavigate={setActiveView} />
            )}

            {(activeView === 'profile' || activeView === 'settings') && (
              <UserProfileSettings user={user} onLogout={handleLogout} onCalibrateTrigger={() => setCalibrationModal(true)} />
            )}
          </div>
        </main>
      </div>

      {/* Copilot AI Drawer / Modal */}
      {showCopilot && (
        <div className="fixed bottom-6 right-6 z-50 w-full max-w-md animate-fade-in-up">
          <div className="relative bg-surface rounded-3xl shadow-2xl border border-primary/20 overflow-hidden p-4">
            <button
              onClick={() => setShowCopilot(false)}
              className="absolute top-4 right-4 text-on-surface-variant hover:text-primary transition-colors z-10"
            >
              <span className="material-symbols-outlined text-xl">close</span>
            </button>
            <StressChatbot />
          </div>
        </div>
      )}

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
            <CalibrationWizard userId="default" onComplete={() => setCalibrationModal(false)} />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

