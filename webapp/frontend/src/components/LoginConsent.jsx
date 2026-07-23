import React, { useState } from 'react';

export default function LoginConsent({ onLogin }) {
  const [isSignup, setIsSignup] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(true);
  const [consentChecked, setConsentChecked] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [showConsentModal, setShowConsentModal] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!email || !password) {
      setErrorMessage('Please fill in all required credentials.');
      return;
    }
    if (isSignup && !consentChecked) {
      setErrorMessage('Biometric Security Protocol Consent is required for account creation.');
      return;
    }
    onLogin({ email });
  };

  const handleBiometricLogin = (type) => {
    onLogin({ email: type === 'face' ? 'alexander.faceid@vitalmind.ai' : 'alexander.touchid@vitalmind.ai' });
  };

  return (
    <div className="bg-surface/90 min-h-screen w-full flex flex-col items-center justify-center p-6 font-body-md text-on-surface relative overflow-hidden select-none">
      {/* Background Mesh Gradient */}
      <div className="fixed inset-0 -z-10 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] bg-primary/10 blur-[140px] rounded-full"></div>
        <div className="absolute top-[50%] -right-[10%] w-[40%] h-[40%] bg-tertiary/10 blur-[120px] rounded-full"></div>
      </div>

      {/* Main Container */}
      <main className="w-full max-w-[480px] z-10 animate-fade-in-up">
        {/* Brand & Header Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center space-x-3 mb-3">
            <span className="material-symbols-outlined text-[42px] text-primary">
              psychology
            </span>
            <h1 className="font-display-hero text-3xl md:text-4xl font-bold text-primary tracking-tight">
              VitalMind Pro
            </h1>
          </div>
          <p className="font-body-lg text-sm md:text-base text-on-surface-variant">
            {isSignup ? 'Create your clinical account sanctuary.' : 'Welcome back to your clinical sanctuary.'}
          </p>
        </div>

        {/* Login / Signup Glass Card */}
        <div className="glass-card rounded-3xl p-8 shadow-xl">
          {errorMessage && (
            <div className="mb-6 p-4 bg-error-container/80 text-on-error-container text-xs rounded-2xl flex items-center gap-2 border border-error/20">
              <span className="material-symbols-outlined text-base">error</span>
              <span>{errorMessage}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Field */}
            <div className="space-y-1">
              <label className="font-label-caps text-xs text-outline font-semibold uppercase px-1">Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => { setEmail(e.target.value); setErrorMessage(''); }}
                placeholder="alexander@vitalmind.ai"
                className="w-full px-4 py-3.5 rounded-2xl border border-outline-variant/40 focus:border-primary focus:ring-2 focus:ring-primary/20 bg-surface-container-lowest/80 text-on-surface transition-all outline-none text-sm font-medium"
              />
            </div>

            {/* Password Field */}
            <div className="space-y-1">
              <label className="font-label-caps text-xs text-outline font-semibold uppercase px-1">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => { setPassword(e.target.value); setErrorMessage(''); }}
                placeholder="••••••••••••"
                className="w-full px-4 py-3.5 rounded-2xl border border-outline-variant/40 focus:border-primary focus:ring-2 focus:ring-primary/20 bg-surface-container-lowest/80 text-on-surface transition-all outline-none text-sm font-medium"
              />
            </div>

            {/* Utilities Row */}
            <div className="flex items-center justify-between text-xs">
              <label className="flex items-center space-x-2 cursor-pointer group select-none">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-4 h-4 rounded text-primary focus:ring-primary border-outline-variant"
                />
                <span className="font-body-md text-on-surface-variant group-hover:text-primary transition-colors">Remember me</span>
              </label>
              <button
                type="button"
                onClick={() => setShowConsentModal(true)}
                className="font-body-md text-primary font-semibold hover:underline"
              >
                Security Protocols
              </button>
            </div>

            {/* Signup Consent Checkbox */}
            {isSignup && (
              <div className="p-4 bg-surface-container-low rounded-2xl border border-primary/10 text-xs space-y-2">
                <label className="flex items-start gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={consentChecked}
                    onChange={(e) => setConsentChecked(e.target.checked)}
                    className="mt-0.5 w-4 h-4 rounded text-primary focus:ring-primary border-outline-variant"
                  />
                  <span className="text-on-surface-variant leading-relaxed">
                    I consent to real-time 468-point facial landmark processing, acoustic feature extraction, and encrypted biometric telemetry analysis.
                  </span>
                </label>
              </div>
            )}

            {/* Primary Action Button */}
            <button
              type="submit"
              className="w-full bg-primary text-on-primary font-display-hero text-sm py-4 rounded-2xl shadow-lg shadow-primary/20 hover:bg-primary-container transition-all flex items-center justify-center space-x-2 font-bold"
            >
              <span>{isSignup ? 'Create Account' : 'Sign In'}</span>
              <span className="material-symbols-outlined text-lg">arrow_forward</span>
            </button>

            {/* Divider */}
            <div className="relative flex items-center py-2">
              <div className="flex-grow border-t border-outline-variant/30"></div>
              <span className="flex-shrink mx-4 text-on-surface-variant/70 font-label-caps text-[10px] font-bold uppercase tracking-wider">
                OR CONTINUE WITH
              </span>
              <div className="flex-grow border-t border-outline-variant/30"></div>
            </div>

            {/* Biometric & Fast Access Buttons */}
            <div className="grid grid-cols-2 gap-4">
              <button
                type="button"
                onClick={() => handleBiometricLogin('face')}
                className="flex items-center justify-center space-x-2 p-3.5 border border-outline-variant/40 rounded-2xl hover:bg-primary/5 transition-all active:scale-95 text-xs font-semibold text-on-surface-variant"
              >
                <span className="material-symbols-outlined text-primary text-xl">face</span>
                <span>Face ID</span>
              </button>
              <button
                type="button"
                onClick={() => handleBiometricLogin('touch')}
                className="flex items-center justify-center space-x-2 p-3.5 border border-outline-variant/40 rounded-2xl hover:bg-primary/5 transition-all active:scale-95 text-xs font-semibold text-on-surface-variant"
              >
                <span className="material-symbols-outlined text-primary text-xl">fingerprint</span>
                <span>Touch ID</span>
              </button>
            </div>
          </form>
        </div>

        {/* Toggle Account Mode Footer */}
        <p className="text-center mt-6 font-body-md text-xs text-on-surface-variant">
          {isSignup ? 'Already have an executive account?' : "Don't have a professional account?"}
          <button
            onClick={() => { setIsSignup(!isSignup); setErrorMessage(''); }}
            className="text-primary font-bold hover:underline ml-1"
          >
            {isSignup ? 'Sign In' : 'Create an Account'}
          </button>
        </p>
      </main>

      {/* Security Consent Protocol Modal */}
      {showConsentModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-6 animate-fade-in">
          <div className="relative w-full max-w-md bg-surface rounded-3xl overflow-hidden shadow-2xl p-8 border border-primary/20 space-y-4">
            <div className="flex items-center justify-between border-b border-primary/10 pb-3">
              <h3 className="font-display-hero text-lg font-bold text-primary">Biometric Security Protocols</h3>
              <button onClick={() => setShowConsentModal(false)} className="text-on-surface-variant hover:text-primary">
                <span className="material-symbols-outlined text-xl">close</span>
              </button>
            </div>
            <div className="space-y-3 text-xs text-on-surface-variant leading-relaxed">
              <p>
                <strong>HIPAA & ISO 27001 Certified:</strong> All video frames, facial landmark vectors, and acoustic recordings are processed locally or via encrypted 256-bit TLS streams.
              </p>
              <p>
                No raw imagery is stored permanently on remote servers without explicit user approval.
              </p>
            </div>
            <button
              onClick={() => setShowConsentModal(false)}
              className="w-full py-3 bg-primary text-on-primary rounded-xl font-label-caps text-xs font-bold shadow-md hover:bg-primary/90 transition-all"
            >
              Acknowledge & Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
