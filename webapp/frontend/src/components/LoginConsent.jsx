import React, { useState } from 'react';

export default function LoginConsent({ onLogin }) {
  const [isSignup, setIsSignup] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [consentChecked, setConsentChecked] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleLogin = (e) => {
    e.preventDefault();
    if (!email || !password) {
      setErrorMessage('Please fill in all fields.');
      return;
    }
    // Perform simulated login
    onLogin({ email });
  };

  const handleSignup = (e) => {
    e.preventDefault();
    if (!email || !password) {
      setErrorMessage('Please fill in all fields.');
      return;
    }
    if (!consentChecked) {
      setErrorMessage('You must accept the Biometric Security Protocol consent.');
      return;
    }
    // Perform simulated sign up
    onLogin({ email });
  };

  return (
    <div className="bg-warm-sand text-on-surface font-body-md min-h-screen flex items-center justify-center p-6 w-full">
      <main className="w-full max-w-[1100px] grid md:grid-cols-2 bg-surface-container-lowest rounded-[32px] overflow-hidden shadow-sm min-h-[700px] transition-all duration-300">
        
        {/* Left Visual Panel */}
        <section className="relative hidden md:flex flex-col justify-between p-12 overflow-hidden bg-primary text-white">
          <div className="relative z-10">
            <h1 className="font-headline-md text-[32px] font-bold text-surface-bright mb-2">VitalMind</h1>
            <p className="text-surface-variant/80 max-w-xs font-body-md">Empirical clarity for the modern professional mind.</p>
          </div>
          
          <div className="relative z-10 bg-white/5 backdrop-blur-lg border border-white/10 p-8 rounded-2xl">
            <blockquote className="text-surface-bright italic font-headline-sm mb-6 leading-relaxed">
              "The quiet mind is the most powerful tool in the clinical arsenal."
            </blockquote>
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-surface-container-high/20 flex items-center justify-center border border-white/20">
                <span className="material-symbols-outlined text-surface-bright">monitoring</span>
              </div>
              <div>
                <p className="text-surface-bright font-semibold">Dr. Elena Vance</p>
                <p className="text-surface-variant text-sm">Chief of Neuroscience</p>
              </div>
            </div>
          </div>
          <div className="absolute -bottom-24 -right-24 w-64 h-64 bg-surface-variant/10 rounded-full blur-3xl"></div>
        </section>

        {/* Right Interaction Panel */}
        <section className="p-8 md:p-16 flex flex-col justify-center bg-surface-container-lowest">
          <div className="w-full">
            {errorMessage && (
              <div className="mb-4 p-3 bg-error-container text-on-error-container text-xs rounded-xl flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">error</span>
                <span>{errorMessage}</span>
              </div>
            )}

            {!isSignup ? (
              /* Login Form */
              <form onSubmit={handleLogin} className="space-y-8">
                <div className="space-y-2">
                  <h2 className="font-headline-sm text-headline-sm text-primary">Welcome Back</h2>
                  <p className="text-on-surface-variant">Enter your credentials to access your vitals.</p>
                </div>

                <div className="space-y-4">
                  <div className="space-y-1">
                    <label className="font-label-caps text-[12px] text-outline px-1 focus-within:text-primary block">Email Address</label>
                    <input
                      className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all placeholder:text-outline-variant text-on-surface"
                      placeholder="name@organization.com"
                      type="email"
                      value={email}
                      onChange={(e) => { setEmail(e.target.value); setErrorMessage(''); }}
                      required
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between items-center px-1">
                      <label className="font-label-caps text-[12px] text-outline block">Password</label>
                      <a className="text-primary font-label-caps text-[11px] hover:underline" href="#forgot" onClick={(e) => e.preventDefault()}>Forgot?</a>
                    </div>
                    <input
                      className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all placeholder:text-outline-variant text-on-surface"
                      placeholder="••••••••"
                      type="password"
                      value={password}
                      onChange={(e) => { setPassword(e.target.value); setErrorMessage(''); }}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <button
                    type="submit"
                    className="w-full bg-primary text-on-primary py-4 rounded-xl font-semibold shadow-sm hover:opacity-90 active:scale-[0.98] transition-all flex justify-center items-center gap-2"
                  >
                    Sign In
                    <span className="material-symbols-outlined text-sm">arrow_forward</span>
                  </button>
                  <div className="relative flex items-center gap-4 py-2">
                    <div className="flex-grow border-t border-outline-variant/30"></div>
                    <span className="font-label-caps text-[12px] text-outline-variant">or</span>
                    <div className="flex-grow border-t border-outline-variant/30"></div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onLogin({ email: 'sso-user@hospital.com' })}
                    className="w-full bg-white border border-outline-variant text-on-surface-variant py-3.5 rounded-xl font-medium hover:bg-surface-container-low transition-colors flex items-center justify-center gap-3"
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24">
                      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"></path>
                      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"></path>
                      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"></path>
                      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"></path>
                    </svg>
                    Continue with SSO
                  </button>
                </div>

                <p className="text-center text-on-surface-variant text-sm">
                  New to VitalMind?{' '}
                  <button type="button" className="text-primary font-semibold hover:underline" onClick={() => { setIsSignup(true); setErrorMessage(''); }}>Create Account</button>
                </p>
              </form>
            ) : (
              /* Signup & Consent Form */
              <form onSubmit={handleSignup} className="space-y-6">
                <div className="space-y-2">
                  <h2 className="font-headline-sm text-headline-sm text-primary">Elevate Your Care</h2>
                  <p className="text-on-surface-variant">Step 1: Clinical Data Ethics & Consent</p>
                </div>

                {/* Consent Card */}
                <div className="bg-surface-container-low p-6 rounded-2xl border border-primary/10 space-y-4">
                  <div className="flex items-start gap-4">
                    <div className="bg-primary-container p-2 rounded-lg">
                      <span className="material-symbols-outlined text-on-primary-container">security</span>
                    </div>
                    <div className="space-y-1">
                      <h3 className="font-semibold text-primary">Biometric Security Protocol</h3>
                      <p className="text-xs text-on-surface-variant leading-relaxed">
                        VitalMind uses real-time camera and microphone telemetry to analyze physiological stress markers.
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white/50 p-3 rounded-xl flex items-center gap-2 border border-outline-variant/20">
                      <span className="material-symbols-outlined text-primary text-lg">videocam</span>
                      <span className="text-[11px] font-label-caps text-on-surface-variant">Micro-Expressions</span>
                    </div>
                    <div className="bg-white/50 p-3 rounded-xl flex items-center gap-2 border border-outline-variant/20">
                      <span className="material-symbols-outlined text-primary text-lg">mic</span>
                      <span className="text-[11px] font-label-caps text-on-surface-variant">Vocal Tonality</span>
                    </div>
                    <div className="bg-white/50 p-3 rounded-xl flex items-center gap-2 border border-outline-variant/20">
                      <span className="material-symbols-outlined text-primary text-lg">lock</span>
                      <span className="text-[11px] font-label-caps text-on-surface-variant">E2E Encrypted</span>
                    </div>
                    <div className="bg-white/50 p-3 rounded-xl flex items-center gap-2 border border-outline-variant/20">
                      <span className="material-symbols-outlined text-primary text-lg">cloud_off</span>
                      <span className="text-[11px] font-label-caps text-on-surface-variant">Local Processing</span>
                    </div>
                  </div>

                  <div className="flex items-start gap-3 pt-2">
                    <input
                      className="mt-1 rounded text-primary focus:ring-primary h-4 w-4 accent-primary"
                      id="consent"
                      type="checkbox"
                      checked={consentChecked}
                      onChange={(e) => { setConsentChecked(e.target.checked); setErrorMessage(''); }}
                    />
                    <label className="text-xs text-on-surface-variant leading-tight" htmlFor="consent">
                      I understand that my biometric data is strictly used for clinical analysis and will never be shared with third parties or advertisers.
                    </label>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="space-y-1">
                    <label className="font-label-caps text-[12px] text-outline px-1 block">Institutional Email</label>
                    <input
                      className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all text-on-surface"
                      placeholder="name@hospital.com"
                      type="email"
                      value={email}
                      onChange={(e) => { setEmail(e.target.value); setErrorMessage(''); }}
                      required
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="font-label-caps text-[12px] text-outline px-1 block">Create Password</label>
                    <input
                      className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all text-on-surface"
                      placeholder="Minimum 12 characters"
                      type="password"
                      value={password}
                      onChange={(e) => { setPassword(e.target.value); setErrorMessage(''); }}
                      required
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full bg-primary text-on-primary py-4 rounded-xl font-semibold shadow-sm hover:opacity-90 active:scale-[0.98] transition-all"
                >
                  Complete Registration
                </button>
                <p className="text-center text-on-surface-variant text-sm">
                  Already have an account?{' '}
                  <button type="button" className="text-primary font-semibold hover:underline" onClick={() => { setIsSignup(false); setErrorMessage(''); }}>Log In</button>
                </p>
              </form>
            )}
          </div>

          <footer className="mt-12 pt-6 border-t border-outline-variant/10 text-center">
            <p className="text-[10px] text-outline uppercase tracking-widest mb-2 font-label-caps">Clinical Standard ISO-27001 Certified</p>
            <div className="flex justify-center gap-4 text-[11px] text-on-surface-variant">
              <a className="hover:text-primary transition-colors" href="#privacy" onClick={(e) => e.preventDefault()}>Privacy Policy</a>
              <span className="text-outline-variant">•</span>
              <a className="hover:text-primary transition-colors" href="#terms" onClick={(e) => e.preventDefault()}>Terms of Use</a>
              <span className="text-outline-variant">•</span>
              <a className="hover:text-primary transition-colors" href="#support" onClick={(e) => e.preventDefault()}>Support</a>
            </div>
          </footer>
        </section>

      </main>
    </div>
  );
}
