# Master Frontend Architecture, Feature & Page Audit Report

This master report combines both the **Component & System Architecture Audit** and the **Page-by-Page Feature Hierarchy Audit** for the entire `webapp/frontend` codebase. It details every feature, component, sub-component, button, input field, toggle, modal, metric, chart, and API endpoint integration without missing a single element.

---

# Part I: Component & System Level Architecture Audit

## 1. Global App Shell (`App.js`)

The main application shell (`App.js`) serves as the central state controller, handling authentication, routing, top app bar title generation, modal visibility, and responsive layout spacing.

### State Variables & Configuration:
- `user`: Persisted user session object stored in `localStorage['vitalmind-user']`.
- `activeView`: Active primary view (`'dashboard'`, `'insights'`, `'recovery'`, `'profile'`).
- `dashboardMode`: Sub-mode within Dashboard (`'realtime'` or `'upload'`).
- `showCopilot`: Boolean flag toggling the slide-out AI Stress Copilot drawer.
- `calibrationModal`: Boolean flag controlling the 3-step baseline calibration modal window.
- `isSidebarOpen`: Desktop sidebar collapse state (`true` = 256px wide, `false` = 80px icon-only).

---

## 2. Navigation & Shell Components

### A. SideNavBar (`SideNavBar.jsx`)
- **Type**: Desktop Fixed Navigation Sidebar (`hidden md:flex`).
- **Interactive Controls & Buttons**:
  - **Sidebar Toggle Button**: Icon button (`chevron_left` / `menu`) to toggle sidebar between expanded (256px) and collapsed (80px) modes with a 300ms transition.
  - **Dashboard Tab Button**: Icon `dashboard`, label "Dashboard".
  - **Insights Tab Button**: Icon `insights`, label "Insights".
  - **Recovery Tab Button**: Icon `spa`, label "Recovery".
  - **Profile Tab Button**: Icon `person`, label "Profile".
  - **Calibrate Sensors CTA Button**: Primary button triggering sensor calibration modal.
  - **User Profile Pill**: Avatar circle with initial letter, user name, and role badge ("Clinical Lead").

### B. BottomNavBar (`BottomNavBar.jsx`)
- **Type**: Mobile Fixed Bottom Navigation Bar (`md:hidden`).
- **Interactive Controls**:
  - **Dashboard Nav Button**: Icon `dashboard`, label "Dashboard".
  - **Insights Nav Button**: Icon `insights`, label "Insights".
  - **Recovery Nav Button**: Icon `spa`, label "Recovery".
  - **Profile Nav Button**: Icon `person`, label "Profile".
  - **Calibrate Sensor Button**: Icon `tune`, label "Calibrate".

### C. TopAppBar (`TopAppBar.jsx`)
- **Type**: Dynamic Header Bar.
- **Interactive Controls & Displays**:
  - **Title Heading**: Dynamic page title ("Dashboard", "Insights Summary", "Recovery & Resilience", "User Profile").
  - **Realtime Mode Button**: Tab switcher for Realtime Stream mode.
  - **Upload Mode Button**: Tab switcher for Batch File Upload mode.
  - **Notifications Icon Button**: (`notifications_active`) with feedback animation.
  - **System Telemetry Icon Button**: (`settings_input_component`) with rotation feedback.
  - **Stress Copilot Toggle Button**: (`smart_toy` / "STRESS COPILOT") Toggles AI Copilot drawer.

### D. Authentication & Security Wall (`LoginConsent.jsx`)
- **Type**: Authentication & HIPAA Consent Security Wall.
- **Interactive Controls**:
  - **Login Form**:
    - Email Address Input: `<input type="email">`.
    - Password Input: `<input type="password">`.
    - Forgot Password Link: `<a href="#forgot">Forgot?</a>`.
    - Access Workspace Submit Button: `<button type="submit">Access Workspace</button>`.
    - Sign Up Toggle Link: `<button onClick={() => setIsSignup(true)}>Create Clinical Account</button>`.
  - **Sign Up & Consent Form**:
    - Full Name Input: `<input type="text">`.
    - Work Email Input: `<input type="email">`.
    - Password Input: `<input type="password">`.
    - HIPAA Consent Checkbox: `<input type="checkbox">` ("I accept the Biometric Security Protocol & HIPAA data encryption terms").
    - Accept Terms & Launch Submit Button: `<button type="submit">Accept Terms & Launch</button>`.
    - Back to Sign In Link: `<button onClick={() => setIsSignup(false)}>Sign In</button>`.

---

## 3. Realtime Stress Monitor (`RealtimeMonitor.jsx`)

The Realtime Monitor delivers real-time multimodal stress detection via live video streaming and audio recording.

### A. Live Streams Section
- **Live Face Camera Stream (`FaceStream.jsx`)**:
  - Video Stream Canvas: MediaPipe Face Mesh landmark tracking.
  - Stream Header: Title "LIVE BIOMETRIC CAMERA & AUDIO STREAM", Live FPS counter (`FPS: 30`).
  - Anti-Spoofing / Liveness Badge: `REAL HUMAN FACE VERIFIED (98.6% LIVENESS)`.
  - Standby Overlay: Standby video icon & text prompt when stream is inactive.
- **Live Audio Waveform Recorder (`WaveformRecorder.jsx`)**:
  - Canvas Visualizer: Live audio waveform frequency spectrum.
  - Mic Telemetry: Live RMS volume readout and mic status badge.

### B. Per-Modality Score & Telemetry Cards
- **Fused Stress Index Card**: Overall fused sympathetic stress score (%) + animated progress bar.
- **Face Score Card**: Individual facial micro-expression stress score (%) + progress bar.
- **Voice Score Card**: Individual vocal acoustic stress score (%) + progress bar.
- **Feature Capturing Rate & Throughput Bar**:
  - Stream Health progress meter (0-100%).
  - `Face Mesh Rate` (FPS counter).
  - `Audio Sample Rate` (kHz counter).

### C. Biometric Reliability & Realness Inspector
- **Face Input Verification Badge**: Confirms genuine face stream liveness (`✓ Real Live Face (98.6%)`).
- **Voice Spectrum Integrity Badge**: Confirms human voice acoustic harmonics (`✓ Human Acoustics (96.4%)`).
- **Sympathetic Stress Reliability Badge**: Confirms empirical stress detection reliability (`High Confidence (94.2%)`).

### D. Session Controls Bar
- **Start Session Button**: Icon `play_arrow`, initiates real-time SSE stream.
- **Stop Session Button**: Icon `stop`, terminates stream.
- **Recalibrate Button**: Opens baseline calibration wizard.
- **Server Status Badge**: (`CONNECTED` / `DISCONNECTED`).
- **Classification Level Display**: (`Calm / Baseline`, `Moderate Stress`, `High Stress`).

### E. Expert Feature Banks
- **Face Expert Features Card**: `Blink Velocity`, `Eye Aspect (EAR)`, `Jaw Displ.`, `Head Tilt (°)`, `Brow Descent`, `Lip Compress`.
- **Voice Expert Features Card**: `Pitch (F0 Hz)`, `Jitter (%)`, `Shimmer (dB)`, `Intensity`, `ZCR Rate`, `Stream Status`.

---

## 4. Multimodal Analysis & Diagnosis Panel (`AnalysisPanel.jsx`)

Displays offline analysis results and model explanations.

### Metrics & Components:
- **Stress Classification Banner**: Large status pill (`LOW STRESS`, `MODERATE STRESS`, `HIGH STRESS`) with fused score percentage.
- **Dynamic Modality Contribution Bars**: `Facial Stream Contribution %`, `Voice Stream Contribution %`, `Physiological Stream Contribution %`.
- **Biometric Radar Chart**: Recharts radar visualization mapping biometric feature dimensions.
- **Top Stress Driver Features List**: Ranked list of top biometric features driving the prediction.
- **Model Orchestration Card**: Displays active ML engine (`SSVB-CASA-AIS`), version, and latency (ms).

---

## 5. Guided Baseline Calibration Wizard (`CalibrationWizard.jsx`)

A 3-step wizard to calibrate personal baseline biometric thresholds.

### Controls & Steps:
- **Step 1 — Silence Baseline (15s)**: Measures environmental room noise level.
- **Step 2 — Voice Baseline (40s)**: User reads a calm passage to calibrate pitch and rhythm.
- **Step 3 — Face Baseline (45s)**: User looks at camera with neutral expression to calibrate eye openness & brow position.
- **Progress Bar & Ring Timer**: Visual ring timer showing phase countdown.
- **Complete & Save Button**: Stores calibrated baseline parameters to backend.
- **Reset Baseline Button**: Clears personal baseline back to factory defaults.

---

## 6. Offline File Upload & Batch Processing (`Dashboard.js` Upload Mode)

Allows uploading recorded media or physiological CSV files.

### Form Inputs & Buttons:
- **Facial Image Upload Card**:
  - Drag-and-drop image file input (`.jpg`, `.png`).
  - `Use Live WebCam` capture button.
  - `Take Snapshot` button.
- **Voice Audio Upload Card**:
  - Drag-and-drop audio file input (`.wav`, `.mp3`).
  - `Record Audio` button with built-in mic recorder.
  - `Stop Recording` button.
- **EEG Signal CSV Input**: File uploader (`.csv`), text area input, and Recharts line graph preview.
- **GSR Signal CSV Input**: File uploader (`.csv`), text area input, and Recharts line graph preview.
- **Submit Batch Analysis Button**: Triggers `/api/predict` endpoint.

---

## 7. Personal Insights & Longitudinal Baseline Analytics (`PersonalInsights.jsx`)

Longitudinal analytics tracking weekly stress resilience and circadian trends.

### Visualizations:
- **Weekly Stress Resilience Area Chart**: Recharts AreaChart graphing weekly sympathetic tone (Mon-Sun).
- **Circadian Stress Load Chart**: Line graph showing stress load across hours of the day.
- **Sleep vs. Stress Scatter Plot**: Correlation plot comparing hours of sleep vs. stress index.
- **7-Day Baseline Summary Cards**: `Average Stress Score`, `Peak Stress Hour`, `Recovery Score`.

---

## 8. Guided Recovery & Resilience Activities (`RecoveryActivities.jsx` & `GamePanel.jsx`)

Gamified micro-interventions to reduce stress.

### Games & Features:
- **4-7-8 Guided Breathing Game (`GamePanel.jsx`)**: Animated SVG expanding ring timer guiding inhale (4s), hold (7s), and exhale (8s) with target cycle counter.
- **Cognitive Reframing Challenge**: Interactive prompt guiding positive cognitive restructuring.
- **Progressive Muscle Relaxation (PMR) Guide**: Step-by-step muscle relaxation guide.
- **Gamified Reward System (`RewardSystem.jsx`)**: Streak counter, badges, and resilience points.

---

## 9. AI Stress Copilot Assistant (`StressChatbot.jsx` & `CopilotMessage.jsx`)

Real-time clinical AI assistant providing evidence-based coping strategies.

### Interactive Controls:
- **Chat Drawer Toggle**: Opens AI Copilot sidebar drawer.
- **Message History List**: Chat messages with formatted copilot recommendations.
- **Quick Action Chips**: Prompt shortcuts.
- **Message Input Box & Send Button**: Text area with submit action.

---

## 10. Complete API Endpoint Integration Index

| Endpoint | Method | Component | Purpose |
| :--- | :--- | :--- | :--- |
| `/api/health` | GET | `Dashboard.js`, `RealtimeMonitor.jsx` | Health check & server monitoring |
| `/api/stream/start` | POST | `RealtimeMonitor.jsx` | Initiates real-time multimodal SSE stream |
| `/api/stream/stop` | POST | `RealtimeMonitor.jsx` | Stops SSE stream |
| `/api/stream/fused` | GET (SSE) | `RealtimeMonitor.jsx` | Real-time Server-Sent Events stream for fused score |
| `/api/analyze/voice` | POST | `RealtimeMonitor.jsx`, `WaveformRecorder.jsx` | Uploads voice chunk for acoustic analysis |
| `/api/calibrate/status` | GET | `RealtimeMonitor.jsx`, `CalibrationWizard.jsx` | Fetches personal calibration status |
| `/api/calibrate/reset` | POST | `RealtimeMonitor.jsx` | Resets calibration baseline |
| `/api/model/version` | GET | `RealtimeMonitor.jsx` | Fetches active ML model metadata |
| `/api/fallback/status` | GET | `RealtimeMonitor.jsx` | Checks ML resilience fallback status |
| `/api/predict` | POST | `Dashboard.js` | Submits batch multimodal inputs for analysis |

---

# Part II: Page-by-Page Component & Feature Hierarchy

## Page 1: Authentication & HIPAA Consent Security Wall
- **Route / State**: `!user`
- **Main Component**: `LoginConsent.jsx`
- **Child Elements**: Visual Quote Hero Panel, Error Alert Bar, Login Form (Email, Password, Access Button, Signup Link), Signup Form (Name, Email, Password, HIPAA Checkbox, Accept Button, Sign In Link).

## Page 2: Realtime Stress Monitoring Dashboard Page
- **Route / State**: `user !== null && activeView === 'dashboard' && dashboardMode === 'realtime'`
- **Main Components**: `App.js` → `Dashboard.js` → `RealtimeMonitor.jsx`
- **Child Components**: `SideNavBar.jsx`, `BottomNavBar.jsx`, `TopAppBar.jsx`, `FaceStream.jsx`, `WaveformRecorder.jsx`, `StressChatbot.jsx`, `CalibrationWizard.jsx` (Modal).
- **Interactive Controls & Displays**: Live webcam feed, MediaPipe landmarks, FPS counter, Anti-Spoofing liveness overlay, audio waveform spectrum, Fused Score card, Face Score card, Voice Score card, Feature Capturing Rate bar, Biometric Reliability inspector, Session Start/Stop controls, Face Expert Features (6 metrics), Voice Expert Features (6 metrics).

## Page 3: Batch Offline Upload Analysis Page
- **Route / State**: `user !== null && activeView === 'dashboard' && dashboardMode === 'upload'`
- **Main Components**: `App.js` → `Dashboard.js` → `AnalysisPanel.jsx`
- **Child Components**: `SideNavBar.jsx`, `BottomNavBar.jsx`, `TopAppBar.jsx`, `AnalysisPanel.jsx`, `StressChatbot.jsx`.
- **Interactive Controls & Displays**: Facial image uploader/webcam snapshot, voice audio uploader/mic recorder, EEG CSV uploader & Recharts preview, GSR CSV uploader & Recharts preview, Run Analysis button, Stress Classification Banner, Modality Weight bars, Recharts Radar Chart, Ranked Stress Drivers.

## Page 4: Personal Insights & Longitudinal Baseline Page
- **Route / State**: `user !== null && activeView === 'insights'`
- **Main Components**: `App.js` → `PersonalInsights.jsx` & `InsightCards.jsx`
- **Child Components**: `SideNavBar.jsx`, `BottomNavBar.jsx`, `TopAppBar.jsx`, `PersonalInsights.jsx`.
- **Interactive Controls & Displays**: Weekly Stress Resilience Area Chart, Circadian Stress Load Line Chart, Sleep vs. Stress Scatter Plot, 7-day metric summary cards.

## Page 5: Recovery, Resilience & Guided Micro-Interventions Page
- **Route / State**: `user !== null && activeView === 'recovery'`
- **Main Components**: `App.js` → `RecoveryActivities.jsx`, `GamePanel.jsx`, `BreathingExercise.jsx`, `RewardSystem.jsx`
- **Child Components**: `SideNavBar.jsx`, `BottomNavBar.jsx`, `TopAppBar.jsx`, `RecoveryActivities.jsx`, `GamePanel.jsx`, `RewardSystem.jsx`.
- **Interactive Controls & Displays**: 4-7-8 Guided Breathing ring timer game, Cognitive Reframing Challenge, Progressive Muscle Relaxation guide, Gamified streak badges and resilience points counter.

## Page 6: User Profile & Security Settings Page
- **Route / State**: `user !== null && activeView === 'profile'`
- **Main Component**: `App.js` (Profile Section)
- **Child Components**: `SideNavBar.jsx`, `BottomNavBar.jsx`, `TopAppBar.jsx`.
- **Interactive Controls & Displays**: User avatar circle, Email readout, HIPAA & ISO 27001 compliance cards, Calibrate Sensors CTA button, Log Out button.

---

# Part III: Exhaustive Itemized Element Checklist

### All Interactive Buttons:
1. `Sidebar Toggle Button` (`SideNavBar.jsx`)
2. `Dashboard Tab Button` (`SideNavBar.jsx`, `BottomNavBar.jsx`)
3. `Insights Tab Button` (`SideNavBar.jsx`, `BottomNavBar.jsx`)
4. `Recovery Tab Button` (`SideNavBar.jsx`, `BottomNavBar.jsx`)
5. `Profile Tab Button` (`SideNavBar.jsx`, `BottomNavBar.jsx`)
6. `Calibrate Sensors CTA Button` (`SideNavBar.jsx`, `BottomNavBar.jsx`, `App.js`, `RealtimeMonitor.jsx`)
7. `Realtime Mode Switch Tab` (`TopAppBar.jsx`)
8. `Upload Mode Switch Tab` (`TopAppBar.jsx`)
9. `Notifications Icon Button` (`TopAppBar.jsx`)
10. `System Telemetry Icon Button` (`TopAppBar.jsx`)
11. `Stress Copilot Toggle Button` (`TopAppBar.jsx`)
12. `Access Workspace Login Submit Button` (`LoginConsent.jsx`)
13. `Create Clinical Account Link` (`LoginConsent.jsx`)
14. `Accept Terms & Launch Signup Submit Button` (`LoginConsent.jsx`)
15. `Back to Sign In Link` (`LoginConsent.jsx`)
16. `Start Session Button` (`RealtimeMonitor.jsx`)
17. `Stop Session Button` (`RealtimeMonitor.jsx`)
18. `Recalibrate Baseline Button` (`RealtimeMonitor.jsx`, `CalibrationWizard.jsx`)
19. `Reset Baseline Button` (`RealtimeMonitor.jsx`, `CalibrationWizard.jsx`)
20. `Use Live WebCam Button` (`Dashboard.js`)
21. `Take Snapshot Button` (`Dashboard.js`)
22. `Record Audio Button` (`Dashboard.js`)
23. `Stop Recording Button` (`Dashboard.js`)
24. `Run Multimodal Analysis Button` (`Dashboard.js`)
25. `Start 4-7-8 Breathing Game Button` (`GamePanel.jsx`)
26. `Pause Breathing Exercise Button` (`GamePanel.jsx`)
27. `Generate Clinical Reframe Button` (`RecoveryActivities.jsx`)
28. `Copilot Quick Action Chip 1` (`StressChatbot.jsx`)
29. `Copilot Quick Action Chip 2` (`StressChatbot.jsx`)
30. `Copilot Quick Action Chip 3` (`StressChatbot.jsx`)
31. `Copilot Send Message Button` (`StressChatbot.jsx`)
32. `Log Out Button` (`App.js`)

### All Form Input Fields & Controls:
1. `Email Input` (`LoginConsent.jsx`)
2. `Password Input` (`LoginConsent.jsx`)
3. `Full Name Input` (`LoginConsent.jsx`)
4. `HIPAA Consent Checkbox` (`LoginConsent.jsx`)
5. `Facial Image Drag-and-Drop Uploader` (`Dashboard.js`)
6. `Voice Audio Drag-and-Drop Uploader` (`Dashboard.js`)
7. `EEG CSV File Uploader` (`Dashboard.js`)
8. `EEG Raw Text Area` (`Dashboard.js`)
9. `GSR CSV File Uploader` (`Dashboard.js`)
10. `GSR Raw Text Area` (`Dashboard.js`)
11. `Cognitive Reframing Trigger Text Area` (`RecoveryActivities.jsx`)
12. `Copilot Chat Input Text Area` (`StressChatbot.jsx`)

### All Data Charts & Visualizations:
1. `Live Audio Waveform Frequency Spectrum Canvas` (`WaveformRecorder.jsx`)
2. `Live MediaPipe Face Mesh Video Canvas` (`FaceStream.jsx`)
3. `Biometric Feature Dimension Radar Chart` (`AnalysisPanel.jsx`)
4. `EEG Multi-Channel Signal Line Graph` (`Dashboard.js`)
5. `GSR Skin Conductance Line Graph` (`Dashboard.js`)
6. `Weekly Stress Resilience Area Chart` (`PersonalInsights.jsx`)
7. `Circadian Stress Load Line Chart` (`PersonalInsights.jsx`)
8. `Sleep vs Stress Correlation Scatter Plot` (`PersonalInsights.jsx`)
9. `4-7-8 Animated Guided Breathing SVG Ring` (`GamePanel.jsx`)
