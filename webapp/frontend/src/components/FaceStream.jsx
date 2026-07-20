import React, { useEffect, useRef, useState, useCallback } from 'react';
import { API_BASE } from '../config';
// Compute 18 stress indicators from MediaPipe landmarks in browser
function computeStressIndicators(landmarks, imageWidth, imageHeight, history) {
  const pt = (i) => ({
    x: landmarks[i].x * imageWidth,
    y: landmarks[i].y * imageHeight,
  });
  const dist = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

  const faceH = dist(pt(10), pt(152)) + 1e-6;
  const iod   = dist(pt(33), pt(263)) + 1e-6; // inter-ocular distance

  // EAR (Eye Aspect Ratio)
  const earL = (dist(pt(159), pt(145)) + dist(pt(158), pt(153))) / (2 * dist(pt(33), pt(133)) + 1e-6);
  const earR = (dist(pt(386), pt(374)) + dist(pt(385), pt(380))) / (2 * dist(pt(362), pt(263)) + 1e-6);
  const avgEAR = (earL + earR) / 2;

  // Blink velocity from history (frames differences)
  const blinkVelocity = history.length >= 3
    ? Math.abs(avgEAR - history[history.length - 1].ear) / 0.033  // per second proxy at 30fps
    : 0;

  // Brow descent (normalized by face height)
  const browDescL = dist(pt(55), pt(159)) / faceH;
  const browDescR = dist(pt(285), pt(386)) / faceH;
  const browAsym  = Math.abs(browDescL - browDescR);

  // Lip compression
  const lipGap   = dist(pt(13), pt(14));
  const lipWidth = dist(pt(61), pt(291)) + 1e-6;
  const lipCompression = lipGap / lipWidth;

  // Jaw displacement normalized by stable IOD (inter-ocular distance)
  const stableIOD = dist(pt(33), pt(263)) + 1e-6;
  const noseToChinn = dist(pt(4), pt(152));
  const jawDisplacement = noseToChinn / stableIOD;

  // Masseter tension proxy (jaw angle width normalized by stable IOD)
  const jawAngleWidth = dist(pt(172), pt(397)) / stableIOD;

  // Mouth corner pull
  const noseTip = pt(4);
  const mcPull  = (dist(pt(61), noseTip) + dist(pt(291), noseTip)) / (2 * faceH);

  // Forehead tension
  const foreheadTension = dist(pt(10), pt(151)) / faceH;

  // Head tilt
  const eyeL = pt(33); const eyeR = pt(263);
  const headTilt = Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x) * (180 / Math.PI);

  // Temporal variance (nose tip movement)
  const nosePt = pt(4);
  const xVar = history.length >= 5
    ? history.slice(-5).reduce((acc, h) => acc + (h.nx - nosePt.x / imageWidth) ** 2, 0) / 5
    : 0;
  const yVar = history.length >= 5
    ? history.slice(-5).reduce((acc, h) => acc + (h.ny - nosePt.y / imageHeight) ** 2, 0) / 5
    : 0;

  const detection_confidence = landmarks[0]?.visibility ?? 0.9;

  // Smile/laughter detector based on mouth corner elevation and cheek raise compression
  const lipCenterY = (pt(13).y + pt(14).y) / 2;
  const leftCornerY = pt(61).y;
  const rightCornerY = pt(291).y;
  const cornerElevation = ((lipCenterY - leftCornerY) + (lipCenterY - rightCornerY)) / (2 * faceH);
  
  const leftUnderEyeH = dist(pt(159), pt(145));
  const rightUnderEyeH = dist(pt(386), pt(374));
  const cheekRaise = (leftUnderEyeH + rightUnderEyeH) / (2 * faceH);

  const smileScore = Math.max(0, (cornerElevation * 5.0) + ((0.025 - cheekRaise) * 20.0));
  const smileDetected = smileScore > 0.3;

  return {
    left_ear: earL,
    right_ear: earR,
    avg_ear: avgEAR,
    blink_velocity: blinkVelocity,
    brow_descent_left: browDescL,
    brow_descent_right: browDescR,
    brow_asymmetry: browAsym,
    lip_compression: lipCompression,
    jaw_displacement: jawDisplacement,
    jaw_angle_width: jawAngleWidth,
    mouth_corner_pull: mcPull,
    forehead_tension: foreheadTension,
    face_height_norm: faceH / iod,
    head_tilt: Math.abs(headTilt),
    temporal_x_var: xVar,
    temporal_y_var: yVar,
    eye_openness_ratio: avgEAR,
    landmark_confidence: detection_confidence,
    nose_wrinkle: dist(pt(4), pt(50)) / faceH,
    smile_score: Math.min(1.0, smileScore),
    smile_detected: smileDetected,
    corner_elevation: cornerElevation
  };
}

const SEND_INTERVAL_MS = 800;  // send every 800ms (faster real-time updates)

// Keep a single global FaceMesh instance to avoid WebAssembly race conditions and duplicate script loading
let globalMesh = null;

export default function FaceStream({ onResult, onIndicatorsUpdate, active, calibrationMode = false, userId = 'default' }) {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const camRef    = useRef(null);
  const histRef   = useRef([]);
  const lastSend  = useRef(0);
  const lastIndicatorsSend = useRef(0);
  const [fps, setFps]     = useState(0);
  const frameCount = useRef(0);
  const fpsTimer   = useRef(null);
  const workerRef  = useRef(null);

  const [libsReady, setLibsReady] = useState(!!window.FaceMesh && !!window.Camera);

  // Stable references for callback functions to prevent recreation overhead
  const onResultRef = useRef(onResult);
  const onIndicatorsUpdateRef = useRef(onIndicatorsUpdate);
  const calibrationModeRef = useRef(calibrationMode);
  const userIdRef = useRef(userId);

  useEffect(() => {
    calibrationModeRef.current = calibrationMode;
  }, [calibrationMode]);

  useEffect(() => {
    userIdRef.current = userId;
  }, [userId]);

  useEffect(() => {
    onResultRef.current = onResult;
  }, [onResult]);

  useEffect(() => {
    onIndicatorsUpdateRef.current = onIndicatorsUpdate;
  }, [onIndicatorsUpdate]);

  // Dedicated Web Worker setup per component mount
  useEffect(() => {
    workerRef.current = new Worker('/facePostWorker.js');
    workerRef.current.onmessage = (e) => {
      if (e.data.type === 'face_result' && onResultRef.current) {
        onResultRef.current(e.data.data);
      } else if (e.data.type === 'error') {
        console.error("Web Worker post error:", e.data.error);
      }
    };
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (libsReady) return;
    const checkLibs = setInterval(() => {
      if (window.FaceMesh && window.Camera) {
        setLibsReady(true);
        clearInterval(checkLibs);
      }
    }, 100);
    return () => clearInterval(checkLibs);
  }, [libsReady]);

  const handleResults = useCallback((results) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const iw = videoRef.current?.videoWidth  || 320;
    const ih = videoRef.current?.videoHeight || 240;

    if (canvas && iw && ih) {
      canvas.width = iw;
      canvas.height = ih;
      ctx.clearRect(0, 0, iw, ih);
    }

    if (!results.multiFaceLandmarks?.length) {
      const now = Date.now();
      if (onIndicatorsUpdateRef.current && (now - lastIndicatorsSend.current > 500)) {
        lastIndicatorsSend.current = now;
        onIndicatorsUpdateRef.current(null);
      }
      return;
    }

    const lm = results.multiFaceLandmarks[0];
    
    if (ctx && iw && ih) {
      let minX = iw, minY = ih, maxX = 0, maxY = 0;
      for (const p of lm) {
        const x = p.x * iw;
        const y = p.y * ih;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
      // Add a slight margin
      const padding = 15;
      minX = Math.max(0, minX - padding);
      minY = Math.max(0, minY - padding);
      maxX = Math.min(iw, maxX + padding);
      maxY = Math.min(ih, maxY + padding);
      
      ctx.strokeStyle = 'rgba(165, 200, 255, 0.8)';
      ctx.lineWidth = 2;
      ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
      // Draw corner accents
      ctx.beginPath();
      ctx.moveTo(minX, minY + 10); ctx.lineTo(minX, minY); ctx.lineTo(minX + 10, minY);
      ctx.moveTo(maxX, minY + 10); ctx.lineTo(maxX, minY); ctx.lineTo(maxX - 10, minY);
      ctx.moveTo(minX, maxY - 10); ctx.lineTo(minX, maxY); ctx.lineTo(minX + 10, maxY);
      ctx.moveTo(maxX, maxY - 10); ctx.lineTo(maxX, maxY); ctx.lineTo(maxX - 10, maxY);
      ctx.stroke();
    }

    const indicators = computeStressIndicators(lm, iw, ih, histRef.current);

    // Update history (keep last 10 frames)
    histRef.current = [
      ...histRef.current.slice(-9),
      { nx: lm[4].x, ny: lm[4].y, ear: indicators.avg_ear },
    ];

    // Throttle UI indicators update to 500ms to avoid React performance lag
    const now = Date.now();
    if (onIndicatorsUpdateRef.current && (now - lastIndicatorsSend.current > 150)) {
      lastIndicatorsSend.current = now;
      onIndicatorsUpdateRef.current(indicators);
    }

    // Send to backend every SEND_INTERVAL_MS via Web Worker
    if (now - lastSend.current > SEND_INTERVAL_MS) {
      lastSend.current = now;
      if (workerRef.current) {
        workerRef.current.postMessage({ 
          indicators, 
          timestamp: now,
          calibrationMode: calibrationModeRef.current,
          userId: userIdRef.current,
          apiBase: API_BASE
        });
      }
    }
  }, []);

  useEffect(() => {
    if (!active || !libsReady) {
      if (camRef.current) {
        camRef.current.stop();
        camRef.current = null;
      }
      if (fpsTimer.current) {
        clearInterval(fpsTimer.current);
        fpsTimer.current = null;
      }
      return;
    }

    const FaceMeshClass = window.FaceMesh;
    const CameraClass = window.Camera;

    if (!FaceMeshClass || !CameraClass) return;

    // Initialize singleton instance once
    if (!globalMesh) {
      globalMesh = new FaceMeshClass({
        locateFile: (file) => `/mediapipe/face_mesh/${file}`,
      });
      globalMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: false,  // false = faster, saves ~20ms per frame
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }

    // Bind results handler immediately upon starting the camera
    globalMesh.onResults(handleResults);

    let frameCounter = 0;

    const cam = new CameraClass(videoRef.current, {
      onFrame: async () => {
        frameCounter++;
        // Skip every other frame to process landmarks at 15fps, saving 50% CPU
        if (frameCounter % 2 !== 0) return;

        if (videoRef.current && globalMesh) {
          try {
            await globalMesh.send({ image: videoRef.current });
          } catch (e) {
            console.error("Error sending frame to MediaPipe mesh:", e);
          }
        }
      },
      width: 320,   // lower resolution = much less memory and CPU
      height: 240,
    });
    cam.start();
    camRef.current = cam;

    // FPS counter
    fpsTimer.current = setInterval(() => {
      setFps(frameCount.current);
      frameCount.current = 0;
    }, 1000);

    return () => {
      if (camRef.current) {
        camRef.current.stop();
        camRef.current = null;
      }
      if (fpsTimer.current) {
        clearInterval(fpsTimer.current);
        fpsTimer.current = null;
      }
    };
  }, [active, libsReady, handleResults]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 12, overflow: 'hidden' }}>
      <video ref={videoRef} style={{ width: '100%', height: '100%', objectFit: 'cover', background: 'var(--chat-bg)', display: 'block', transform: 'scaleX(-1)' }} playsInline />
      <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', pointerEvents: 'none' }} />
      {active && (
        <div style={{ position: 'absolute', top: 8, right: 8, background: 'var(--card-bg)',
                       color: 'var(--primary-color)', borderRadius: 6, padding: '3px 10px',
                       fontSize: '0.75rem', fontFamily: 'monospace', border: 'var(--glass-border)' }}>
          {fps} fps · Face (WebGL)
        </div>
      )}
      {!active && (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--card-bg)', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          <span>Camera Standby</span>
        </div>
      )}
    </div>
  );
}
