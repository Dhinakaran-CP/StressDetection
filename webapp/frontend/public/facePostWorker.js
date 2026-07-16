// Web Worker — runs in its own OS thread
// Receives face indicator objects from main thread
// Sends them to the Flask backend without blocking MediaPipe or the UI

let pendingPost = false;  // rate limiter — one POST in flight at a time

self.onmessage = async function(e) {
    const { indicators, timestamp, calibrationMode, userId, apiBase } = e.data;
    
    // Drop this frame if a POST is already in flight
    // This prevents queue buildup when server is slow
    if (pendingPost) return;
    
    pendingPost = true;
    try {
        const base = apiBase || 'http://127.0.0.1:5000';
        const url = calibrationMode 
            ? `${base}/api/calibrate/face_sample` 
            : `${base}/api/stream/face`;
            
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ indicators, timestamp, user_id: userId || 'default' }),
        });
        const data = await response.json();
        // Send result back to main thread
        self.postMessage({ type: 'face_result', data });
    } catch (err) {
        self.postMessage({ type: 'error', error: err.message });
    } finally {
        pendingPost = false;
    }
};
