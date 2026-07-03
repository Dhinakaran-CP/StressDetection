import re

with open('frontend/src/pages/Dashboard.js', 'r', encoding='utf-8') as f:
    content = f.read()

old_controls = '''      {/* SHUTDOWN CONTROLS */}
      <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 1000 }}>
        <button 
          title="Turn off Application Completely"
          style={{
            width: '45px',
            height: '45px',
            borderRadius: '50%',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 0 10px rgba(220, 53, 69, 0.6)'
          }}
          onClick={() => {
            if(window.confirm("Are you sure you want to shut down the entire app? This will kill both the backend server and frontend server.")) {
              fetch('http://127.0.0.1:5000/api/shutdown/all', { method: 'POST' })
                .then(() => alert("Entire app is shutting down. You can now close this tab."))
                .catch(e => console.error(e));
            }
          }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18.36 6.64a9 9 0 1 1-12.73 0"></path>
            <line x1="12" y1="2" x2="12" y2="12"></line>
          </svg>
        </button>
      </div>'''

new_controls = '''      {/* SHUTDOWN & RESTART CONTROLS */}
      <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 1000, display: 'flex', gap: '10px' }}>
        <button 
          title="Restart Backend Server"
          style={{
            width: '45px',
            height: '45px',
            borderRadius: '50%',
            backgroundColor: '#fd7e14',
            color: 'white',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 0 10px rgba(253, 126, 20, 0.6)'
          }}
          onClick={() => {
            if(window.confirm("Are you sure you want to restart the backend server?")) {
              fetch('http://127.0.0.1:5000/api/restart/backend', { method: 'POST' })
                .then(() => alert("Backend is restarting. Please wait a few seconds before analyzing again."))
                .catch(e => console.error(e));
            }
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="1 4 1 10 7 10"></polyline>
            <polyline points="23 20 23 14 17 14"></polyline>
            <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path>
          </svg>
        </button>
        <button 
          title="Shutdown Backend Server"
          style={{
            width: '45px',
            height: '45px',
            borderRadius: '50%',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 0 10px rgba(220, 53, 69, 0.6)'
          }}
          onClick={() => {
            if(window.confirm("Are you sure you want to shut down the backend server?")) {
              fetch('http://127.0.0.1:5000/api/shutdown/backend', { method: 'POST' })
                .then(() => alert("Backend is shutting down."))
                .catch(e => console.error(e));
            }
          }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18.36 6.64a9 9 0 1 1-12.73 0"></path>
            <line x1="12" y1="2" x2="12" y2="12"></line>
          </svg>
        </button>
      </div>'''

content = content.replace(old_controls, new_controls)

with open('frontend/src/pages/Dashboard.js', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated controls in Dashboard.")
