import re

with open('backend/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_route = '''@app.route('/api/restart/backend', methods=['POST'])
def restart_backend():
    print("[Shutdown] Restarting backend server...")
    def restart_self():
        import time, os, sys, subprocess
        time.sleep(1)
        # On Windows, ping localhost for a few seconds to let the port free up, then restart
        if os.name == 'nt':
            cmd = f'ping 127.0.0.1 -n 3 > nul && "{sys.executable}" "{sys.argv[0]}"'
            subprocess.Popen(cmd, shell=True)
        else:
            cmd = f'sleep 2 && "{sys.executable}" "{sys.argv[0]}"'
            subprocess.Popen(cmd, shell=True)
        os._exit(0)
    import threading
    threading.Thread(target=restart_self).start()
    return jsonify({'status': 'success', 'message': 'Backend is restarting...'})

'''

if '/api/restart/backend' not in content:
    content = content.replace("@app.route('/api/shutdown/backend', methods=['POST'])", new_route + "@app.route('/api/shutdown/backend', methods=['POST'])")
    with open('backend/app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Added restart backend route.")
else:
    print("Route already exists.")
