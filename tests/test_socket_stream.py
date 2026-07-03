
import socketio
import time
import numpy as np

# Standard Python Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")

@sio.event
def status(data):
    print('Server status:', data)

@sio.event
def stress_update(data):
    print('Received stress update:', data)

def test_streaming():
    try:
        sio.connect('http://localhost:5000')
        
        # Simulate Audio Stream
        print("Sending audio chunks...")
        for i in range(5):
            # Create dummy audio chunk (float array)
            # 44100 Hz, 0.5 seconds
            chunk = np.random.uniform(-1, 1, int(44100 * 0.5)).tolist()
            sio.emit('stream_audio', {'audio': chunk, 'sr': 44100})
            time.sleep(0.5)
            
        # Simulate Video Frame (dummy base64)
        print("Sending moving video frame...")
        # Minimal valid base64 image (1x1 pixel)
        dummy_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP////////////8////////////////////////////////////////////////////////////////////////////wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAAA//EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8A/wA//9k="
        sio.emit('stream_video', {'image': dummy_image})
        
        time.sleep(2)
        sio.disconnect()
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == '__main__':
    test_streaming()
