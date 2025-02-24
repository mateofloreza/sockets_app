import cv2
import asyncio
import websockets
import base64
import threading
import time

# Shared resources
raw_frame = None
processed_frame = None
frame_lock = threading.Lock()
processed_lock = threading.Lock()

def capture_frames():
    """Continuously captures frames from the RTSP feed."""
    global raw_frame
    cap = cv2.VideoCapture("rtsp://192.168.178.25:8554/video_feed")
    
    if not cap.isOpened():
        print("Failed to open RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received, retrying...")
            time.sleep(0.1)
            continue

        with frame_lock:
            raw_frame = frame.copy()
        
        time.sleep(0.01)

def process_frames():
    """Processes frames (adds test rectangle) in a separate thread."""
    global raw_frame, processed_frame
    while True:
        # Get raw frame
        with frame_lock:
            if raw_frame is None:
                time.sleep(0.01)
                continue
            current_frame = raw_frame.copy()

        # Add processing here (test rectangle)
        cv2.rectangle(current_frame, 
                     (100, 100),  # Start point
                     (200, 200),  # End point
                     (0, 255, 0),  # Green color
                     2)  # Thickness

        # Update processed frame
        with processed_lock:
            processed_frame = current_frame

        time.sleep(0.01)

async def stream_frames(websocket):
    """Sends processed frames over websocket."""
    global processed_frame
    while True:
        with processed_lock:
            if processed_frame is not None:
                frame_copy = processed_frame.copy()
            else:
                await asyncio.sleep(0.01)
                continue

        # Encode and send
        ret, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ret:
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(jpg_as_text)

        await asyncio.sleep(0.033)  # ~30 FPS

def start_websocket_server():
    async def server_main():
        async with websockets.serve(stream_frames, '0.0.0.0', 8765):
            print("WebSocket server started on ws://0.0.0.0:8765")
            await asyncio.Future()
    
    asyncio.run(server_main())

if __name__ == '__main__':
    # Start capture thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    # Start processing thread
    process_thread = threading.Thread(target=process_frames, daemon=True)
    process_thread.start()

    # Start websocket thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down")