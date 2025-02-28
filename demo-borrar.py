import cv2
import asyncio
import websockets
import base64
import threading
import time
from ultralytics import YOLO
import supervision as sv
import requests
import os

# Shared resources
raw_frame = None
processed_frame = None
latest_detections = None
cropped_frame = None
name_label = "Mateo"
frame_lock = threading.Lock()
processed_lock = threading.Lock()
detection_lock = threading.Lock()
name_lock = threading.Lock()
llm_lock = threading.Lock()


model = YOLO("weights.pt")
# Web server configuration
IMAGE_DIR = "./images"
IMAGE_FILENAME = "latest.jpg"
IMAGE_PATH = os.path.join(IMAGE_DIR, IMAGE_FILENAME)
SERVER_PORT = 9595

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
    global raw_frame, processed_frame, latest_detections, cropped_frame, name_label
    while True:
        # Get raw frame
        with frame_lock:
            if raw_frame is None:
                time.sleep(0.01)
                continue
            current_frame = raw_frame.copy()
        results = model.track(source=current_frame, conf=0.65, device='cpu', max_det=1, persist=True)
        detections = sv.Detections.from_ultralytics(results[0])
        # box_annotator = sv.BoxAnnotator()
        # label_annotator = sv.LabelAnnotator(text_scale = 0.5)
        if len(results[0].boxes) > 0:
            x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            cropped_image = current_frame[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(f'{IMAGE_DIR}/badge/', IMAGE_FILENAME), cropped_image)
            #results[0].save_crop(IMAGE_DIR, file_name=IMAGE_FILENAME)
        labels = []

        for box in results[0].boxes:
            labels.append(f'Hello {name_label}')
        # annotated_image = box_annotator.annotate(scene=current_frame, detections=detections)
        # annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        #current_frame = cv2.resize(current_frame, (640,480))
        with detection_lock:
            latest_detections = (detections, labels)
        time.sleep(0.01)

def llm_question():
    global name_label
    llm_endpoint = "http://100.74.10.42:8080/v1/models/internvl2:predict"
    while True:
        time.sleep(2)
        with name_lock:
                name_label = "Mateo"
        time.sleep(2)


async def stream_frames(websocket):
    """Sends processed frames over websocket."""
    global processed_frame, raw_frame, latest_detections
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=1)
    while True:
        with frame_lock:
            if raw_frame is not None:
                frame_copy = raw_frame.copy()
            else:
                await asyncio.sleep(0.01)
                continue
        with detection_lock:
            if latest_detections is not None:
                detections, labels = latest_detections
       
        if detections is not None:
            annotated_frame = box_annotator.annotate(scene=frame_copy, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        else:
            annotated_frame = frame_copy

        # Encode and send
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
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

    # Start llm thread
    llm_thread = threading.Thread(target=llm_question, daemon=True)
    llm_thread.start()

    # Start websocket thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()



    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down")