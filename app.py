# from flask import Flask, render_template, Response
# import cv2
# from ultralytics import YOLO

# app = Flask(__name__)

# # Load YOLO model
# model = YOLO("yolov8n.pt")

# # Start webcam
# cap = cv2.VideoCapture(0)

# def generate_frames():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Resize for performance
#         frame = cv2.resize(frame, (640, 480))

#         # YOLO detection
#         results = model(frame, imgsz=416, verbose=False)

#         # Draw boxes
#         annotated_frame = results[0].plot()

#         # Convert to JPEG
#         ret, buffer = cv2.imencode('.jpg', annotated_frame)
#         frame = buffer.tobytes()

#         # Stream frame
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/video')
# def video():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model (CPU only)
model = YOLO("yolov8n.pt")  

cap = None  # Webcam will start on button press
streaming = False  # Flag to control webcam
frame_count = 0  

def generate_frames():
    global cap
    global frame_count
    while streaming:
        frame_count += 1
        if frame_count % 2 != 0:  # Skip every 2nd frame for FPS boost
            continue

        success, frame = cap.read()
        if not success:
            break

        # Dynamic resize for performance
        target_width = 640
        height, width, _ = frame.shape
        scale = target_width / width
        frame = cv2.resize(frame, (int(width*scale), int(height*scale)))

        # YOLO detection (CPU)
        results = model(frame, imgsz=416, verbose=False)
        annotated_frame = results[0].plot()

        # Count objects
        object_counts = {}
        for cls in results[0].names.values():
            object_counts[cls] = 0
        for r in results[0].boxes.cls:
            cls_name = results[0].names[int(r)]
            object_counts[cls_name] += 1

        # Overlay counts on frame
        y = 30
        for name, count in object_counts.items():
            cv2.putText(annotated_frame, f"{name}: {count}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    if streaming:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Webcam not started."


@app.route('/start', methods=['POST'])
def start():
    global cap, streaming
    if not streaming:
        cap = cv2.VideoCapture(0)
        streaming = True
    return ("", 204)


@app.route('/stop', methods=['POST'])
def stop():
    global cap, streaming
    streaming = False
    if cap:
        cap.release()
        cap = None
    return ("", 204)


if __name__ == "__main__":
    app.run(debug=True)