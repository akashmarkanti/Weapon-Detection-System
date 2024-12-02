from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import cvzone

app = Flask(__name__)
model = YOLO('best.pt')  # Load the YOLOv10 model

# Set up the video capture
cap = cv2.VideoCapture(0)  #loop

def generate_frames():
    while True: 
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run object detection
            results = model(frame)
            for info in results:
                parameters = info.boxes
                for box in parameters:
                    # Extract box coordinates, confidence, and class name
                    x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
                    confidence = box.conf[0].numpy() * 100  # Confidence in percentage
                    class_detected_number = int(box.cls[0].numpy())
                    class_detected_name = results[0].names[class_detected_number]

                    # Draw rectangle and add label on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{class_detected_name} {confidence:.1f}%', (x1, y1 - 10), scale=1.5, thickness=2)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Home page for video feed

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
