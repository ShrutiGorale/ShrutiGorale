import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Database setup
def setup_database():
    conn = sqlite3.connect('people_tracking.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS visits
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT)''')
    conn.commit()
    return conn, c

def log_visit(conn, c):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO visits (timestamp) VALUES (?)", (now,))
    conn.commit()

# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Process frame
def process_frame(frame, net, output_layers):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    height, width, _ = frame.shape
    people_detected = False

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Confidence threshold and class ID for 'person'
                    people_detected = True
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, people_detected

def main():
    # Setup database
    conn, c = setup_database()

    # Load YOLO
    net, output_layers = load_yolo()

    # Open webcam (change to IP camera URL if needed)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, people_detected = process_frame(frame, net, output_layers)

        if people_detected:
            log_visit(conn, c)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
    
