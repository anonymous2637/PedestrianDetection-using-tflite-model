import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import threading
import time
import json
from queue import Queue
from gpiozero import LED
from gpiozero.pins.lgpio import LGPIOFactory

# Use LGPIOFactory for gpiozero
factory = LGPIOFactory()
led = LED(18, pin_factory=factory)  # Change GPIO if needed

# Load ROI
with open("roi_coordinates.json", "r") as f:
    roi_coordinates = json.load(f)

def is_inside_roi(x, y):
    if len(roi_coordinates) != 4:
        return False
    pts = np.array(roi_coordinates, np.int32)
    return cv2.pointPolygonTest(pts, (x, y), False) >= 0

def draw_roi(frame):
    if len(roi_coordinates) == 4:
        pts = np.array(roi_coordinates, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="/home/admin/Pedestrian-Using-tflite/tflite_model/lite2.tflite", num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.05
PERSON_CLASS_ID = 0
SAVE_INTERVAL = 10

# Shared flag for LED control
person_detected = False

def led_control_loop():
    while True:
        if person_detected:
            led.on()
            time.sleep(0.1)
            led.off()
            time.sleep(0.1)
        else:
            led.on()
            time.sleep(0.1)

# Start LED control thread
threading.Thread(target=led_control_loop, daemon=True).start()

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                try:
                    self.ret, self.frame = self.cap.read()
                    if not self.ret:
                        time.sleep(0.1)
                except cv2.error as e:
                    print(f"[OpenCV Error] {e}")
                    time.sleep(0.1)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def preprocess(frame):
    input_shape = input_details[0]['shape'][1:3]
    image = cv2.resize(frame, tuple(input_shape))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if input_details[0]['dtype'] == np.uint8:
        image = np.expand_dims(image, axis=0).astype(np.uint8)
    else:
        image = np.expand_dims(image / 255.0, axis=0).astype(np.float32)
    return image

def run_inference(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    return boxes, class_ids, scores

def apply_nms(boxes, scores, conf_thresh, iou_thresh):
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    if isinstance(indices, np.ndarray):
        return indices.flatten().tolist()
    elif isinstance(indices, list):
        return [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]
    else:
        return []

def process_output(boxes, class_ids, scores, frame_shape):
    class_ids = np.array(class_ids).flatten().tolist()
    scores = np.array(scores).flatten().tolist()
    boxes = np.array(boxes).tolist()

    detections = []
    height, width, _ = frame_shape
    for i in range(min(len(class_ids), len(scores), len(boxes))):
        if int(class_ids[i]) == PERSON_CLASS_ID and scores[i] > CONF_THRESHOLD:
            y_min, x_min, y_max, x_max = boxes[i]
            x = int(x_min * width)
            y = int(y_min * height)
            w = int((x_max - x_min) * width)
            h = int((y_max - y_min) * height)
            detections.append((x, y, x + w, y + h, scores[i]))
    return detections

def process_frames():
    global person_detected
    stream_url = "rtsp://admin:admin123@192.168.1.213/cam/realmonitor?channel=1&subtype=0&rtsp_transport=tcp&buffer_size=1024"
    stream = VideoStream(stream_url)

    prev_time = time.time()
    detection_counter = 0

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        boxes, class_ids, scores = run_inference(frame)
        detections = process_output(boxes, class_ids, scores, frame.shape)

        boxes_list = []
        scores_list = []
        for x1, y1, x2, y2, score in detections:
            boxes_list.append([x1, y1, x2 - x1, y2 - y1])
            scores_list.append(score)

        indices = apply_nms(boxes_list, scores_list, CONF_THRESHOLD, IOU_THRESHOLD)
        final_detections = [detections[i] for i in indices]

        person_count = 0
        for x1, y1, x2, y2, score in final_detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if is_inside_roi(cx, cy):
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                label = f"person: {int(score * 100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Update LED control flag
        person_detected = person_count > 0

        draw_roi(frame)
        fps = 1.0 / (time.time() - prev_time)
        prev_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"Persons: {person_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Pedestrian Detection", frame)

        if cv2.getWindowProperty("Pedestrian Detection", cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.waitKey(1) & 0xFF == ord('q'):
            break

    person_detected = False
    stream.stop()
    led.off()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frames()
