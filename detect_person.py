import cv2
import numpy as np
import tensorflow as tf
import threading
import time
from roi import is_inside_roi, draw_roi
from db import save_to_db
from excel import save_to_excel
from queue import Queue

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="D:/project/1/tflite_model/yolov5s_416_fp16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.4
PERSON_CLASS_ID = 0

# Threaded video capture
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def preprocess(frame):
    input_shape = input_details[0]['shape'][1:3]
    image = cv2.resize(frame, tuple(input_shape))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image / 255.0, axis=0).astype(np.float32)
    return image

def run_inference(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

def apply_nms(boxes, scores, conf_thresh, iou_thresh):
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    if isinstance(indices, np.ndarray):
        return indices.flatten().tolist()
    elif isinstance(indices, list):
        return [int(i) if isinstance(i, (np.integer, int)) else int(i[0]) for i in indices]
    else:
        return []

def process_output(output, frame_shape):
    h, w, _ = frame_shape
    results = []
    for det in output:
        x_center, y_center, width, height, conf = det[:5]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        class_score = class_probs[class_id]
        score = conf * class_score

        if score > CONF_THRESHOLD and class_id == PERSON_CLASS_ID:
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            results.append([x1, y1, x2, y2, float(score)])
    return results

# I/O thread worker
def io_worker(queue):
    while True:
        count = queue.get()
        if count is None:
            break
        save_to_db(count)
        save_to_excel(count)
        queue.task_done()

def process_frames():
    stream_url = "rtsp://admin:admin123@192.168.1.213/cam/realmonitor?channel=1&subtype=0&rtsp_transport=tcp&buffer_size=1024"
    stream = VideoStream(stream_url)
    io_queue = Queue()
    threading.Thread(target=io_worker, args=(io_queue,), daemon=True).start()

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        # FPS calculation
        curr_time = time.time()
        delta_time = curr_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0.0
        prev_time = curr_time

        output = run_inference(frame)
        detections = process_output(output, frame.shape)

        boxes = []
        scores = []
        for x1, y1, x2, y2, score in detections:
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)

        indices = apply_nms(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        final_detections = [detections[i] for i in indices]

        person_count = 0
        for x1, y1, x2, y2, score in final_detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if is_inside_roi(cx, cy):
                person_count += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                # Label: 'person: XX%'
                label = f"person: {int(score * 100)}%"
                font_scale = 0.7
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                label_x = x1
                label_y = max(y1 - 10, label_height + 10)

                # Draw label background
                cv2.rectangle(frame,
                              (label_x, label_y - label_height - baseline),
                              (label_x + label_width, label_y + baseline),
                              (0, 255, 0), cv2.FILLED)

                # Draw label text
                cv2.putText(frame, label, (label_x, label_y),
                            font, font_scale, (0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)

        if person_count > 0:
            io_queue.put(person_count)

        draw_roi(frame)

        # Draw FPS and count
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"Persons: {person_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display window
        cv2.namedWindow("Pedestrian Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Pedestrian Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Pedestrian Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    io_queue.put(None)
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frames()
