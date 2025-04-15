import cv2
import json

ROI_FILE = "roi_coordinates.json"
coordinates = []
current_mouse_pos = (0, 0)

def save_roi_to_json(coords):
    with open(ROI_FILE, 'w') as file:
        json.dump(coords, file)
    print(f"\nROI Coordinates saved to {ROI_FILE}:\n{coords}")

def mouse_callback(event, x, y, flags, param):
    global coordinates, current_mouse_pos
    current_mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN and len(coordinates) < 4:
        coordinates.append((x, y))
        print(f"Clicked: ({x}, {y})")
        if len(coordinates) == 4:
            save_roi_to_json(coordinates)

def draw_dynamic_roi(frame):
    # Draw fixed lines
    for i in range(len(coordinates) - 1):
        cv2.line(frame, coordinates[i], coordinates[i + 1], (0, 255, 0), 2)

    # Draw line following the cursor
    if 0 < len(coordinates) < 4:
        cv2.line(frame, coordinates[-1], current_mouse_pos, (255, 0, 0), 2)

    # Complete the shape after 4 points
    if len(coordinates) == 4:
        cv2.line(frame, coordinates[3], coordinates[0], (0, 255, 0), 2)

    # Draw small circles at each point
    for point in coordinates:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

def select_roi():
    global coordinates, current_mouse_pos

    stream_url = "rtsp://admin:admin123@192.168.1.213/cam/realmonitor?channel=1&subtype=0&rtsp_transport=tcp&buffer_size=1024"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Failed to open the IP camera stream.")
        return

    cv2.namedWindow("Select ROI", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Select ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        draw_dynamic_roi(frame)
        cv2.imshow("Select ROI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(coordinates) == 4:  # ENTER
            break
        elif key == 27:  # ESC
            coordinates = []
            print("ROI selection canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    select_roi()
