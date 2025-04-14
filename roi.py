# roi.py

import cv2

# Define the ROI (x1, y1, x2, y2)
ROI_COORDS = (350, 250, 1700, 650)

def is_inside_roi(x, y):
    x1, y1, x2, y2 = ROI_COORDS
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_roi(frame):
    x1, y1, x2, y2 = ROI_COORDS
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
