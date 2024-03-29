from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from sort.sort import *
import os

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
mot_tracker = Sort()
cap = cv2.VideoCapture('./sample.mp4')
vehicles = [2, 3, 5, 7]

ret = True
cv2.namedWindow('Processed Frame', cv2.WINDOW_GUI_EXPANDED)
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    detections = coco_model(frame)[0]
    vehicles_boxes = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            vehicles_boxes.append([x1, y1, x2, y2, score])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    track_ids = mot_tracker.update(np.asarray(vehicles_boxes))

    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        plate_resized = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        border_size = 20
        plate_with_border = cv2.copyMakeBorder(plate_resized, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        start_x = int(x1)
        start_y = int(y1) - plate_with_border.shape[0] - 150
        if start_y < 0:
            start_y = 0
        end_y = start_y + plate_with_border.shape[0]
        end_x = start_x + plate_with_border.shape[1]
        if end_y <= frame.shape[0] and end_x <= frame.shape[1]:
            frame[start_y:end_y, start_x:end_x] = plate_with_border
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)
        plate_binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(plate_binary, lang='eng', config='--psm 7')
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
        text_x = start_x + (plate_with_border.shape[1] - text_size[0]) // 2
        text_y = start_y - 30
        cv2.rectangle(frame, (text_x - 10, text_y + 10), (text_x + text_size[0] + 10, text_y - text_size[1] - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.imshow('Processed Frame', frame)
    cv2.resizeWindow('Processed Frame', 1080, 720)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
