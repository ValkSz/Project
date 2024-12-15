import cv2
import csv
import numpy as np

video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("out_vid.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

csv_path = "./test_interpolated.csv"
file = open(csv_path, 'r')
results = csv.DictReader(file)
boxes_by_frame = {}

id_list =[]
for row in results:
    id_list.append(row["car_id"])
id_list = list(dict.fromkeys(id_list))

for i in range(len(id_list)):
    id_list[i] = int(id_list[i])


with open(csv_path, mode='r') as file:
    reader = csv.DictReader(file)
    test_val = 0
    for row in reader:
        frame_number = int(row['frame_nmr'])
        car_id = int(row['car_id'])
        car_bbox = row['car_bbox'].split()
        x_min_car, y_min_car, x_max_car, y_max_car = map(float, car_bbox)

        license_plate_bbox = row['license_plate_bbox'].split()
        x_min_lp, y_min_lp, x_max_lp, y_max_lp = map(float, license_plate_bbox)

        if frame_number not in boxes_by_frame:
            boxes_by_frame[frame_number] = []


        boxes_by_frame[frame_number].append(('car', x_min_car, y_min_car, x_max_car, y_max_car))
        boxes_by_frame[frame_number].append(('license_plate', x_min_lp, y_min_lp, x_max_lp, y_max_lp))

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    if frame_number in boxes_by_frame:
        for label, x_min, y_min, x_max, y_max in boxes_by_frame[frame_number]:
            color = (0, 255, 0) if label == 'car' else (255, 0, 0)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 5)


    # frame = cv2.resize(frame, (1280, 720))
    # cv2.imshow('Frame with Bounding Boxes', frame)


    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()
