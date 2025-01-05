from scipy.interpolate import interp1d
from ultralytics import YOLO
from sort.sort import *
import numpy as np
import easyocr
import cv2
import csv
import re

#If you want to save the file with data then set this to 1
save_file = 1

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr','car_id','car_bbox','license_plate_bbox','license_plate_bbox_score','license_number','license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

def check_lp(lp, ids):
    x1, y1, x2, y2, score, class_id = lp
    foundIt = False
    for j in range(len(ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break
    if foundIt:
        return ids[car_indx]
    return -1, -1, -1, -1, -1

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0,
                                           kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                original_row = [p for p in data if
                                int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(
                                    float(car_id))][0]
                row['license_plate_bbox_score'] = original_row[
                    'license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row[
                    'license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

def complies(text):
    if len(text) <=8 and len(text) >5 and text != None:
        check_plate = re.findall('[1-9A-Z]', text)
        if check_plate != None and len(check_plate) == len(text):
            return True
        else:
            return False
    else:
        return False

reader = easyocr.Reader(['en'], gpu=False)
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if complies(text):
            return text, score

    return None, None

results = {}

tracker = Sort()
model = YOLO('./models/yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
base_path = str(glob.glob('video_input/*.mp4')).split("'")[1]
cap = cv2.VideoCapture(base_path)

f_nr = -1
ret = True
while ret:
    f_nr += 1
    ret, frame = cap.read()
    if ret:
        results[f_nr] = {}
        detections = model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:
                detections_.append([x1, y1, x2, y2, score])

        track_ids = tracker.update(np.asarray(detections_))

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = check_lp(license_plate, track_ids)

            if car_id != -1:

                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[f_nr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]}, 'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text, 'bbox_score': score, 'text_score': license_plate_text_score}}

temp_path = './temp.csv'
write_csv(results, temp_path)

with open(temp_path, 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

interpolated_data = interpolate_bounding_boxes(data)

header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('data.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

if(os.path.exists(temp_path) and os.path.isfile(temp_path)):
  os.remove(temp_path)
else:
    pass
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("out_vid.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

csv_path = "data.csv"
file = open(csv_path, 'r')
results = csv.DictReader(file)
boxes_by_frame = {}
car_license_plate_dict = {}


id_list =[]
for row in results:
    id_list.append(row["car_id"])
id_list = list(dict.fromkeys(id_list))
file.close()

for i in range(len(id_list)):
    try:
        id_list[i] = int(id_list[i])
    except ValueError:
        pass


with open(csv_path, mode='r') as file:
    reader = csv.DictReader(file)
    test_val = 0
    new_score = 0.0
    new_license_plate = ""
    high_score = 0.0
    initial_car_id = 1
    for row in reader:
        frame_number = int(row['frame_nmr'])
        car_id = int(row['car_id'])
        car_bbox = row['car_bbox'].split()
        x_min_car, y_min_car, x_max_car, y_max_car = map(float, car_bbox)

        license_plate_bbox = row['license_plate_bbox'].split()
        x_min_lp, y_min_lp, x_max_lp, y_max_lp = map(float, license_plate_bbox)

        if frame_number not in boxes_by_frame:
            boxes_by_frame[frame_number] = []

        boxes_by_frame[frame_number].append(('car', x_min_car, y_min_car, x_max_car, y_max_car, car_id))
        boxes_by_frame[frame_number].append(('license_plate', x_min_lp, y_min_lp, x_max_lp, y_max_lp, ''))

        if initial_car_id == car_id:
            try:
                new_score = float(row['license_number_score'])
                if new_score > high_score:
                    new_score = high_score
                    new_license_plate = row["license_number"]
                else:
                    pass
            except ValueError:
                pass
        elif initial_car_id != car_id:
            car_license_plate_dict[initial_car_id] = new_license_plate
            initial_car_id = car_id

    print("Done with reading the file")
    file.close()


license_plate = {}
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number in boxes_by_frame:
        for label, x_min, y_min, x_max, y_max, car_id in boxes_by_frame[frame_number]:
            try:
                license_plate = car_license_plate_dict[car_id]
            except:
                pass
            if label == 'car':
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, license_plate, (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            elif label == "license_plate":
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

    out.write(frame)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Frame with Bounding Boxes', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

if save_file == 0:
    if(os.path.exists(csv_path) and os.path.isfile(csv_path)):
        os.remove(csv_path)
    else:
        pass

cap.release()
out.release()
cv2.destroyAllWindows()
