import cv2 as cv
from ultralytics import YOLO
import sys
import cvzone
import math
from os.path import join
from person import Person
from roi import RoiReaderEmulator
import numpy as np

if len(sys.argv) < 3:
    raise ValueError("NAME OF VIDEO FILE WASN'T GIVEN")

video_name = sys.argv[1]
viedo_file = join('./data','videos',video_name)
cap = cv.VideoCapture(viedo_file)

model_name = sys.argv[2]
model_file = join('./weights',model_name)
model = YOLO(model_file)


# this should be a real roi reader 
roi_reader = RoiReaderEmulator()
roi1,roi2 = roi_reader.load()
detected = {}
# Here should be a real tracker 
tracker = None

while True:

    success,img = cap.read()
    if success == False:
        print('END')
        break

    orig_img = img.copy()
    detections = np.empty((0, 5))
    results = model(img,stream=True,verbose = False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        id = result[4]
        bbox = result[0:4]
        if roi1.include(bbox) or roi2.include(bbox):
                if result[4] not in detected.keys():
                    detected[id] = Person(id)
                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    extract = orig_img[y1 + 1:y2 -1,x1 + 1:x2 - 1]

                    # here should be referencing extract img to a model regonizning other features
                    # and writing this features to object by maybe a nice property 
                    # for now only feature is id another features will be added with models detectin it
                    # it's only my concept so it could be done better
                    # extract is a signle person

                detected[id].isInRoi1(roi1.include(bbox))
                detected[id].isInRoi2(roi2.include(bbox))




    cv.imshow('People Detection Video',img)
    cv.waitKey(1)
# instead of printing should be making a result.txt file with features of object 
# in my opinion should be sth like ResultWriter getting list of People objects and making file
print(detected)