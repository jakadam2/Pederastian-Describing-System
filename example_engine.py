import numpy as np
import math

import sys
from os.path import join
from pathlib import Path

from person import Person
from roi import RoiReaderEmulator
from result_writer import ResultWriter

import cv2 as cv
from ultralytics import YOLO
from boxmot import DeepOCSORT
import torch


if len(sys.argv) < 3:
    raise ValueError("NAME OF VIDEO FILE WASN'T GIVEN")

color = (0, 0, 255)
thickness = 2
fontscale = 0.5

video_name = sys.argv[1]
viedo_file = join('./data','videos',video_name)
cap = cv.VideoCapture(viedo_file)

model_name = sys.argv[2]
model_file = join('./weights',model_name)
model = YOLO(model_file)

roi_reader = RoiReaderEmulator()
roi1,roi2 = roi_reader.load()

detected = {}
result_writer = ResultWriter('ex.json')


tracker = DeepOCSORT( 
    model_weights= Path('./weights/osnet_ain_x1_0_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=True,
)

while True:

    success,img = cap.read()
    if success == False:
        print('END')
        break

    orig_img = img.copy()
    detections = np.empty((0, 6))
    results = model(img,stream=True,verbose = False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].type(torch.int).cpu()
                conf = math.ceil((box.conf[0] * 100)) / 100
                currentArray = np.array([x1, y1, x2, y2, conf,0])
                detections = np.vstack((detections, currentArray))

    tracks= tracker.update(detections,orig_img)
    present_people = set()
    for track in tracks:
        # showing bbox
        bbox = track[0:4].astype(int)
        id = track[4].astype(int)
        img = cv.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                thickness
            )
        cv.putText(
                img,
                f'id: {id}',
                (bbox[0], bbox[1]-10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )
        # features
        if roi1.include(bbox) or roi2.include(bbox):
            if id not in detected.keys(): # that means that we see this pearson first time 
                x1, y1, x2, y2 = bbox
                extract = orig_img[y1 + 1:y2 -1,x1 + 1:x2 - 1]
                # here should be referencing extract img to a model regonizning other features
                # and writing this features to object by maybe a nice property 
                # for now only feature is id another features will be added with models detectin it
                # it's only my concept so it could be done better
                # extract is a signle person 
                detected[id] = Person(int(id)) # this object should store info about person
            # rois needs additional thinking because now it vunerable on blinking bboxies
            detected[id].is_in_roi1(roi1.include(bbox))
            detected[id].is_in_roi2(roi2.include(bbox))
            present_people.add(id)

    for id in detected:
        if id not in present_people:
            detected[id].is_in_roi1(False)
            detected[id].is_in_roi2(False)

    cv.imshow('People Detection Video',img)
    cv.waitKey(1)

for id in detected:
    detected[id].is_in_roi1(False)
    detected[id].is_in_roi2(False)   
    
result_writer.write_ans(detected.values())