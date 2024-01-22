import numpy as np
import math

import sys
from os.path import join
from pathlib import Path

from TOOLS.person import Person
from TOOLS.roi import RoiReaderEmulator
from TOOLS.result_writer import ResultWriter

import cv2 as cv
from ultralytics import YOLO
from boxmot import DeepOCSORT
import torch

from PAR.resnet_extractor import Resnet50Extractor
from PAR.gender_classifier import GenderClassiefierResNet
from torchvision.models import ResNet50_Weights as rw
from torchvision.transforms.functional import adjust_contrast


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

feature_extractor = Resnet50Extractor()
par_model = GenderClassiefierResNet(feature_extractor).to('cuda')
par_model.load_state_dict(torch.load('./weights/resnetgender.pt'))
par_model.eval()

transform = rw.IMAGENET1K_V2.transforms()

while True:

    success,img = cap.read()
    mmimg = img.copy()
    img = cv.convertScaleAbs(img,alpha = 0.2, beta = 150)
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
                extract = orig_img[y1 + 1:y2 -1,x1 + 1:x2 - 1]
                #extract = cv.convertScaleAbs(extract,alpha = 1.2, beta = - 50)
                extract = torch.from_numpy(extract.astype(np.float32))
                extract = extract.permute(2,0,1)
                extract = transform(extract).to('cuda').unsqueeze(0)
                gender_ratio = par_model(extract)
                if gender_ratio > 0.5:
                    detected[id].gender = 'female'
                else:
                    detected[id].gender = 'male'
        # rois needs additional thinking because now it vunerable on blinking bboxies
            detected[id].is_in_roi1(roi1.include(bbox))
            detected[id].is_in_roi2(roi2.include(bbox))
            present_people.add(id)  

        img = cv.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                thickness
            )
        cv.putText(
                img,
                f'id: {id} {detected[id].gender}',
                (bbox[0], bbox[1]-10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )
        

    for id in detected:
        if id not in present_people:
            detected[id].is_in_roi1(False)
            detected[id].is_in_roi2(False)

    cv.imshow('People Detection Video',img)
    cv.imshow('People Detection Vidgggeo',mmimg)
    cv.waitKey(1)

for id in detected:
    detected[id].is_in_roi1(False)
    detected[id].is_in_roi2(False)   
    
result_writer.write_ans(detected.values())

# TODO: think about when and how decide about pederastian features !!!!!!!!!!!!!!!!!!!
# TODO: think about contrast 
# TODO: think about blinking bboxes