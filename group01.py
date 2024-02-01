import numpy as np
import math

from os.path import join
from pathlib import Path

from TOOLS.person import Person
from TOOLS.result_writer import ResultWriter
from TOOLS.roi import RoiReader

import cv2 as cv
from ultralytics import YOLO
from boxmot import DeepOCSORT
import torch

from PAR.multi_task import DMTPAR,DMTPARpart,AMTPAR,AMTPARpart
from torchvision.models import ResNet18_Weights as rw

from TOOLS.argparser import Parser
from TOOLS.bg_remover import BgRemover

bgr = BgRemover()

parser = Parser()
arguments = parser.parse()

color = (0, 0, 255)
thickness = 2
fontscale = 0.5

video_name = arguments.video
viedo_file = join('./data','videos',video_name)
cap = cv.VideoCapture(viedo_file)

model_file = join('./weights','yolov8l.pt')
model = YOLO(model_file)

roi1,roi2 = RoiReader(1080,1920).load(arguments.configuration)

detected = {}
result_writer = ResultWriter(arguments.results)

tracker = DeepOCSORT( 
    model_weights= Path('./weights/osnet_ain_x1_0_msmt17.pt'),
    device='cuda:0',
    fp16=True,
)

par_modeld = DMTPAR()
par_modeld.load_state_dict(torch.load('./weights/color_multi.pt'))
par_modeld.eval()
color_model = DMTPARpart(par_modeld)

par_model = AMTPAR()
par_model.load_state_dict(torch.load('./weights/multi_model.pt'))
par_model.eval()
attr_model = AMTPARpart(par_model)

transform = rw.IMAGENET1K_V1.transforms()

SPARSE = 50
iterator = 0

while True:
    if iterator == SPARSE:
        iterator = 0
    iterator += 1

    success,img = cap.read()
    if success == False:
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
        bbox = track[0:4].astype(int)
        id = track[4].astype(int)

        if id not in detected.keys() or iterator == SPARSE:
            x1, y1, x2, y2 = bbox
            extract = orig_img[y1 + 1:y2 -1,x1 + 1:x2 - 1]
            detected[id] = Person(int(id))
            extract = orig_img[y1 + 1:y2 -1,x1 + 1:x2 - 1]
            extract = torch.from_numpy(extract.astype(np.float32))
            extract = extract.permute(2,0,1)
            color_extract = bgr.clahe(extract) 
            extract = transform(extract).to('cuda').unsqueeze(0)
            color_extract = transform(color_extract).to('cuda').unsqueeze(0)
            upper_color,lower_color = color_model(color_extract)
            bag,gender,hat = attr_model(extract)
            detected[id]([upper_color,lower_color,gender,bag,hat])

        detected[id].is_in_roi1(roi1.include(bbox))
        detected[id].is_in_roi2(roi2.include(bbox))
        present_people.add(id)  
        
        if roi1.include(bbox) or roi2.include(bbox): 
            color  = (255, 0, 0)
        else:
            color = (0, 0, 255) 

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
                (bbox[0], bbox[1]-36),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                1
            )
        cv.putText(
                img,
                f'hat:{detected[id].hat} bag:{detected[id].bag}',
                (bbox[0], bbox[1]-23),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                1
            )     
        cv.putText(
                img,
                f'U:{detected[id].upper_color} L:{detected[id].lower_color}',
                (bbox[0], bbox[1]-10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                1
            )      

    for id in detected:
        if id not in present_people:
            detected[id].is_in_roi1(False)
            detected[id].is_in_roi2(False)


    img = cv.rectangle(
            img,
            roi1.bbox[0],
            roi1.bbox[1],
            (0,  255,0),
            3
        )
    
    img = cv.rectangle(
        img,
        roi2.bbox[0],
        roi2.bbox[1],
        (0,  255,0),
        3
    )

    cv.imshow('People Detection Video',img)
    cv.waitKey(1)

for id in detected:
    detected[id].end_rois()
      
result_writer.write_ans(detected.values())