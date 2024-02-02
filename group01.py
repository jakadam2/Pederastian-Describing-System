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
    people_in_rois = 0
    roi1_passages = 0
    roi2_passages = 0
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
        
        if roi1.include(bbox): 
            color  = (255,0,0)
        elif roi2.include(bbox):
            color = (0,255,0)
        else:
            color = (0, 0, 255)

        if detected[id].in_rois:
            people_in_rois += 1

        roi1_passages += detected[id].roi1_passages
        roi2_passages += detected[id].roi2_passages




        width = 80 if id >= 10 else 40
        gender_text = 'F' if detected[id].gender == 'female' else 'M'
        bag_text = 'Bag' if detected[id].bag else 'No Bag'
        hat_text = 'Hat' if detected[id].hat else 'No Hat'
        #bbox
        img = cv.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                thickness
            )
        #background for ID
        img = cv.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[0] + width,bbox[1] + 70),
                (255,255,255),
                thickness = -1
            )
        #ID on this background
        cv.putText(
                img,
                f'{id}',
                (bbox[0], bbox[1] + 50),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                color,
                3
            )
        #background for attributes
        img = cv.rectangle(
                img,
                (bbox[0], bbox[3]),
                (bbox[2] + 20,bbox[3] + 60),
                (255,255,255),
                thickness = -1
            )
        #gender  
        cv.putText(
                img,
                f'Gender:{gender_text}',
                (bbox[0] + 2, bbox[3] + 12),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0,0,0),
                2
            )
        #bag hat
        cv.putText(
                img,
                f'{bag_text} {hat_text}',
                (bbox[0] + 2, bbox[3] + 29),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0,0,0),
                2
            )    
        #colors
        cv.putText(
                img,
                f'U-L:{detected[id].upper_color}-{detected[id].lower_color}',
                (bbox[0] + 2, bbox[3] + 46),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0,0,0),
                2
            )
    #background for general
    img = cv.rectangle(
            img,
            (0,0),
            (400,200),
            (255,255,255),
            thickness = -1
        )   
    #people in roi
    cv.putText(
            img,
            f'People in ROI:{people_in_rois}',
            (2, 46),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            2
        )
    #present people
    cv.putText(
        img,
        f'Total perons:{len(present_people)}',
        (2, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        2
    )  
    #roi1 passages
    cv.putText(
        img,
        f'Passages in ROI1:{roi1_passages}',
        (2, 114),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        2
    )   
    #roi2 passages
    cv.putText(
        img,
        f'Passages in ROI2:{roi2_passages}',
        (2, 148),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0),
        2
    )
    #ROI1
    img = cv.rectangle(
            img,
            roi1.bbox[0],
            roi1.bbox[1],
            (0,  0,0),
            thickness = 3
        )
    #ROI2
    img = cv.rectangle(
        img,
        roi2.bbox[0],
        roi2.bbox[1],
        (0,  0,0),
        thickness = 3
    )
    #ROI2 digit
    cv.putText(
        img,
        f'2',
        (roi2.bbox[0][0] + 2,roi2.bbox[0][1] + 70),
        cv.FONT_HERSHEY_SIMPLEX,
        3,
        (0,0,0),
        2
    )
    #ROI1 digit
    cv.putText(
        img,
        f'1',
        (roi1.bbox[0][0] + 2,roi1.bbox[0][1] + 70),
        cv.FONT_HERSHEY_SIMPLEX,
        3,
        (0,0,0),
        2
    )

    for id in detected:
        if id not in present_people:
            detected[id].is_in_roi1(False)
            detected[id].is_in_roi2(False)
    cv.namedWindow('People Detection Video', cv.WINDOW_NORMAL)
    # cv.resizeWindow("People Detection Video", 1080, 1920) 
    cv.imshow('People Detection Video',img)
    cv.waitKey(0)

for id in detected:
    detected[id].end_rois()
      
result_writer.write_ans(detected.values())