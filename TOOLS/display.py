import cv2 as cv


def draw_person(img,person,bbox,color):

    width = 80 if person.id >= 10 else 40
    gender_text = 'F' if person.gender == 'female' else 'M'
    bag_text = 'Bag' if person.bag else 'No Bag'
    hat_text = 'Hat' if person.hat else 'No Hat'
    #bbox
    img = cv.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            thickness = 2
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
            f'{person.id}',
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
            0.5,
            (0,0,0),
            2
        )
    #bag hat
    cv.putText(
            img,
            f'{bag_text} {hat_text}',
            (bbox[0] + 2, bbox[3] + 29),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            2
        )    
    #colors
    cv.putText(
            img,
            f'U-L:{person.upper_color}-{person.lower_color}',
            (bbox[0] + 2, bbox[3] + 46),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            2
        )
    return img


def draw_general(img,people_in_rois,people_amount,roi1_passages,roi2_passages,roi1,roi2):

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
        f'Total persons:{people_amount}',
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
    return img
