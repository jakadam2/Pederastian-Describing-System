from time import perf_counter
from typing import Any
from TOOLS.annoucer import TextAnnoucer
from TOOLS.predict_chooser import *

chooser = AvgPredictChooser

class Person:
    '''
    Idea of counting a time:
        *every entry is a start of parctial timer and after exit from roi partial time from partial time will be added to global timer
        *it means that at the end on every people shoudl be done method sth like end 
        *every loop in detector object was given a presence of each roi
    IMPORTANT: to know idea about the names look at the comment in ResultWritter file
    '''

    color_dict = {0:'black', 1: 'blue',2:'brown',3: 'gray', 4:'green', 5:'orange', 6:'pink', 7:'purple', 8:'red', 9:'white',10: 'yellow'}
    gender_dict = {0:'male',1:'female'}
    bag_dict = {0:False,1:True}
    hat_dict = {0:False,1:True}

    def __init__(self,id) -> None:
        # here should be all of features
        self.id = id
        self.roi1_time = 0.0
        self.roi2_time = 0.0
        self.roi1_passes = 0
        self.roi2_passes = 0
        self.upper_color = 'unknown'
        self._inroi1 = False
        self._inroi2 = False
        self._annoucer = TextAnnoucer()
        self._tollerance_time = 5
        self._pass_time1 = 0
        self._pass_time1 = 0
        self._bag_chooser = chooser(2,Person.bag_dict)
        self._hat_chooser = chooser(2,Person.hat_dict)
        self._gender_chooser = chooser(2,Person.gender_dict)
        self._upper_chooser = chooser(11,Person.color_dict)
        self._lower_chooser = chooser(11,Person.color_dict)
        
    def __call__(self, predicts) -> None:
        predicts = predicts.squeeze(0)
        self.upper_color = self._upper_chooser(predicts[0:11])
        self.lower_color = self._lower_chooser(predicts[11:22])
        self.gender = self._gender_chooser(predicts[22:24])
        self.hat = self._hat_chooser(predicts[24:26])
        self.bag = self._bag_chooser(predicts[26:28])

    def __str__(self) -> str:
        return f'Person({self.id})'
    
    def __repr__(self) -> str:
        return f'Person({self.id})'
     
    def _startRoi1(self) -> None:
        if self._inroi1:
            raise LookupError(f'{self} is already in roi1')
        self._roi1_ptime = perf_counter()
        self._inroi1 = True
        self._annoucer.annouce(self.id,'roi1')

    def _stopRoi1(self) -> None:
        if not self._inroi1:
            raise LookupError(f'{self} is not in roi1')
        
        self.roi1_time += (perf_counter() - self._roi1_ptime)
        self.roi1_passes += 1
        self._inroi1 = False

    def _startRoi2(self) -> None:
        if self._inroi2:
            raise LookupError(f'{self} is already in roi2')
        self._roi2_ptime = perf_counter()
        self._inroi2 = True
        self._annoucer.annouce(self.id,'roi2')

    def _stopRoi2(self) -> None:
        if not self._inroi2:
            raise LookupError(f'{self} is not in roi2')
        
        self.roi2_time += (perf_counter() - self._roi2_ptime)
        self.roi2_passes += 1
        self._inroi2 = False

    def is_in_roi1(self,presence) -> None:
        if presence == self._inroi1:
            return
        
        elif presence:
            self._pass_time1 = 0
            self._startRoi1()

        else:
            self._pass_time1 += 1
            if self._pass_time1 == self._tollerance_time:
                self._stopRoi1()

    def is_in_roi2(self,presence) -> None:
        if presence == self._inroi2:
            return
        elif presence == True:
            self._startRoi2()
        else:
            self._stopRoi2()

    def end_rois(self):
        if self._inroi1:
            self._stopRoi1()
        if self._inroi2:
            self._stopRoi2()
