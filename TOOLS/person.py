from time import perf_counter
from typing import Any
from TOOLS.annoucer import TextAnnoucer
from TOOLS.predict_chooser import *


class Person:
    '''
    Idea of counting a time:
        *every entry is a start of parctial timer and after exit from roi partial time from partial time will be added to global timer
        *it means that at the end on every people shoudl be done method sth like end 
        *every loop in detector object was given a presence of each roi
    IMPORTANT: to know idea about the names look at the comment in ResultWritter file
    '''
    _color_dict = {0:'black', 1: 'blue',2:'brown',3: 'gray', 4:'green', 5:'orange', 6:'pink', 7:'purple', 8:'red', 9:'white',10: 'yellow'}
    _gender_dict = {0:'male',1:'female'}
    _bag_dict = {0:False,1:True}
    _hat_dict = {0:False,1:True}
    _annoucer = TextAnnoucer()
    _tollerance_time = 5
    _chooser = SamePredictChooser

    def __init__(self,id) -> None:
        self.id = id
        self.roi1_time = 0.0
        self.roi2_time = 0.0
        self.roi1_passes = 0
        self.roi2_passes = 0
        self._inroi1 = False
        self._inroi2 = False
        self._pass_time1 = 0
        self._pass_time1 = 0
        self._bag_chooser = self._chooser(2,self._bag_dict)
        self._hat_chooser = self._chooser(2,self._hat_dict)
        self._gender_chooser = self._chooser(2,self._gender_dict)
        self._upper_chooser = self._chooser(11,self._color_dict)
        self._lower_chooser = self._chooser(11,self._color_dict)
        
    def __call__(self, predicts) -> None:
        # predicts = predicts.squeeze(0)
        self.upper_color = self._upper_chooser(predicts[0])
        self.lower_color = self._lower_chooser(predicts[1])
        # self.gender = self._gender_chooser(torch.tensor([1-predicts[2],predicts[2]]))
        # self.hat = self._hat_chooser(torch.tensor([1-predicts[3],predicts[3]]))
        # self.bag = self._bag_chooser(torch.tensor([1-predicts[4],predicts[4]]))
        self.gender = self._gender_dict[torch.where(predicts[2] > .5, 1.0, 0.0).int().item()]
        self.bag = self._bag_dict[torch.where(predicts[3] > .5, 1.0, 0.0).int().item()]
        self.hat = self._hat_dict[torch.where(predicts[4] > .5, 1.0, 0.0).int().item()]
        

    def __str__(self) -> str:
        return f'Person({self.id})'
    
    def __repr__(self) -> str:
        return f'Person({self.id})'
     
    def _startRoi1(self) -> None:
        if self._inroi1:
            raise LookupError(f'{self} is already in roi1')
        self._roi1_ptime = perf_counter()
        self._inroi1 = True
        self._annoucer(self.id,'roi1')

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
        self._annoucer(self.id,'roi2')

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