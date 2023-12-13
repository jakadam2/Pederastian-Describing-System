from time import perf_counter

class Person:

    '''
    Idea of counting a time:
        *every entry is a start of parctial timer and after exit from roi partial time from partial time will be added to global timer
        *it means that at the end on every people shoudl be done method sth like end 
        *every loop in detector object was given a presence of each roi
    IMPORTANT: to know idea about the names look at the comment in ResultWritter file
    '''
    def __init__(self,id,) -> None:
        # here should be all of features
        self.id = id
        self.roi1_time = 0.0
        self.roi2_time = 0.0
        self.roi1_passes = 0
        self.roi2_passes = 0
        self.upper_color = 'unknown'
        self._inroi1 = False
        self._inroi2 = False

    def __str__(self) -> str:
        return f'Person({self.id})'
    
    def __repr__(self) -> str:
        return f'Person({self.id})'
     
    def _startRoi1(self) -> None:
        if self._inroi1:
            raise LookupError(f'{self} is already in roi1')
        self._roi1_ptime = perf_counter()
        self._inroi1 = True

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

    def _stopRoi2(self) -> None:
        if not self._inroi2:
            raise LookupError(f'{self} is not in roi2')
        
        self.roi2_time += (perf_counter() - self._roi2_ptime)
        self.roi2_passes += 1
        self._inroi2 = False

    def is_in_roi1(self,presence) -> None:
        if presence == self._inroi1:
            return
        elif presence == True:
            self._startRoi1()
        else:
            self._stopRoi1()

    def is_in_roi2(self,presence) -> None:
        if presence == self._inroi2:
            return
        elif presence == True:
            self._startRoi2()
        else:
            self._stopRoi2()
