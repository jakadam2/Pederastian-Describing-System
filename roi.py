import json
from os.path import join


class RoiReaderEmulator:
    @staticmethod
    def load(*args,**kwargs):
        return RoiEmulator(),RoiEmulator()


class RoiReader:
    # the idea is that roi knows real size of themselfs in pixel(to make a hetmetic containing check) so load method cannot be static
    # the second option is make a load static without giving a real size and give a Roi object method update to give them real size
    def __init__(self,x,y) -> None:
        self._size_x = x
        self._size_y = y

    def load(self,roi_path):
        path = join('./config',roi_path)
        with open(path) as f:
            config = json.load(f)
            roi1_config = config['roi1']
            roi2_config = config['roi2']
            roi1 = Roi(roi1_config['x'],roi1_config['y'],roi1_config['w'],roi1_config['h'],self._size_x,self._size_y)
            roi2 = Roi(roi2_config['x'],roi2_config['y'],roi2_config['w'],roi2_config['h'],self._size_x,self._size_y)
            return roi1,roi2


class RoiEmulator:
    @staticmethod
    def include(bbox):
        return True


class Roi:
    # x1,y1 left uppper corner
    # x2,y2 right bottom corner
    def __init__(self,x1,y1,w,h,image_sizex,image_sizey) -> None:
        self._x = x1 * image_sizex
        self._y = y1 * image_sizey
        self._w = w * image_sizex
        self._h = h * image_sizey

    def __include_point(self,x,y) -> bool:
        return x >= self._x and x <= self._x + self._w and y >= self._y and y <= self._y + self._h

    def include(self,bbox) -> bool:       
        #bbox format: [x1,y1,x2,y2]
        x1,y1,x2,y2 = bbox
        bbx = (x1,x2)
        bby = (y1,y2)
        for x in bbx:
            for y in bby:
                if self.__include_point(x,y):
                    return True
        return False