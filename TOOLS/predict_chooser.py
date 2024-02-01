from abc import ABC,abstractmethod
from typing import Any
import torch
from queue import PriorityQueue


class PredictChooser(ABC):

    def __init__(self,feature_dim,feature_dict) -> None:
        self._feature_dim = feature_dim
        self._fdict = feature_dict

    def __call__(self,predict) -> Any:
        return self._parse(predict)

    @abstractmethod
    def _parse(self,predict) -> Any:
        pass


class SamePredictChooser(PredictChooser):

    def __init__(self, feature_dim, feature_dict) -> None:
        super().__init__(feature_dim, feature_dict)

    def _parse(self, predict) -> Any:
        return self._fdict[int(torch.argmax(predict))]
    

class MaxPredictChooser(PredictChooser):
    
    def __init__(self, feature_dim, feature_dict) -> None:
        super().__init__(feature_dim, feature_dict)
        self._max = None
        self._max_conf = 0
        self._softmax = torch.nn.Softmax(0)

    def _parse(self, predict) -> Any:
        predict = self._softmax(predict)
        if float(torch.max(predict)) > self._max_conf:
            self._max_conf = float(torch.max(predict))
            self._max = int(torch.argmax(predict))
        return self._fdict[self._max]
    

class AvgPredictChooser(PredictChooser):

    def __init__(self, feature_dim, feature_dict) -> None:
        super().__init__(feature_dim, feature_dict)
        self._current_state = torch.zeros(feature_dim)
        self._curren_pred = 0
        self._softmax = torch.nn.Softmax(0)

    def _parse(self, predict) -> Any:
        predict = self._softmax(predict).to('cpu')
        self._current_state *= self._curren_pred
        self._current_state += predict
        self._curren_pred += 1
        self._current_state /= self._curren_pred
        return self._fdict[int(torch.argmax(self._current_state))]
    

class KMaxPredictChooser(PredictChooser):

    def __init__(self, feature_dim, feature_dict,k = 5) -> None:
        super().__init__(feature_dim, feature_dict)
        self._q = PriorityQueue(k)
        self._softmax = torch.nn.Softmax(0)

    def _parse(self, predict) -> Any:
        predict = self._softmax(predict).to('cpu')
        if not self._q.full():
            self._q.put((float(torch.max(predict)),predict))
        else:
            value,element = self._q.get()
            if value > float(torch.max(predict)):
                self._q.put((value,element))
            else:
                self._q.put((float(torch.max(predict)),predict))
        i = 0
        current_state = torch.zeros(self._feature_dim)
        print(self._q.qsize())
        to_add = set()
        while not self._q.empty():
            value,element = self._q.get()
            print(value)
            current_state *= i
            current_state += element
            i += 1
            current_state /= i
            to_add.add((value,element))
        for pair in to_add:
            self._q.put(pair)
        return self._fdict[int(torch.argmax(current_state))]