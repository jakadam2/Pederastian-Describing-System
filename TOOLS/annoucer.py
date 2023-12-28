from abc import ABC,abstractmethod


class Annoucer(ABC):

    @abstractmethod
    def annouce(person,place) -> None:
        pass



class TextAnnoucer(Annoucer):

    @staticmethod
    def annouce(person, place):
        print(f'Person {person} enter in {place}')