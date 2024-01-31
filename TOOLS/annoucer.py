from abc import ABC,abstractstaticmethod


class Annoucer(ABC):

    def __call__(self, person,place) -> None:
        self._annouce(person,place)

    @abstractstaticmethod
    def _annouce(person,place) -> None:
        pass


class TextAnnoucer(Annoucer):

    @staticmethod
    def _annouce(person, place):
        print(f'Person {person} enter in {place}')