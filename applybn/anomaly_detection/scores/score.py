from abc import ABC, abstractmethod


class Score(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def score(self, X):
        pass
