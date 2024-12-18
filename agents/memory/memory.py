from abc import ABC, abstractmethod
from src.luxai_s3.state import EnvObs


class Memory(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, obs: EnvObs):
        pass

    @abstractmethod
    def expand(self, obs: EnvObs):
        pass

    @abstractmethod
    def reset(self):
        pass

class DefaultMemory(Memory):
    def __init__(self):
        pass

    def update(self, obs: EnvObs):
        pass

    def expand(self, obs: EnvObs):
        return obs

    def reset(self):
        pass