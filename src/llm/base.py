from abc import ABC, abstractmethod

class LLM(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def run(self, prompt: str) -> str:
        pass