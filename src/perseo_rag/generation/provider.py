from typing import Protocol

from perseo_rag.generation.models import GenerationOutput, GenerationRequest


class GenerationProvider(Protocol):
    @property
    def key(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GenerationOutput: ...
