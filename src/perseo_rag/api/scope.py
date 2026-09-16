from typing import Protocol

from fastapi import Request

from perseo_rag.security import ScopeContext


class ScopeResolver(Protocol):
    def __call__(self, request: Request) -> ScopeContext: ...
