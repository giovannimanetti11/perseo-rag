from dataclasses import dataclass, field
from uuid import UUID

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from perseo_rag.api import create_app
from perseo_rag.application import QueryInput, QueryResult
from perseo_rag.config import Settings
from perseo_rag.generation import AnswerStatus, GroundedAnswer
from perseo_rag.grounding import EvidenceCitation
from perseo_rag.security import ScopeContext


@dataclass
class QueryHandlerFixture:
    calls: list[tuple[ScopeContext, QueryInput]] = field(default_factory=list)

    def answer(self, scope: ScopeContext, query: QueryInput) -> QueryResult:
        self.calls.append((scope, query))
        citation = EvidenceCitation(
            id="S1",
            scope_id=scope.id,
            chunk_id=UUID(int=2),
            document_version_id=UUID(int=3),
            document_id=UUID(int=4),
            source_ref="fixture:source",
        )
        return QueryResult(
            answer=GroundedAnswer(
                status=AnswerStatus.ANSWERED,
                text="Grounded answer",
                citations=(citation,),
                abstention_reason=None,
            ),
            evidence_count=1,
            context_chars=20,
            context_truncated=False,
        )


def _scope_resolver(request: Request) -> ScopeContext:
    assert request.url.path == "/query"
    return ScopeContext(UUID(int=1))


def test_query_route_uses_resolved_scope_outside_request_body() -> None:
    handler = QueryHandlerFixture()
    app = create_app(
        Settings(_env_file=None),
        query_handler=handler,
        scope_resolver=_scope_resolver,
    )
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "question": "  explain this  ",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "answered",
        "text": "Grounded answer",
        "citations": [{"id": "S1", "source_ref": "fixture:source"}],
        "abstention_reason": None,
        "evidence_count": 1,
        "context_chars": 20,
        "context_truncated": False,
    }
    assert len(handler.calls) == 1
    scope, query = handler.calls[0]
    assert scope.id == UUID(int=1)
    assert query.question == "explain this"
    assert query.limit == 5


def test_query_route_rejects_scope_from_request_body() -> None:
    handler = QueryHandlerFixture()
    app = create_app(
        Settings(_env_file=None),
        query_handler=handler,
        scope_resolver=_scope_resolver,
    )
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "question": "question",
            "scope_id": str(UUID(int=999)),
        },
    )

    assert response.status_code == 422
    assert handler.calls == []


def test_query_route_is_not_registered_without_host_dependencies() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    assert client.post("/query", json={"question": "question"}).status_code == 404


def test_query_dependencies_must_be_configured_together() -> None:
    with pytest.raises(ValueError):
        create_app(
            Settings(_env_file=None),
            query_handler=QueryHandlerFixture(),
        )
