from typing import Annotated

from fastapi import APIRouter, Depends, Request

from perseo_rag.application import QueryHandler, QueryInput
from perseo_rag.api.contracts import CitationResponse, QueryRequest, QueryResponse
from perseo_rag.api.scope import ScopeResolver
from perseo_rag.security import ScopeContext


def build_query_router(
    query_handler: QueryHandler,
    scope_resolver: ScopeResolver,
) -> APIRouter:
    router = APIRouter()

    def resolve_scope(request: Request) -> ScopeContext:
        return scope_resolver(request)

    @router.post("/query", response_model=QueryResponse)
    def answer_query(
        payload: QueryRequest,
        scope: Annotated[ScopeContext, Depends(resolve_scope)],
    ) -> QueryResponse:
        result = query_handler.answer(
            scope,
            QueryInput(
                question=payload.question,
                collection_id=payload.collection_id,
                limit=payload.limit,
            ),
        )
        answer = result.answer

        return QueryResponse(
            status=answer.status.value,
            text=answer.text,
            citations=tuple(
                CitationResponse(
                    id=citation.id,
                    source_ref=citation.source_ref,
                )
                for citation in answer.citations
            ),
            abstention_reason=(
                answer.abstention_reason.value if answer.abstention_reason is not None else None
            ),
            evidence_count=result.evidence_count,
            context_chars=result.context_chars,
            context_truncated=result.context_truncated,
        )

    return router
