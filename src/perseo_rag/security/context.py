from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True, slots=True)
class ScopeContext:
    id: UUID


def apply_scope_context(session: Session, scope: ScopeContext) -> None:
    session.execute(
        text("SELECT set_config('perseo.scope_id', :scope_id, true)"),
        {"scope_id": str(scope.id)},
    )


@contextmanager
def scoped_session(
    session_factory: sessionmaker[Session],
    scope: ScopeContext,
) -> Iterator[Session]:
    with session_factory.begin() as session:
        apply_scope_context(session, scope)
        yield session
