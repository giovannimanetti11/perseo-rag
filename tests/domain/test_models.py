from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from perseo_rag.domain import Scope


def test_domain_entities_are_immutable() -> None:
    scope = Scope(id=uuid4(), slug="acme", name="Acme")

    with pytest.raises(FrozenInstanceError):
        scope.name = "Changed"  # type: ignore[misc]
