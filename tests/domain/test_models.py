from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from perseo_rag.domain import Tenant


def test_domain_entities_are_immutable() -> None:
    tenant = Tenant(id=uuid4(), slug="acme", name="Acme")

    with pytest.raises(FrozenInstanceError):
        tenant.name = "Changed"  # type: ignore[misc]
