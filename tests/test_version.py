import tomllib
from pathlib import Path

from perseo_rag import __version__


def test_runtime_version_matches_project_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
    assert __version__ == metadata["project"]["version"]
