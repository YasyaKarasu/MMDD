from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_working_directory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep default configuration discovery away from user-owned repository files."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path_factory.mktemp("working-directory"))
        yield
