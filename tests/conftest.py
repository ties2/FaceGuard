"""Test configuration.

MSP's suite ran only inside Docker against PostgreSQL. FaceGuard keeps that
path (``docker compose run --rm api pytest``) and adds a zero-dependency
default: a throw-away SQLite database and artifact directory, configured
*before* any application module is imported.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="faceguard-tests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ.setdefault("LOCAL_ARTIFACT_DIR", f"{_tmp}/artifacts/")
os.environ.setdefault("FACEGUARD_POLICY_PATH", f"{_tmp}/policy.yaml")

import pytest  # noqa: E402

import src.liveness.db_models  # noqa: E402,F401
import src.serving.db_models  # noqa: E402,F401
from src.serving.database import Base, engine  # noqa: E402

Base.metadata.create_all(bind=engine)


@pytest.fixture(scope="session")
def tmp_root() -> str:
    return _tmp
