"""SQLAlchemy engine/session factory.

PostgreSQL is the production target (partial unique indexes, concurrency).
SQLite is supported for local runs and CI so the full test-suite runs without
Docker; partial indexes are declared for both dialects in ``db_models``.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.serving.config import settings

SQLALCHEMY_DATABASE_URL = settings.sqlalchemy_url

_connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
