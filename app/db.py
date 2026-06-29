from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()


def create_session_factory(database_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, future=True)
    if database_url.startswith("sqlite"):
        # Concurrent pollers (and the atomic step claim in MissionManager) rely on
        # SQLite serializing writers and waiting — not erroring — when a write lock
        # is briefly held. Own these guarantees explicitly instead of inheriting a
        # driver default: WAL lets readers proceed while one writer commits, and a
        # busy timeout makes a contended write wait rather than raise "database is
        # locked".
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # pragma: no cover - exercised via engine
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(session_factory: sessionmaker) -> None:
    engine = session_factory.kw["bind"]
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def session_scope(session_factory: sessionmaker) -> Generator:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
