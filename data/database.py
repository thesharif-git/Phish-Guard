from sqlmodel import SQLModel, create_engine
from typing import Generator

from core.config import settings

# Database URL from settings
sqlite_url = settings.DATABASE_URL

# echo=True is helpful for debugging to see the generated SQL
engine = create_engine(sqlite_url, echo=False)

def create_db_and_tables():
    """Creates the database tables if they do not exist."""
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator:
    """Dependency to get a database session."""
    from sqlmodel import Session
    with Session(engine) as session:
        yield session
