import os
from typing import Annotated
from fastapi import Depends
from sqlmodel import SQLModel, Session, create_engine
from .decision import *

if "DATABASE_URL" in os.environ:
    url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://")
else:
    url = f"sqlite:///./test_db.sqlite3"

connect_args = {}
if "postgresql://" in url:
    connect_args = {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }

engine = create_engine(
    url, pool_pre_ping=True, pool_recycle=3600, connect_args=connect_args
)


def get_session():
    with Session(engine) as session:
        yield session


def create_db(drop: bool = False):
    if drop:
        SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


DatabaseConnection = Annotated[Session, Depends(get_session)]
