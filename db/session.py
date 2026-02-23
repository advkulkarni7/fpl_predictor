import os

import psycopg


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return database_url


def get_connection(autocommit: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(get_database_url())
    conn.autocommit = autocommit
    return conn
