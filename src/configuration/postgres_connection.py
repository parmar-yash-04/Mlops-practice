import os
import sys
import psycopg2
from psycopg2 import OperationalError
from sqlalchemy import create_engine
from src.constants import (
    POSTGRES_HOST_KEY,
    POSTGRES_PORT_KEY,
    POSTGRES_DATABASE_KEY,
    POSTGRES_USER_KEY,
    POSTGRES_PASSWORD_KEY,
)
from src.exception import MyException
from dotenv import load_dotenv

load_dotenv()

class PostgreSQLClient:
    def __init__(self):
        try:
            self.host = os.getenv(POSTGRES_HOST_KEY, "localhost")
            self.port = os.getenv(POSTGRES_PORT_KEY, "5432")
            self.dbname = os.getenv(POSTGRES_DATABASE_KEY, "postgres")
            self.user = os.getenv(POSTGRES_USER_KEY, "postgres")
            self.password = os.getenv(POSTGRES_PASSWORD_KEY, "postgres")
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
            )
            self.connection.autocommit = True
            self.engine = create_engine(
                f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
            )
        except OperationalError as e:
            raise MyException(e, sys)

    def close(self):
        if self.connection and not self.connection.closed:
            self.connection.close()
