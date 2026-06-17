import sys
import pandas as pd
from src.configuration.postgres_connection import PostgreSQLClient
from src.constants import POSTGRES_TABLE_NAME
from src.exception import MyException
from src.logger import logging


class Proj1Data:
    def __init__(self):
        try:
            self.pg_client = PostgreSQLClient()
        except Exception as e:
            raise MyException(e, sys)

    def export_table_as_dataframe(self, table_name=POSTGRES_TABLE_NAME) -> pd.DataFrame:
        try:
            logging.info(f"Fetching data from PostgreSQL table: {table_name}")
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.pg_client.engine)
            logging.info(f"Fetched {len(df)} rows from table: {table_name}")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def close(self):
        self.pg_client.close()
