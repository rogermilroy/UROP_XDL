from playhouse.postgres_ext import *


class BaseModel(Model):
    """
    A base class to prevent repetition
    """
    class Meta:  # TODO make configurable so we are not storing sensitive info in source code.
        database = PostgresqlExtDatabase(
            "test_postgres",
            host="localhost",
            user="postgres",
            password="pa55w0rd"
        )
        legacy_table_names = False