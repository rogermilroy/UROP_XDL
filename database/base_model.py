from peewee import *
from playhouse.postgres_ext import *


class BaseModel(Model):
    """
    A base class to prevent repetition
    """
    class Meta:  # TODO make configurable so we are not storing sensitive info in source code.
        database = PostgresqlExtDatabase(
            database="something",
            host="something",
            password="something"
        )
        legacy_table_names = False
