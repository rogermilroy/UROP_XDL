from abc import ABC

import pymongo


class DatabaseInterface(ABC):

    def __init__(self):
        super().__init__()

    def insert_many(self, data, **kwargs) -> None:
        """
        Base insert_many. TODO abstract for relational dbs as well. kwargs.
        :param data: Data to insert
        :param kwargs: Keyword arguments
        :return: None
        """
        pass


class MongoInterface(DatabaseInterface):

    def __init__(self, db_connection_information):
        super().__init__()
        self.db = pymongo.MongoClient(db_connection_information).training_data

    def insert_many(self, data, **kwargs) -> None:
        """
        Method to insert many items into a database.
        :param data: The data to insert.
        :param kwargs: collection must be present
        :return:
        """
        if 'collection' not in kwargs:
            raise Exception("collection must be specified...")
        self.db[kwargs['collection']].insert_many(data)
