from flask_restful import reqparse, Resource, abort
import pymongo


class TrainingRuns(Resource):

    def __init__(self):
        # Use a local db. Makes the most sense for now. May change to be configurable in the future.
        # currently db used is training_data. This is due to code reuse. TODO change?
        self.db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    def get(self):
        # get all collections.
        data = self.db.list_collection_names()
        if data is not None:
            return {'data': data}
        else:
            abort(404, message="There are no collections in this database right now.")
