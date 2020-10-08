import pymongo
from flask_restful import reqparse, Resource, abort


class VisData(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('training_run')
        self.parser.add_argument('minibatch_start')
        self.parser.add_argument('minibatch_end')
        # Use a local db. Makes the most sense for now. May change to be configurable in the future.
        # currently db used is training_data. This is due to code reuse. TODO check
        self.db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    def get(self):
        args = self.parser.parse_args(strict=True)
        # training_run gets us the collection. type the specific record.
        data = list(self.db[args['training_run']].find(
            {"minibatch": {"$gt": int(args['minibatch_start']),
                           "$lt": int(args['minibatch_end'])}},
            {'minibatch': True, 'data': True, '_id': False}))
        if data is not None:
            return {'data': data}
        else:
            abort(404, message="Analysis type: {} "
                               "for training_run {} doesn't appear to be in "
                               "the database right now.".format(args['analysis_type'],
                                                                args['training_run']))
