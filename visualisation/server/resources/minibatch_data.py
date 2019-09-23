from flask_restful import reqparse, Resource, abort
import pymongo


class MinibatchData(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('training_run')
        self.parser.add_argument('type')
        self.parser.add_argument('minibatch')
        # Use a local db. Makes the most sense for now. May change to be configurable in the future.
        # currently db used is training_data. This is due to code reuse. TODO change?
        self.db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    # TODO use url parsing instead?
    def get(self):
        args = self.parser.parse_args(strict=True)
        # training_run gets us the collection. minibatch and type the specific record.
        data = self.db[args['training_run']].find_one({"minibatch": args['minibatch'],
                                                       "type": args['type']})
        if data is not None:
            return {'data': data}
        else:
            abort(404, message="Minibatch {} of type {} in training_run {} doesn't appear to be in "
                               "the database right now.".format(args['minibatch'],
                                                                args[type],
                                                                args['training_run']))
