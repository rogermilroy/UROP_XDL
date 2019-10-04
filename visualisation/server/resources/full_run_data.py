from flask_restful import reqparse, Resource, abort
import pymongo


class FullRunData(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('training_run')
        self.parser.add_argument('analysis_type')
        self.parser.add_argument('weight_selection_type')
        # Use a local db. Makes the most sense for now. May change to be configurable in the future.
        # currently db used is training_data. This is due to code reuse. TODO change?
        self.db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    # TODO use url parsing instead?
    def get(self):
        args = self.parser.parse_args(strict=True)
        # training_run gets us the collection. type the specific record.
        data = self.db[args['training_run']].find_one({"analysis_type": args['analysis_type'],
                                                       "weight_analysis_type": args[
                                                           'weight_analysis_type'],
                                                       "minibatch": {"$exists": False}})
        if data is not None:
            return {'data': data}
        else:
            abort(404, message="Minibatch {} of type {} in training_run {} doesn't appear to be in "
                               "the database right now.".format(args['minibatch'],
                                                                args[type],
                                                                args['training_run']))
