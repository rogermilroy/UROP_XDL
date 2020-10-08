import os

from flask import Flask, render_template
from flask_cors import CORS
from flask_restful import Api
from resources.minibatch_data import MinibatchData
from resources.training_runs import TrainingRuns
from resources.vis_data import VisData

# main script for starting and running the server.

# going to use flask app to create the endpoints and serve the data
app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')

CORS(app)
api = Api(app)

# add resources to the app to serve the data.
api.add_resource(VisData, '/api/vis_data')
api.add_resource(MinibatchData, '/api/minibatch')
api.add_resource(TrainingRuns, '/api/training_runs')


if __name__ == '__main__':
    p = os.environ.get('PORT')
    # TODO Launch receiver to take in the data.
    print(p)

    app.run(debug=True, host='0.0.0.0', port=p, use_reloader=True)
