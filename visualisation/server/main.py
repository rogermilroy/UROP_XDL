import os
from flask import Flask, send_from_directory, render_template
from flask_restful import reqparse, Resource, Api
from visualisation.server.resources.full_run_data import FullRunData
from visualisation.server.resources.minibatch_data import MinibatchData

# main script for starting and running the server.

# going to use flask app to create the endpoints and serve the data
app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')


api = Api(app)

# add resources to the app to serve the data.
api.add_resource(FullRunData, '/api/full_run')
api.add_resource(MinibatchData, '/api/minibatch')


if __name__ == '__main__':
    p = os.environ.get('PORT')
    # TODO Launch receiver to take in the data.

    app.run(debug=True, host='0.0.0.0', port=p, use_reloader=False)
