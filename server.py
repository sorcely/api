import flask
from flask import Flask, request
from flask_restful import Api, Resource

import os
import sys
import json
import time

app = Flask(__name__)
api = Api(app)

PROD = True

# Setup fact check code
from run import fakenews
fakenews = fakenews('google')

class main(Resource):

    def get(self):
        query = request.args.get('query')
        check_news = request.args.get('news')

        # If question is given
        if query:
            # To return time taken
            start_time = time.time()

            # Runs the fact checker
            results = fakenews.check(query, check_news)

            return results

        return {None}

# Add the api view
api.add_resource(main, '/')

if __name__ == '__main__':
    host = '0.0.0.0' if PROD else '127.0.0.1'
    port = 80 # The standard host for webapps

    # Starts the server
    app.run(host=host, port=port, debug=not PROD)
