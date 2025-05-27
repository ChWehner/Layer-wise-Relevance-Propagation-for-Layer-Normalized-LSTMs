'''
Backend to generate predictions and explanations for the demonstrator.

author: Christoph Wehner; wehner.ch@gmx.de
------------------------------------------------------------------------------------------------------------------------
'''


from flask import Flask, request
from make_prediction import predict_for_package
import json

# initiate flask app
app = Flask(__name__)


################
# flask routes #
################

@app.route("/health")
def hello():
    return {"msg": "ok"}


@app.route("/snaps", methods=['POST'])
def process_snap():
    if request.is_json:
        snapshots = request.get_data()
        package = json.loads(snapshots)
        prediction = predict_for_package(package)
        return {"prediction ": prediction}
    return {"error": "Request was not structured as JSON."}


# No need to run the development server any more, since we start with gunicorn.
# app.run(host="0.0.0.0")
