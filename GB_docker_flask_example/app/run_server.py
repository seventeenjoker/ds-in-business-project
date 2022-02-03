import dill
import pandas as pd
import os
import flask
import logging

from logging.handlers import RotatingFileHandler
from time import strftime


dill._dill._reverse_typemap['ClassType'] = type

app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(path_model):
    # load the pre-trained model
    with open(path_model, 'rb') as f:
        model = dill.load(f)
    print(model)
    return model


model_path = "/Users/victoria/Учеба/Машинное обучение в бизнесе/HW/final_project/GB_docker_flask_example/models/randf_classifier.dill"
md = load_model(model_path)


def is_number(s):
    try:
        float(s)  # for int, long and float
    except (ValueError, KeyError):
        return False
    return True


@app.route("/", methods=["GET"])
def general():
    return """Welcome to wine class prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        request_json = flask.request.get_json()

        if is_number(request_json['fixed_acidity']):
            fixed_acidity = request_json['fixed_acidity']
        if is_number(request_json['volatile_acidity']):
            volatile_acidity = request_json['volatile_acidity']
        if is_number(request_json['citric_acid']):
            citric_acid = request_json['citric_acid']
        if is_number(request_json['residual_sugar']):
            residual_sugar = request_json['residual_sugar']
        if is_number(request_json['volatile_acidity']):
            volatile_acidity = request_json['volatile_acidity']
        if is_number(request_json['chlorides']):
            chlorides = request_json['chlorides']
        if is_number(request_json['free_sulfur_dioxide']):
            free_sulfur_dioxide = request_json['free_sulfur_dioxide']
        if is_number(request_json['total_sulfur_dioxide']):
            total_sulfur_dioxide = request_json['total_sulfur_dioxide']
        if is_number(request_json['density']):
            density = request_json['density']
        if is_number(request_json['pH']):
            pH = request_json['pH']
        if is_number(request_json['sulphates']):
            sulphates = request_json['sulphates']
        if is_number(request_json['alcohol']):
            alcohol = request_json['alcohol']

        logger.info(
            f"{dt} Data:\n"
            f"'fixed_acidity': {fixed_acidity}, "
            f"'volatile_acidity' {volatile_acidity}, "
            f"'citric_acid': {citric_acid}, "
            f"'residual_sugar': {residual_sugar}, "
            f"'chlorides': {chlorides}, "
            f"'free_sulfur_dioxide': {free_sulfur_dioxide}, "
            f"'total_sulfur_dioxide': {total_sulfur_dioxide}, "
            f"'density': {density}, "
            f"'pH': {pH}, "
            f"'sulphates': {sulphates}, "
            f"'alcohol': {alcohol}"
        )
        try:
            preds = md.predict_proba(pd.DataFrame({
                "fixed_acidity": [fixed_acidity],
                "volatile_acidity": [volatile_acidity],
                "citric_acid": [citric_acid],
                "residual_sugar": [residual_sugar],
                "chlorides": [chlorides],
                "free_sulfur_dioxide": [free_sulfur_dioxide],
                "total_sulfur_dioxide": [total_sulfur_dioxide],
                "density": [density],
                "pH": [pH],
                "sulphates": [sulphates],
                "alcohol": [alcohol],
            }))
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = preds[:, 1][0]
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print("* Loading the model and Flask starting server... please wait until server has fully started")
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
