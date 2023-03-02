from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        lead_time = request.form["lead_time"]
        avg_price_per_room = request.form["avg_price_per_room"]
        no_of_special_requests = request.form["no_of_special_requests"]
        X = np.array([[float(lead_time), float(avg_price_per_room), float(no_of_special_requests)]])
        pred = model.predict_proba(X)[0][1]
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
