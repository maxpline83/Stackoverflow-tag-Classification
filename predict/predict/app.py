from flask import Flask
from predict.predict import run as run_predict
import json

app = Flask(__name__)


@app.route('/predict', methods=["GET"])
def predict():
    artefact_path = "C:/Users/maxim/Programation/EPF/S9/PocToProd/Lab2/poc-to-prod-capstone/train/data/artefacts/c2024-01-09-12-11-52"
    input_text = ['i got problem with pandas and dataframe']

    model = run_predict.TextPredictionModel.from_artefacts(artefact_path)

    prediction = model.predict(input_text, top_k=5)
    prediction_labelled = [model.labels_to_index[str(idx)] for idx in prediction[0]]
    top_prediction = {"top "+str(idx): val for idx, val in enumerate(prediction_labelled)}
    prediction_labelled_dict = {"input_text": input_text[0], "predicted label": top_prediction}
    result_json = json.dumps(prediction_labelled_dict)
    return result_json


if __name__ == '__main__':
    app.run(debug=True)
