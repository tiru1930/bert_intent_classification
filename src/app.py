import config
import torch
import flask
import time
from flask import Flask
from flask import request
from model import BERTBaseUncased
import functools
import torch.nn as nn
import joblib
import numpy as np 


app = Flask(__name__)

MODEL = None
DEVICE = "cuda"
PREDICTION_DICT = dict()
memory = joblib.Memory("../input/", verbose=0)

@memory.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    outputs_argmax = np.argmax(torch.sigmoid(outputs).cpu().detach().numpy(),axis=1).tolist()
    return outputs_argmax

@app.route("/")
def WelcomeMessage():
    return "Welcome to BERT Intent Classifier"
    
@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        'positive': str(positive_prediction),
        'negative': str(negative_prediction),
        'sentence': str(sentence),
        'time_taken': str(time.time() - start_time)
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased(77)
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()