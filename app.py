from flask import Flask, request

app = Flask(__name__)

def load_model():
    import torch
    from torch import nn
    from model import ScorePredictionModel
    model_state = torch.load("model/In-class-Assignment-Final-Model.ph")
    model = ScorePredictionModel()
    model.load_state_dict(model_state)
    return model

def predict(x):
    import torch
    import numpy as np
    model = load_model()
    x = torch.from_numpy(np.array([0.12])).float()
    val = model(x).detach().numpy()
    return float(val)


@app.route('/')
def hello_world():
    val = predict(0.21)
    print(val)
    return f'helllo'

if __name__ == '__main__':
    app.run()
