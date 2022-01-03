from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app=Flask(__name__)

MODEL_PATH='model_aug_rps.h5'

model = load_model(MODEL_PATH)
model._make_predict_function()

print('Model loaded. check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(300, 300))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, moe='caffe')

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def indexx():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method =='POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)

        preds = model_predict(file_path, model)

        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])
        return result
    return None

if __name__ =='__main__':
    app.run(debug=True)

