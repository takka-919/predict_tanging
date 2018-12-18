# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
from chainer import datasets, iterators, optimizers, serializers
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from datetime import datetime
from flask import Flask, jsonify

class CNN(chainer.Chain):

    def __init__(self, n_mid=512, n_out=5):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=3, ksize=3, stride=1, pad=1)
            self.fc1 = L.Linear(None, n_mid)
            self.fc2 = L.Linear(None, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, 3)
        h = self.fc1(h)
        h = self.fc2(h)
        return h

application = Flask(__name__)
application.config['JSON_AS_ASCII'] = False
@application.route('/', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'GET':
    return render_template('index.html')
  if request.method == 'POST':
    # アプロードされたファイルを保存する
    f = request.files['image']
    filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
    f.save(filepath)
    # モデルを使って判定する
    model = L.Classifier(CNN())
    serializers.load_npz('model_tanging.npz', model)
    image = Image.open(filepath)
    img_resize = image.resize((256, 256))
    img_resize = np.array(img_resize, 'f')
    img_resize = img_resize.transpose((2, 0, 1))
    y = model.predictor(np.array([img_resize], 'f'))
    y = F.softmax(y)
    y = y.array
    print(y)
    if np.argmax(y, axis=1)[0] == 0:
        predict = "tu"
    elif np.argmax(y, axis=1)[0] == 1:
        predict = "du"
    elif np.argmax(y, axis=1)[0] == 2:
        predict = "ta"
    elif np.argmax(y, axis=1)[0] == 3:
        predict = "da"
    elif np.argmax(y, axis=1)[0] == 4:
        predict = "la"
    result = {
        "result" : predict
    }
    return jsonify(result)
if __name__ == '__main__':
  application.run(host="0.0.0.0", port=int("5000"),debug=True)
