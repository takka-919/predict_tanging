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
from pydub import AudioSegment
import glob
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import seaborn as sns
import os


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

def make_spectrogram(filename):
    fig,ax = plt.subplots()
    sound = AudioSegment.from_file(filename, "wav")
    samples = np.array(sound.get_array_of_samples())
    sample = samples[::sound.channels]
    spec = np.fft.fft(sample)
    freq = np.fft.fftfreq(sample.shape[0], 1.0/sound.frame_rate)
    freq = freq[:int(freq.shape[0]/2)]
    spec = spec[:int(spec.shape[0]/2)]
    spec[0] = spec[0] / 2

    #plt.xscale("log")
    # plt.yscale("log")
    #窓幅
    w = 1000
    #刻み
    s = 500
    #スペクトル格納用
    ampList = []
    #偏角格納用
    argList = []

    #刻みずつずらしながら窓幅分のデータをフーリエ変換する
    for i in range(int((sample.shape[0]- w) / s)):
        data = sample[i*s:i*s+w]
        spec = np.fft.fft(data)
        spec = spec[:int(spec.shape[0]/2)]
        spec[0] = spec[0] / 2
        ampList.append(np.abs(spec))
        argList.append(np.angle(spec))

    #周波数は共通なので１回だけ計算（縦軸表示に使う）
    freq = np.fft.fftfreq(data.shape[0], 1.0/sound.frame_rate)
    freq = freq[:int(freq.shape[0]/2)]

    #時間も共通なので１回だけ計算（横軸表示に使う）
    time = np.arange(0, i+1, 1) * s / sound.frame_rate

    #numpyの配列にしておく
    ampList = np.array(ampList)
    argList = np.array(argList)

    df_amp = pd.DataFrame(data=ampList, index=time, columns=freq)

    #seabornのheatmapを使う
    plt.figure(figsize=(20, 6))
    sns.heatmap(data=np.log(df_amp.iloc[:, :100].T),
                xticklabels=100,
                yticklabels=10,
                cmap=plt.cm.gist_rainbow_r,
                cbar=False
               )
    plt.tick_params(length=0)
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    path, ext = os.path.splitext( os.path.basename(filename) )
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    plt.savefig('./output/' + path +'.jpg')
    return './output/' + path +'.jpg'

application = Flask(__name__)
application.config['JSON_AS_ASCII'] = False
@application.route('/', methods = ['GET', 'POST'])

def upload_file():
  if request.method == 'GET':
    return render_template('index.html')
  if request.method == 'POST':
    # アプロードされたファイルを保存する
    f = request.files['sound']
    sound_filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
    f.save(sound_filepath)
    img_filepath = make_spectrogram(sound_filepath)
    # モデルを使って判定する
    model = L.Classifier(CNN())
    serializers.load_npz('model_tanging.npz', model)
    image = Image.open(img_filepath)
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
