import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model, load_model
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
import os
import cv2
import json
from collections import Counter
from sklearn.utils import shuffle
from models import commaai, comma2
from sklearn.model_selection import train_test_split
from keras.callbacks import History, ModelCheckpoint, EarlyStopping

def datafetch(datafolder, blacklist):
    cwd = os.getcwd()
    dirnames = os.listdir(cwd+datafolder)
    df = pd.DataFrame(columns=['path','center','right','left','steering','throttle','brake','speed'])
    for sub in dirnames:
        if sub not in blacklist:
            df1 = pd.read_csv(cwd+datafolder+'/'+sub+'/driving_log.csv', header=None, names=['center','right','left','steering','throttle','brake','speed'])
            print(sub, len(df1))
            df1['path'] = list(map(lambda x: cwd+datafolder+'/'+sub+'/IMG/'+x.split('/')[-1], df1['center']))
            df = pd.concat([df,df1], ignore_index=True) #use for user collected data
    X = df['path']
    y = df['steering']
    X, y = shuffle(X,y,random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    #return df
    return X_train, y_train, X_val, y_val

def gen(paths, angles, batchsz): #with flip enabled, it will produce twice many batch size
    assert len(paths)==len(angles), "Number of features and targets don't match."
    while 1:
        for offset in range(0, len(paths), batchsz):
            batch_X = np.array(list(map(lambda x: preprocess(cv2.imread(x)), paths[offset:offset+batchsz])))
            batch_y = np.array(list(map(lambda x: x, angles[offset:offset+batchsz])))
            yield(batch_X, batch_y)

def preprocess(im, ycrop=(50,140), xcrop=(0,320)):
    #by slice- e.g. crop
    return im[ycrop[0]:ycrop[1],xcrop[0]:xcrop[1]]

if __name__=='__main__':
    BATCH_SIZE = 64
    LEARNINGRATE = 1e-3
    DATAPATH = '/data/alldata/racetrack'
    blacklist = ['irratic']
    Xtr, ytr, Xval, yval = datafetch(DATAPATH, blacklist)
    model = comma2(90,320)
    model.compile(optimizer=Adam(lr=LEARNINGRATE), loss="mse")
    history = History()
    callbackslist=[history]
    model.fit_generator(gen(Xtr,ytr, BATCH_SIZE),samples_per_epoch=len(ytr), nb_epoch=10, validation_data = gen(Xval,yval, BATCH_SIZE), nb_val_samples = len(yval), verbose=1, callbacks = callbackslist)
    #model.save('model.h5')
