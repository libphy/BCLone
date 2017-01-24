import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

df = pd.read_csv('driving_log.csv', header=None, names=['center','right','left','angle','accl','break','speed'])
cpaths=zip(df['center'],df['angle'])
def gen(paths):
    while 1:
        for fp, ang in paths:
            # create numpy arrays of input data
            # and labels, from each line in the file
            im = mpimg.imread(fp)
            im = np.expand_dims(im,axis=0)
            yield((im, ang))

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
model.add(MaxPooling2D((2, 2),dim_ordering='tf'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile('adam', 'mse')

model.fit_generator(gen(cpaths),
        samples_per_epoch=10, nb_epoch=9)

##
#http://stackoverflow.com/questions/38936016/keras-how-are-batches-and-epochs-used-in-fit-generator

X = np.empty((0,160,320,3))
y = []
i=0
for fp, ang in cpaths:
    # create numpy arrays of input data
    # and labels, from each line in the file
    im = mpimg.imread(fp)
    im = np.expand_dims(im,axis=0)
    X = np.concatenate([X,im],axis=0)
    y.append(ang)
    i+=1
    if i%100==0:
        print(i)

# https://carnd-forums.udacity.com/questions/36051083/p3-some-reflection
