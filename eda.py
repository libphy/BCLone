import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import os

BATCH_SIZE = 128
cwd = os.getcwd()
df = pd.read_csv(cwd+'/data/driving_log.csv', header=0)#header=None, names=['center','right','left','steering','throttle','brake','speed'])
# the header setup depends on the data file. check if the data already includes header.
## change df file addresses to full addresses

impaths = list(map(lambda x: cwd+'/data/'+x, df['center']))
angles = list(df['steering'])

def gen(paths, angles, batchsz):
    assert len(paths)==len(angles), "Number of features and targets don't match."
    while 1:
        for offset in range(0, len(paths), batchsz):
            #x = range(25)[offset:offset+10]
            batch_X = np.array(map(lambda x: mpimg.imread(x), paths[offset:offset+batchsz]))
            batch_y = np.array(map(lambda x: x, angles[offset:offset+batchsz]))
            yield batch_X, batch_y

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

model.fit_generator(gen(impaths,angles,BATCH_SIZE),samples_per_epoch=len(angles), nb_epoch=5)

##
#http://stackoverflow.com/questions/38936016/keras-how-are-batches-and-epochs-used-in-fit-generator

# https://carnd-forums.udacity.com/questions/36051083/p3-some-reflection

# https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
# intermediate_layer_model = Model(input=model.input,
#                                  output=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(data)

### EDA with dummy model
g = gen(impaths[2000:4000], angles[2000:4000], 128) #take some data in the middle to avoid a lot of zero angles in the beginning
g0 = next(g) #generate data as tuple
X, y = g0
y = y.reshape((128,1))
yp = model.predict(X, batch_size=128) #comparing yp and y by eyes suggest that the model+training sucks.
## Comment: it could be combination of these
## 1) model is too primitive,
## 2) needs data augmentation and preprocessing,
## 3) the ordered data can cause training problem (does randomizing help?),
## 4) there might be a discontinuity in the data if data collection has been pause and resumed, 
## 5) may need speed information,  (the network fusion at the end can help- 3-4 parameters at the end + speed)
