from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam

def commaai(H,W):
### Comma_AI
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
            input_shape=(H, W, 3),
            output_shape=(H, W, 3))) #change the dimension when croping images (original size is 160x320)
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model

def comma2(H,W):
### Comma_AI
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
            input_shape=(H, W, 3),
            output_shape=(H, W, 3))) #change the dimension when croping images (original size is 160x320)
    model.add(Convolution2D(32, 3, 3, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    # model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    # model.add(ELU())
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model
