from scipy.misc import toimage
import cPickle
import numpy as np

(X_train, y_train), (X_test, y_test)  = cPickle.load(open('mnist2.pkl', 'rb'))
print X_train.shape, y_train.shape


def show_arr(arr):
    if len(arr.shape)!=2:
        dim = int(round(reduce(lambda a, b: a*b, arr.shape)**0.5))
        arr = arr.reshape(dim, dim)
    toimage(arr).show()

def anal(n):
    x = X_test[n:n+1]
    p = model.predict(x)
    print p.argmax()
    show_arr(x.reshape(28,28))


num = 10000
X_train = X_train[:num]
y_train = y_train[:num]


np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Highway
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 2

img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, shuffle=True)


#score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
#print 'Test score:', score[0]
#print 'Test accuracy:', score[1]

from code import InteractiveConsole
InteractiveConsole(globals()).interact()
