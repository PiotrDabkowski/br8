from scipy.misc import toimage
from utils import *
import cPickle
import numpy as np
from dataset_gen import DGen
dg = DGen()
(X_train, y_train), (X_test, y_test)  = dg.get_uni_train(10000, 29, ns=10), dg.get_train(100, 29, n=11)

#print X_train.shape, y_train.shape
def quiz(num=25):
    s = 0
    c = 0
    model = load_model('testing_mem_detection')
    for n in xrange(num):
        x = X_test[30+n:30+n+1]
        show_arr(x)
        ans = 1 if raw_input()=='1' else 0
        comp = 1 if model.predict(x).argmax()==y_test[n] else 0
        c += comp
        if y_test[n]==ans:
            print 'You - Correct,   Computer - ', 'Correct' if comp else 'Wrong'
            s += 1
        else:
            print 'You - Wrong,   Computer - ', 'Correct' if comp else 'Wrong'
    print
    print 'You had', s/float(num), 'correct!'
    print 'Computer had', c/float(num), 'correct!'


def show_arr(arr):
    if len(arr.shape)!=2:
        dim = int(round(reduce(lambda a, b: a*b, arr.shape)**0.5))
        arr = arr.reshape(dim, dim)
    toimage(arr).show()

#quiz()
#sds

def anal(n):
    x = X_test[n:n+1]
    p = model.predict(x)
    print p.argmax()
    show_arr(x.reshape(img_cols,img_rows))


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
nb_classes = 2
nb_epoch = 3

img_rows, img_cols = 29, 29
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
print Y_train[:10]
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, shuffle=True, validation_data=(X_test, Y_test))



save_model(model, 'testing_mem_detection')


from code import InteractiveConsole
InteractiveConsole(globals()).interact()