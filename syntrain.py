from scipy.misc import toimage
from utils import *
import cPickle
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Highway
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
import numpy as np
from dataset_gen import DGen, img, imgv, syn, synv, mem, ves, THRESHOLD
np.random.seed(1337)


# PARAMS:
MODEL_NAME = 'synapse'

IM_SIZE = 77
NUM_TRAIN = 120000
NUM_IMGS = 120
NUM_VAL = 1000
VAL_IMG = 11
assert IM_SIZE % 2
NB_EPOCH = 1

# Model settings
batch_size = 128
nb_classes = 2
img_rows = img_cols = IM_SIZE
nb_filters = 32
nb_pool = 2
nb_conv = 3



def condition(raw, pos, label, others):
    # VERY IMPORTANT - ACCEPT ONLY MEMBRANES !
    others['mem']._setcurrent(pos) # refers to true membrane
    return others['mem'][0,0] > THRESHOLD


#print X_train.shape, y_train.shape
def quiz(num=25):
    (X_train, Y_train, y_train), (X_test, Y_test, y_test) = get_data(100, 1)
    def ev(a):
        return 'Correct' if a else 'Wrong'
    s = 0
    c = 0

    model = load_model('syn_detection_new3')
    print 'Accuracy %f' % (model.evaluate(X_test, Y_test, verbose=1, show_accuracy=1)[1])
    for n in xrange(num):
        x = X_test[n:n+1]
        comp = model.predict(x.reshape(1,1,IM_SIZE,IM_SIZE)).argmax()==y_test[n]
        c += comp
        show_arr(x)
        you = (1 if raw_input()=='1' else 0)==y_test[n]
        s += you
        print 'You', ev(you), '  Computer', ev(comp)
    print
    print 'You had', s/float(num), 'correct!'
    print 'Computer had', c/float(num), 'correct!'



def show_arr(arr):
    if len(arr.shape)!=2:
        dim = int(round(reduce(lambda a, b: a*b, arr.shape)**0.5))
        arr = arr.reshape(dim, dim)
    toimage(arr.T).show()




def get_data(num_train=NUM_TRAIN, num_imgs=NUM_IMGS, num_val=NUM_VAL, val_img=VAL_IMG):
    dg = DGen(img, syn)  # generates training data
    vg = DGen(imgv, synv)  # generates test data

    X_train, y_train  = dg.get_uni_train(num_train, IM_SIZE, ns=num_imgs, condition=condition)
    X_test, y_test = vg.get_train(num_val, IM_SIZE, n=val_img, condition=condition)

    num = num_train
    X_train = X_train[:num]
    y_train = y_train[:num]


    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return (X_train, Y_train, y_train), (X_test, Y_test, y_test)


def get_model():
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
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

    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def train_model(path='new_model'):
    assert isinstance(path, basestring)
    try: # try to train existing model
        model = load_model(path)
    except: # create new if does not exist
        model = get_model()
    (X_train, Y_train, y_train), (X_test, Y_test, y_test) = get_data()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=NB_EPOCH, show_accuracy=True, verbose=1, shuffle=True, validation_data=(X_test, Y_test))
    save_model(model, path)

def test_accuracy(path):
    model = load_model(path)
    (X_train, Y_train, y_train), (X_test, Y_test, y_test) = get_data(100, 1)
    accuracy = model.evaluate(X_test, Y_test, verbose=1, show_accuracy=1)[1]
    print 'Accuracy %f' % accuracy
    return accuracy


