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

class GenericTrain:
    # PARAMS:
    MODEL_NAME = None
    IM_SIZE = None
    NUM_TRAIN = None
    NUM_IMGS = None
    NUM_VAL = None
    VAL_IMG = None
    NB_EPOCH = None

    # Model settings
    batch_size = 128
    nb_classes = 2
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3

    def __init__(self, model_name=None):
        assert self.IM_SIZE % 2
        if model_name:
            self.MODEL_NAME = model_name
        try:
            self.current_model = self.get_best_model()
        except:
            self.current_model = None

    def quiz(self, num=25):
        (X_train, Y_train, y_train), (X_test, Y_test, y_test) = self.get_data(100, 1, self.NUM_VAL, self.VAL_IMG)
        def ev(a):
            return 'Correct' if a else 'Wrong'
        s = 0
        c = 0

        model = load_model('syn_detection_new3')
        print 'Accuracy %f' % (model.evaluate(X_test, Y_test, verbose=1, show_accuracy=1)[1])
        for n in xrange(num):
            x = X_test[n:n+1]
            comp = model.predict(x.reshape(1,1,self.IM_SIZE,self.IM_SIZE)).argmax()==y_test[n]
            c += comp
            self.show_arr(x)
            you = (1 if raw_input()=='1' else 0)==y_test[n]
            s += you
            print 'You', ev(you), '  Computer', ev(comp)
        print
        print 'You had', s/float(num), 'correct!'
        print 'Computer had', c/float(num), 'correct!'


    @staticmethod
    def show_arr(arr):
        if len(arr.shape)!=2:
            dim = int(round(reduce(lambda a, b: a*b, arr.shape)**0.5))
            arr = arr.reshape(dim, dim)
        toimage(arr.T).show()

    def get_tr(self, num_train, num_imgs):
        pass

    def get_val(self, num_val, val_img):
        pass

    def get_data(self, num_train, num_imgs, num_val, val_img):
        X_train, y_train  = self.get_tr(num_train, num_imgs)
        X_test, y_test = self.get_val(num_val, val_img)

        num = num_train
        X_train = X_train[:num]
        y_train = y_train[:num]


        X_train = X_train.reshape(X_train.shape[0], 1, self.IM_SIZE, self.IM_SIZE)
        X_test = X_test.reshape(X_test.shape[0], 1, self.IM_SIZE, self.IM_SIZE)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        Y_train = np_utils.to_categorical(y_train, self.nb_classes)
        Y_test = np_utils.to_categorical(y_test, self.nb_classes)
        return (X_train, Y_train, y_train), (X_test, Y_test, y_test)


    def get_model(self):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model = Sequential()

        model.add(Convolution2D(self.nb_filters, self.nb_conv, self.nb_conv, border_mode='valid', input_shape=(1, self.IM_SIZE, self.IM_SIZE)))
        model.add(Activation('relu'))
        model.add(Convolution2D(self.nb_filters, self.nb_conv, self.nb_conv))
        model.add(Activation('relu'))


        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.15))

        model.add(Dense(128))
        model.add(Activation('relu'))

        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model

    def get_best_model(self):
        '''Returns model with easy_predict method which returns the probability of the img's centered pixel being this feature'''
        model = load_model(self.MODEL_NAME)
        IM_SIZE = self.IM_SIZE
        def easy_predict(img):
            frame = img.get_frame(None, IM_SIZE)
            return model_predict(model, frame, IM_SIZE)[0][1]  #
        model.easy_predict = easy_predict
        return model

    def train_model(self):
        assert isinstance(self.MODEL_NAME, basestring)
        try: # try to train existing model
            model = load_model(self.MODEL_NAME)
        except: # create new if does not exist
            model = self.get_model()
        (X_train, Y_train, y_train), (X_test, Y_test, y_test) = self.get_data(self.NUM_TRAIN, self.NUM_IMGS, self.NUM_VAL, self.VAL_IMG)
        model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.NB_EPOCH, show_accuracy=True, verbose=1, shuffle=True, validation_data=(X_test, Y_test))
        save_model(model, self.MODEL_NAME)

    def test_accuracy(self):
        model = load_model(self.MODEL_NAME)
        (X_train, Y_train, y_train), (X_test, Y_test, y_test) = self.get_data(100, 1, self.NUM_VAL, self.VAL_IMG)
        accuracy = model.evaluate(X_test, Y_test, verbose=1, show_accuracy=1)[1]
        print 'Accuracy %f' % accuracy
        return accuracy
