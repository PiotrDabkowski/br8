from generic_train import GenericTrain
from dataset_gen import DGen, img, imgv, syn, synv, mem, ves, THRESHOLD
import random

class VesTrain(GenericTrain):
    MODEL_NAME = 'ves_detection1'
    IM_SIZE = 23
    NUM_TRAIN = 100000
    NUM_IMGS = 100
    NUM_VAL = 1000
    VAL_IMG = 120
    NB_EPOCH = 1

    def condition(self, raw, pos, label, others):
        raw._setcurrent(pos)
        if label==0 and random.random()>0.2:
            return self.current_model.easy_predict(raw)>0.5  # we want hard cases
        return True

    def get_tr(self, num_train=NUM_TRAIN, num_imgs=NUM_IMGS):
        dg = DGen(img, ves)  # generates training data
        return dg.get_uni_train(num_train, self.IM_SIZE, ns=num_imgs, condition=self.condition)

    def get_val(self, num_val=NUM_VAL, val_img=VAL_IMG):
        vg = DGen(img, ves)  # generates test data
        return vg.get_train(num_val, self.IM_SIZE, n=val_img, condition=None) # this is a bit crap because we dont have true membranes


if __name__=='__main__':
    v = VesTrain()
    v.train_model()