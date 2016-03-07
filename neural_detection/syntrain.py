from .generic_train import GenericTrain
from .dataset_gen import DGen, img, imgv, syn, synv, mem, ves, THRESHOLD

class SynTrain(GenericTrain):
    MODEL_NAME = 'syn_detection_new4'
    IM_SIZE = 77
    NUM_TRAIN = 120000
    NUM_IMGS = 120
    NUM_VAL = 1000
    VAL_IMG = 11
    NB_EPOCH = 1

    @staticmethod
    def condition(raw, pos, label, others):
        # VERY IMPORTANT - ACCEPT ONLY MEMBRANES !
        others['mem']._setcurrent(pos) # refers to true membrane
        return others['mem'][0,0] > THRESHOLD

    @staticmethod
    def val_condition(raw, pos, label, others):  # for verification we should use only membrane pixels
        raw._setcurrent(pos)  # we dont have true membranes but we know mem has to be dark
        return raw[0,0] < 160 # centre pixel of raw image has to be dark


    def get_tr(self, num_train=NUM_TRAIN, num_imgs=NUM_IMGS):
        dg = DGen(img, syn)  # generates training data
        return dg.get_uni_train(num_train, self.IM_SIZE, ns=num_imgs, condition=self.condition)

    def get_val(self, num_val=NUM_VAL, val_img=VAL_IMG):
        vg = DGen(imgv, synv)  # generates test data
        return vg.get_train(num_val, self.IM_SIZE, n=val_img, condition=self.val_condition) # this is a bit crap because we dont have true membranes


if __name__=='__main__':
    s = SynTrain()
    s.test_accuracy()