from generic_train import GenericTrain
from dataset_gen import DGen, img, imgv, syn, synv, mem, ves, THRESHOLD

class MemTrain(GenericTrain):
    MODEL_NAME = 'membrane_clus41_1'

    IM_SIZE = 41
    NUM_TRAIN = 120000
    NUM_IMGS = 120
    NUM_VAL = 1000
    VAL_IMG = 124
    NB_EPOCH = 1


    @staticmethod
    def condition(raw, pos, label, others):
        raw._setcurrent(pos)
        return raw[0,0] < 160 # centre pixel of raw image has to be dark


    def get_tr(self, num_train=NUM_TRAIN, num_imgs=NUM_IMGS):
        dg = DGen(img, mem)  # generates training data
        return dg.get_uni_train(num_train, self.IM_SIZE, ns=num_imgs, condition=self.condition)


    def get_val(self, num_val=NUM_VAL, val_img=VAL_IMG):
        vg = DGen(img, mem)  # generates test data
        return vg.get_train(num_val, self.IM_SIZE, n=val_img, condition=self.condition)


if __name__=='__main__':
    m = MemTrain()
    m.test_accuracy()