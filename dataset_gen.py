from easyimg import EasyImage, show_arr
import random
import numpy as np
from sklearn.utils import shuffle


random.seed(1337)

IMG = 'Archive/image.tif'
img = EasyImage(IMG)

MEM = 'Archive/membrane.tif'
mem = EasyImage(MEM)

VES = 'Archive/vesicle.tif'
ves = EasyImage(MEM)

SYN = 'Archive/vesicle.tif'
syn = EasyImage(MEM)

def rpos():
    return (random.randrange(100, 900),random.randrange(100, 900))


class DGen:
    def __init__(self, raw=img, labels=mem):
        self.labels = labels
        self.raw = raw

    def get_pos_labels(self, n):
        self.labels.seek(n)
        self.labels._setcurrent((0, 0))
        res = {1:[],
               0: []}
        for px in self.labels:
            if self.labels[px]>250:
                res[1].append(px)
            else:
                res[0].append(px)
        random.shuffle(res[0])
        random.shuffle(res[1])
        return res

    def get_uni_train(self, num, size=29, frac_true=0.5, ns=10, condition=None):
        '''Gets traning dataset from ns images'''
        vals = [self.get_train(num/ns, size, frac_true, n, condition) for n in xrange(ns)]
        return np.concatenate(tuple(v[0] for v in vals)), np.concatenate(tuple(v[1] for v in vals))


    def get_train(self, num, size=29, frac_true=0.5, n=0, condition=None):
        positive = int(num*frac_true)
        negative = num - positive
        assert (positive and negative)
        labels = self.get_pos_labels(n)
        self.raw.seek(n)
        pX = []
        for label in (0, 1):
            for pos in labels[label]:
                try:
                    if condition and not condition(self.raw, pos):
                        continue
                    pX.append(self.raw.get_frame(pos, size))
                    if len(pX)==negative+label*positive:
                        break
                except:
                    pass
        X = np.asarray(pX)
        y = np.asarray(negative*[0] + positive*[1])
        X, y = shuffle(X, y, random_state=0)
        return X, y







