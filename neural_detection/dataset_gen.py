import random

import numpy as np
try: # cluster has a problem with this
    from sklearn.utils import shuffle
except:
    pass

from utils.easyimg import EasyImage

THRESHOLD = 200


IMG = 'Archive/image.tif'
img = EasyImage(IMG)


IMGV = 'Archive/imagev.tif'
imgv = EasyImage(IMGV)

MEM = 'Archive/membrane.tif'
mem = EasyImage(MEM)

VES = 'Archive/vesicle.tif'
ves = EasyImage(VES)

SYN = 'Archive/synapse.tif'
syn = EasyImage(SYN)

SYNV = 'Archive/synapsev.tif'
synv = EasyImage(SYNV)

def rpos():
    return (random.randrange(100, 900),random.randrange(100, 900))


OTHERS = {'syn': syn,
          'mem': mem,
          'ves': ves,
          'img': img}

class DGen:
    def __init__(self, raw=img, labels=mem, others=OTHERS):
        self.labels = labels
        self.raw = raw
        self.others = others

    def get_pos_labels(self, n):
        self.labels.seek(n)
        self.labels._setcurrent((0, 0))
        res = {1:[],
               0: []}
        for px in self.labels:
            if self.labels[px]>THRESHOLD:
                res[1].append(px)
            else:
                res[0].append(px)
        random.shuffle(res[0])
        random.shuffle(res[1])
        return res

    def get_uni_train(self, num, size=29, frac_true=0.5, ns=10, condition=None):
        '''Gets traning dataset from ns images'''
        vals = [self.get_train(num/ns, size, frac_true, n, condition) for n in xrange(ns)]
        X, y = np.concatenate(tuple(v[0] for v in vals)), np.concatenate(tuple(v[1] for v in vals))
        X, y = shuffle(X, y, random_state=31)
        return X, y


    def get_train(self, num, size=29, frac_true=0.5, n=0, condition=None):
        positive = int(num*frac_true)
        negative = num - positive
        assert (positive and negative)
        labels = self.get_pos_labels(n)
        self.raw.seek(n)
        for other_img in self.others.values():
            other_img.seek(n)
        pX = []
        lY = []
        for label in (0, 1):
            for pos in labels[label]:
                try:
                    if condition and not condition(self.raw, pos, label, self.others):
                        continue
                    pX.append(self.raw.get_frame(pos, size))
                    lY.append(label)
                    if len(pX)==negative+label*positive:
                        break
                except:
                    pass
            else:
                raise RuntimeError('Could not generate requested number of images!')
        X = np.asarray(pX)
        y = np.asarray(lY)
        X, y = shuffle(X, y, random_state=70)
        return X, y







