from neural_detection import memtrain, syntrain, vtrain
from neural_detection.dataset_gen import imgv, synv, img, syn

from syn_clusters.cluster_analysis import ClusterAnalysis
from utils.easyimg import EasyImage
from utils.transforms import retina
import sys

DEBUG = 0
MEM_COL_THRESH = 160
MEM_GRAD_THRESH = -1
MEM_PROB_TRESH = 0.3  * 256



COUNT = 0


if not DEBUG:
    mem_detector = memtrain.MemTrain().get_best_model()
    syn_detector = syntrain.SynTrain().get_best_model()
    ves_detector = vtrain.VesTrain().get_best_model()

def log_progress(nth=10):
    global COUNT
    COUNT += 1
    if not COUNT%1000:
        print COUNT / 1000
    if COUNT%nth: # if you want to 1 every nth px
        raise

def can_be_a_membrane(img, grad, nth=8):
    if img[0,0]<MEM_COL_THRESH and grad[0, 0]>MEM_GRAD_THRESH:
        log_progress(nth)
        prob_mem = mem_detector.easy_predict(img)
        return int(prob_mem*255)
    else:
        return 0


def can_be_a_synapse(img, grad):
    if can_be_a_membrane(img, grad, 8)>MEM_PROB_TRESH:
        prob_syn = syn_detector.easy_predict(img)
        return int(prob_syn*255)
    else:
        return 0

def can_be_a_vesicle(img, grad):
    log_progress(2)
    prob_ves = ves_detector.easy_predict(img)
    return int(prob_ves*255)


def get_grad(img):
    return img.refactor(retina)

def get_evidence(img, num=None):
    grad = get_grad(img)

    ves = img.merge(grad, can_be_a_vesicle)
    mem = img.merge(grad, can_be_a_membrane)
    syn = img.merge(grad, can_be_a_synapse)
    if num is not None:
        ves.save('Tests/test%d_ves.tif' % num)
        mem.save('Tests/test%d_mem.tif' % num)
        syn.save('Tests/test%d_syn.tif' % num)
    return ves, mem, syn



def load_evidence(num):
    ves = EasyImage('Tests/test%d_ves.tif' % num)
    mem = EasyImage('Tests/test%d_mem.tif' % num)
    syn = EasyImage('Tests/test%d_syn.tif' % num)
    return ves, mem, syn


def save_results(num, ver, cluster_analysis):
    cluster_analysis.image_clusters(areas=True, only_synapses=True).save('Results/res%d.tif' % num)
    ver.save('Results/ver%d.tif' % num)


assert len(sys.argv)==2

NUM = int(sys.argv[1])
print NUM
imgv.seek(NUM)
synv.seek(NUM)


if DEBUG:
    ves, mem, syn = load_evidence(NUM)
else:
    ves, mem, syn = get_evidence(img, NUM)

c = ClusterAnalysis(ves, mem, syn)
save_results(NUM, synv, c)
