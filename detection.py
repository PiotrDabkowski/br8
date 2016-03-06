import memtrain
import syntrain
from transforms import retina
from dataset_gen import img, imgv, synv

DEBUG = 0
MEM_COL_THRESH = 160
MEM_GRAD_THRESH = -1
COUNT = 0
MEM_PROB_TRESH = 0.3  * 256      # 30%

if not DEBUG:
    mem_detector = memtrain.MemTrain().get_best_model()
    syn_detector = syntrain.SynTrain().get_best_model()

def log_progress():
    global COUNT
    COUNT += 1
    if not COUNT%1000:
        print COUNT / 1000
    if COUNT%10: # if you want to 1 every nth px
        raise

def can_be_a_membrane(img, grad):
    if img[0,0]<MEM_COL_THRESH and grad[0, 0]>MEM_GRAD_THRESH:
        log_progress()
        prob_mem = mem_detector.easy_predict(img)
        return int(prob_mem*255)
    else:
        return 0


def can_be_a_synapse(img, grad):
    if can_be_a_membrane(img, grad)>MEM_PROB_TRESH:
        prob_syn = syn_detector.easy_predict(img)
        return int(prob_syn*255)
    else:
        return 0

def get_grad(img):
    return img.refactor(retina)

#grad = get_grad(imgv)
imgv.seek(11)
synv.seek(11)
imgv.red_mark(synv).show()


im = imgv.merge(img, can_be_a_synapse)
im.show()
im.save('test11.tif')