from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import memtrain
import syntrain
import vtrain
from gradient_calc import calc_grad
from easyimg import EasyImage
from transforms import retina, nthresh, img_and, to_proper
from dataset_gen import img, imgv, synv
import numpy as np
from utils import show_arr, to_bool_arr, get_point_spans
import math
import matplotlib.path as mplPath
from utils import point_inside_polygon



DEBUG = 1
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

#
imgv.seek(11)
synv.seek(11)

def get_evidence(img):
    grad = get_grad(img)

    ves = img.merge(grad, can_be_a_vesicle)
    ves.save('test_ves.tif')

    mem = img.merge(grad, can_be_a_membrane)
    mem.save('test_mem.tif')

    syn = img.merge(grad, can_be_a_synapse)
    syn.save('test_syn.tif')
    return ves, mem, syn

class ClusterAnalysis:
    DBSCAN_EPS = 6
    DB_SCAN_MIN_SAMPLES = 4
    MIN_CLUSTER_MEMBERS = 25
    def __init__(self, ves, mem, syn):
        self.ves = to_bool_arr(ves)
        self.mem = to_bool_arr(mem)
        self.syn = to_bool_arr(syn)

        self.psyn = self.syn & self.mem
        self.nsyn = ~self.syn & self.mem
        self.psynpoints = np.asarray(zip(*self.psyn.nonzero()))

        self.db = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DB_SCAN_MIN_SAMPLES)
        self.db.fit(self.psynpoints)

        self.clusters = self._get_clusters()

    def _get_clusters(self):
        clus = {}
        labels = self.db.labels_
        for label in set(labels):
            if label==-1:
                continue
            members = self.psynpoints[labels==label]
            if len(members)<self.MIN_CLUSTER_MEMBERS:
                continue
            clus[len(clus)] = Cluster(members, len(clus), self)
        return clus


    def show_clusters(self, show_only_one=None, areas=False, background=None):
        m = np.zeros_like(self.psyn)
        for i, c in self.clusters.items():
            if show_only_one is not None and i!=show_only_one:
                continue
            c = c.members if not areas else c.area
            m[c[:,0], c[:,1]] = 1
        if not background:
            show_arr(m.T)
        else:
            background = background.convert('RGB')
            bg = background.load()
            for p in zip(*m.T.nonzero()):
                bg[p] = (255,0,0)
            return background




class Cluster:
    #     Forward (anterior box)
    #      |
    #   ---|---  TANGENT (Along membrane) (self.angle is the global angle of the tangent membrane)
    #      |
    # length is the length of tangent membrane
    # centrid - its centre

    FORWARD_FACTOR = 1.2
    TANGENT_FACTOR = 1.1
    MAX_TANGENT = 80
    MAX_FORWARD = 80
    def __init__(self, members, label, parent):
        self.members = members
        self.parent = parent
        self.label = label

        self.angle = calc_grad(members)[0]
        dis = squareform(pdist(self.members))
        self.length, self.extremes =  np.nanmax(dis), np.unravel_index(np.argmax(dis), dis.shape)
        self.centroid = (self.members[self.extremes[0]] + self.members[self.extremes[1]])/2
        self.tangent = np.asarray([math.cos(self.angle), math.sin(self.angle)])
        self.forward =  np.asarray([math.sin(self.angle), -math.cos(self.angle)])

        self.anterior_box, self.posterior_box = self.get_boxes()
        self.anterior_ves_count = self.anterior_box.count_in_box(self.parent.ves)
        self.posterior_ves_count = self.posterior_box.count_in_box(self.parent.ves)
        self.area = self._get_area()

    def _get_area(self):
        xspan, yspan = get_point_spans(self.members)
        points = np.asarray([[x,y] for x in xrange(*xspan) for y in xrange(*yspan)])
        # fuckers this not implement predict method...
        #ehh
        eps = self.parent.DBSCAN_EPS
        min_samples = self.parent.DB_SCAN_MIN_SAMPLES
        area = list(self.members)
        for point in points:  # n2 still fast though
            friends = 0
            for member in self.members:
                tr = point-member
                if np.sqrt(tr.dot(tr)) < eps:
                    friends+=1
                    if friends >= min_samples:
                        area.append(point)
                        break
        return np.asarray(area)




    def _get_box(self, perp):
        width = min(self.TANGENT_FACTOR* self.length, self.MAX_TANGENT)
        deph = min(self.FORWARD_FACTOR * self.length, self.MAX_FORWARD)
        CLOSE_A = self.centroid + width/2.0 * self.tangent
        CLOSE_B = self.centroid - width/2.0 * self.tangent
        FAR_A = CLOSE_A + deph * perp
        FAR_B = CLOSE_B + deph * perp
        return Box((CLOSE_A, CLOSE_B, FAR_B, FAR_A))

    def get_boxes(self):
        anterior = self._get_box(self.forward)
        posterior = self._get_box(-self.forward)
        return anterior, posterior

class Box:
    def __init__(self, points):
        self.points = points
        self.xspan, self.yspan = get_point_spans(points)


    def iter_int_points(self):
        # dumb but easy approach. fast enough when compared to neural network anyway :)
        for x in xrange(*self.xspan):
            for y in xrange(*self.yspan):
                if point_inside_polygon(x, y, self.points):
                    yield (x,y)

    def count_in_box(self, matrix):
        '''Counts number of true entries in matrix that are inside the box'''
        count = 0
        for point in self.iter_int_points():
            if matrix[point]:
                count += 1
        return count


ves = EasyImage('test_ves.tif')
mem = EasyImage('test_mem.tif')
syn = EasyImage('test_syn.tif')
c = ClusterAnalysis(ves, mem, syn)


for num in c.clusters:
    anterior, posterior = c.clusters[num].get_boxes()
    a_count = anterior.count_in_box(c.ves)
    p_count = posterior.count_in_box(c.ves)
    dif = abs(p_count-a_count)
    if dif<20 or dif<2*min(a_count, p_count):
        c.clusters[num].area = np.asarray([[0,0]])
        continue
    print a_count, p_count
    if a_count>p_count:
        c.clusters[num].members = np.asarray(list(anterior.iter_int_points()))
    else:
        c.clusters[num].members = np.asarray(list(posterior.iter_int_points()))

c.show_clusters(areas=True, background=imgv).show()