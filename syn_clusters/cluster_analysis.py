from sklearn.cluster import DBSCAN
from utils import to_bool_arr
import numpy as np
from PIL import Image
from .cluster import Cluster


class ClusterAnalysis:
    ''' To get results c.image_clusters(areas=True, only_synapses=True)
    '''
    DBSCAN_EPS = 6
    DB_SCAN_MIN_SAMPLES = 4
    MIN_CLUSTER_MEMBERS = 25
    def __init__(self, ves, mem, syn):
        self.ves = to_bool_arr(ves)
        self.mem = to_bool_arr(mem)
        self.syn = to_bool_arr(syn)

        self.psyn = self.syn #& self.mem
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


    def image_clusters(self, show_only_one=None, areas=False, background=None, only_synapses=False):
        '''Background has to be a PIL.Image. Returns Image'''
        m = np.zeros_like(self.psyn)
        for i, c in self.clusters.items():
            if show_only_one is not None and i!=show_only_one:
                continue
            if only_synapses and not c.synapse:
                continue
            c = c.members if not areas else c.area
            m[c[:,0], c[:,1]] = 1
        if not background:
            return Image.fromarray((m*255).astype(np.uint8))
        else:
            background = background.convert('RGB')
            bg = background.load()
            for p in zip(*m.T.nonzero()):
                bg[p] = (255,0,0)
            return background