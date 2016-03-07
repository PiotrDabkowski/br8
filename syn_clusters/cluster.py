import math

import numpy as np
from scipy.spatial.distance import pdist, squareform

from utils import get_point_spans
from utils.gradient_calc import calc_grad
from .box import Box


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

    MIN_VES_DIF = 17
    MIN_VES_RATIO = 2.2

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

        self.synapse, self.front_vesicles = self._is_a_synapse()

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

    def _is_a_synapse(self):
        '''Returns whether the it's the synapse and whether vesicles are in the front'''
        pc, ac = self.posterior_ves_count, self.anterior_ves_count
        if pc>ac:
            m, s = pc, ac
            front = False
        else: # more vesicles in the anterior
            m, s = ac, pc
            front = True
        #todo implement more sophisticated analysis!
        if m-s>=self.MIN_VES_DIF and m/(s+0.0001)>=self.MIN_VES_RATIO:
            synapse = True
        else:
            synapse = False
        return synapse, front

