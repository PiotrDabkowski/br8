from PIL import Image
from PIL import TiffImagePlugin
import numpy as np
from scipy.misc import toimage

def show_arr(arr):
    if len(arr.shape)!=2:
        dim = int(round(reduce(lambda a, b: a*b, arr.shape)**0.5))
        arr = arr.reshape(dim, dim)
    toimage(arr).show()

class EasyImage(TiffImagePlugin.TiffImageFile):
    def __init__(self, path):
        self.path = path
        TiffImagePlugin.TiffImageFile.__init__(self, path, 'L')
        self.px = self.load()
        self.current_px = 0, 0
        self.x_size, self.y_size = self.size
        self.is_crap = isinstance(self.px[0,0], tuple)


    def __iter__(self):
        return ((x, y) for x in xrange(self.x_size) for y in xrange(self.y_size))


    def refactor(self, func):
        self.px = self.load()
        new = EasyImage(self.path)
        for pixel in self:
            self._setcurrent(pixel)
            try:
                val = func(self)
            except:
                val = 0
            new.px[pixel] = val
        return new

    def merge(self, other, func):
        self.px = self.load()
        new = EasyImage(self.path)
        for pixel in self:
            self._setcurrent(pixel)
            other._setcurrent(pixel)
            try:
                val = func(self, other)
            except:
                val = 0
            new.px[pixel] = val
        return new

    def get_frame(self, cen, dim):
        if cen is not None:
            self._setcurrent(cen)
        #assert dim%2
        d = dim/2
        arr = []
        for x in xrange(-d, d+1):
            row = []
            for y in xrange(-d, d+1):
                row.append(self[x,y])
            arr.append(row)
        return np.asarray(arr)/255.0



    def _setcurrent(self, pixel):
        self.current_px = pixel

    def __getitem__(self, item):
        x, y= item[0] + self.current_px[0], item[1] + self.current_px[1]
        return self.px[x, y] if not self.is_crap else self.px[x,y][0]

    def __setitem__(self, item, value):
        x, y= item[0] + self.current_px[0], item[1] + self.current_px[1]
        self.px[x, y] = value if not self.is_crap else (value, 255)

    def copy_from(self, eim):
        for px in eim:
            self.px[px] = eim.px[px]

    def expand_self(self, rate, func):
        assert rate%2
        offset = rate // 2
        size = rate*self.x_size, rate*self.y_size
        self.resize(size).convert('L').convert('LA').save('tmp.tif')
        new = EasyImage('tmp.tif')
        for pixel in new: # clear the image
            new._setcurrent(pixel)
            new[0,0] = 0
        for pixel in self:
            self._setcurrent(pixel)
            new._setcurrent((pixel[0]*rate+offset, pixel[1]*rate+offset))
            func(new, self[0,0])
        return new
