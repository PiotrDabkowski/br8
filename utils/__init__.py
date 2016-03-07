from scipy.misc import toimage
import numpy as np
import scipy as sp

def to_bool_arr(pic, thresh=200):
    return (np.asarray(pic)>thresh).astype(np.bool)*1



def get_point_spans(points):
    points = np.asarray(points)
    xspan = map(int, (points[:,0].min(), points[:,0].max()))
    yspan = map(int, (points[:,1].min(), points[:,1].max()))
    return xspan, yspan

def point_inside_polygon(x,y,poly):
    n = len(poly)
    inside =False
    p1x,p1y = poly[0]
    for i in xrange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

def save_model(model, path):
    json_string = model.to_json()
    with open('Models/%s.json' % path, 'wb') as f:
        f.write(json_string)
    model.save_weights('Models/%s.h5' % path)

def load_model(path):
    from keras.models import model_from_json
    model = model_from_json(open('Models/%s.json' % path, 'rb').read())
    model.load_weights('Models/%s.h5' % path)
    return model

def model_predict(model, x, dim=None):
    if dim is None:
        dim = x.shape[0]
    return model.predict(x.reshape(1, 1, dim, dim))#[1]#.argmax()


def show_arr(arr):
        if len(arr.shape)!=2:
            dim = int(round(reduce(lambda a, b: a*b, arr.shape)**0.5))
            arr = arr.reshape(dim, dim)
        toimage(arr.T).show()
