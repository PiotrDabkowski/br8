
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

def model_predict(model, x, dim=None, div=1.0):
    if dim is None:
        dim = x.shape[0]
    return model.predict(x.reshape(1, 1, dim, dim)/div)#[1]#.argmax()


def red_mark(img, source, threshold=150):
    source._setcurrent((0,0))
    new = img.convert('RGB')
    newpx = new.load()
    for pixel in source:
        if source[pixel]>threshold:
            newpx[pixel] = (255, 0, 0) # red
    return new
