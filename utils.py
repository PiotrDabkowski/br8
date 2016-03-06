
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



