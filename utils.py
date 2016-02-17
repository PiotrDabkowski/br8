
def save_model(model, path):
    json_string = model.to_json()
    with open('%s.json' % path, 'wb') as f:
        f.write(json_string)
    model.save_weights('%s.h5' % path)

def load_model(path):
    from keras.models import model_from_json
    model = model_from_json(open('%s.json' % path, 'rb').read())
    model.load_weights('%s.h5' % path)
    return model

