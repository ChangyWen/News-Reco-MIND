from keras import models

def unfreeze(model):
    """Unfreeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = True
        if isinstance(layer, models.Model):
            unfreeze(layer)
