import pickle
from tensorflow import keras

def LoadDeepfakeModel():
    
    with open("PickledModel/modelWeights.pickle", "rb") as f:
        weights_loadin = pickle.load(f)
    with open("PickledModel/modelConfig.pickle", "rb") as f:
        configuration_loadin = pickle.load(f)
        
    new_model = keras.Sequential.from_config(configuration_loadin)
    new_model.compile(optimizer = 'adam', 
                      metrics = [keras.metrics.BinaryAccuracy()],
                      loss = keras.losses.BinaryCrossentropy(from_logits = True,
                                                             name = 'binary_crossentropy'))
    new_model.set_weights(weights_loadin)
    
    return new_model