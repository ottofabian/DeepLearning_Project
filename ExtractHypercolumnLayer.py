from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# contains only some code for investigation
class ExtractHypercolumnLayer(Layer):

    def __init__(self, model, **kwargs):
        super(ExtractHypercolumnLayer, self).__init__(**kwargs)
        self.model = model

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(ExtractHypercolumnLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #return K.dot(x, self.kernel)
        outputs = [layer.output for layer in self.model.layers]
        print(self.model.layers[0].output)
        print(x)
        #print(outputs)
        print("-------------------")
        return self.model.layers[0].output
        
    def compute_output_shape(self, input_shape):
        return self.model.layers[0].output_shape
        
    def update_model(self, model):
        self.model = model
        
    def sample_pixels(self):
        pass