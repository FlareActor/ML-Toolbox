"""combine some keras layers with input mask.
"""
import tensorflow as tf
from keras import backend as K
from keras.utils.conv_utils import conv_output_length
from keras.layers import Convolution1D,GlobalAvgPool1D

class MaskedConvolution1D(Convolution1D):
    def _cal_output_length(self,input_length):
        return conv_output_length(input_length=input_length,
                                  filter_size=self.kernel_size[0],
                                  stride=self.strides[0],
                                  padding=self.padding,
                                  dilation=self.dilation_rate[0])
    
    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            m=K.cast(mask,tf.int32)
            s=K.sum(m,axis=-1,keepdims=True)
            o=K.map_fn(self._cal_output_length,s)
            o=K.one_hot(o,self.compute_output_shape(inputs.shape)[1])
            o=K.squeeze(o,axis=1)
            o=1-K.cumsum(o,axis=-1)
            mask=K.cast(o,tf.bool)
        return mask

class MaskedGlobalAvgPool1D(GlobalAvgPool1D):
    def call(self,x,mask=None):
        if mask is not None:
            m=K.cast(mask,tf.float32)
            m=K.expand_dims(m,-1)
            x=x*m
            return K.sum(x,axis=1)/K.sum(m,axis=1)
        return super(MaskedGlobalAvgPool1D,self).call(x)
        
    def compute_mask(self, inputs, mask=None):
        return None


