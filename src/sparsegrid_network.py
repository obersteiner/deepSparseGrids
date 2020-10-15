import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
import keras.layers as layers
import keras.backend as KB

from typing import List

import sys

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

class ModelBuilder:
    """
    Constructs neural network models with specified topology.
    """
    
    @staticmethod
    def define_encoder_model(n_input_dimensions,
                             n_layers,
                              n_dim_per_layer, name,
                             n_latent_dimensions=None,
                             activation="tanh", dtype=tf.float64,
                              last_activation=None):
        
        if n_latent_dimensions is None:
            n_latent_dimensions = n_input_dimensions
            
        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        network_x = inputs
        for i in range(n_layers):
            network_x = layers.Dense(n_dim_per_layer, activation=activation, dtype=dtype,
                                name=name + "_hidden_{}".format(i))(network_x)
        network_output = layers.Dense(n_latent_dimensions, dtype=dtype,
                                      name=name + "_output", activation=last_activation)(network_x)
            
        network = tf.keras.Model(inputs=inputs, outputs=network_output, name=name + "_forward_model")
        return network
    

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def __log(self, message, flush=True):
        sys.stdout.write(message)
        if flush:
            sys.stdout.flush()
            
    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.__log(
            "\rThe average loss for epoch {} is {:7.2f} ".format(
                epoch, logs["loss"]
            )
        )
        

class SparseGridTest():
    def __init__(self, positions, scales, l2_regularization=1e-5):
        self.size = (positions.shape[0])
        self.l2_regularization = l2_regularization
        self._construct_grid(positions, scales)
    
    @staticmethod
    def hat_function(x, loc=0, scale=1):
        # squared pairwise distances
        pairwise_diff = tf.abs(KB.expand_dims(x, 0) - KB.expand_dims(loc, 1))
        #pairwise_squared_distance = KB.prod(pairwise_diff/scale, axis=-1)
        pairwise_diff /= KB.expand_dims(scale,1)
        #rint(pairwise_diff, x)
        return tf.transpose(KB.prod(keras.backend.clip(1-pairwise_diff, 0, 1), axis=-1))
    
    def _construct_grid(self, positions, scales):
        """
        This could be way more complicated...
        """
        self.positions = positions
        self.scales = scales
        
        # self._kernel = self.kernel(positions)
        # self._kernel_L = np.linalg.cholesky(self._kernel + self.l2_regularization * np.identity(self._kernel.shape[0]))
        # self._kernel_LT = self._kernel_L.T.conj() # now, _kernel = L @ LT
    
    def kernel(self, inputs):
        return self.hat_function(inputs, self.positions, self.scales)
        
    def compute_coefficients(self, _inputs, _outputs):
        _kernel = self.kernel(_inputs)
        
        # solve K x = b through (L @ LT) x = b.
        # _y = tf.linalg.lstsq(self.sparse_grid._kernel_L, outputs, l2_regularizer=self.l2_regularizer)
        # _sg_coeff = tf.linalg.lstsq(self.sparse_grid._kernel_LT, _y, l2_regularizer=self.l2_regularizer)
        
        self._sg_coeff = tf.linalg.lstsq(_kernel, _outputs, l2_regularizer=self.l2_regularization)
        #return tf.linalg.lstsq(_kernel, _outputs, l2_regularizer=self.l2_regularization)
    
    def __call__(self, inputs):
        """
        Compute the grid with given coefficients
        """
        kernel_result = self.kernel(inputs)
        #print(kernel_result, self.positions, self.scales)
        return kernel_result  @ self._sg_coeff
    
    
class CombiGridTest():
    def __init__(self, positions, scales, combi_coefficients, l2_regularization=1e-5):
        self.size = (positions.shape[0])
        self.l2_regularization = l2_regularization
        self._construct_grid(positions, scales)
        self._combi_coefficients = combi_coefficients
    
    @staticmethod
    def hat_function(x, loc=0, scale=1, combi_coefficients=1):
        # squared pairwise distances
        pairwise_diff = tf.abs(KB.expand_dims(x, 0) - KB.expand_dims(loc, 1))
        #pairwise_squared_distance = KB.prod(pairwise_diff/scale, axis=-1)
        pairwise_diff /= KB.expand_dims(scale,1)
        #rint(pairwise_diff, x)
        hat_evaluation = keras.backend.clip(1-pairwise_diff, 0, 1)
        #print(KB.prod(hat_evaluation, axis=-1), KB.expand_dims(combi_coefficients,0))
        return tf.transpose(KB.prod(hat_evaluation, axis=-1)) * combi_coefficients
    
    def _construct_grid(self, positions, scales):
        """
        This could be way more complicated...
        """
        self.positions = positions
        self.scales = scales
        
        # self._kernel = self.kernel(positions)
        # self._kernel_L = np.linalg.cholesky(self._kernel + self.l2_regularization * np.identity(self._kernel.shape[0]))
        # self._kernel_LT = self._kernel_L.T.conj() # now, _kernel = L @ LT
    
    def kernel(self, inputs):
        return self.hat_function(inputs, self.positions, self.scales, self._combi_coefficients)
        
    def compute_coefficients(self, _inputs, _outputs):
        _kernel = self.kernel(_inputs)
        
        # solve K x = b through (L @ LT) x = b.
        # _y = tf.linalg.lstsq(self.sparse_grid._kernel_L, outputs, l2_regularizer=self.l2_regularizer)
        # _sg_coeff = tf.linalg.lstsq(self.sparse_grid._kernel_LT, _y, l2_regularizer=self.l2_regularizer)
        
        self._sg_coeff = tf.linalg.lstsq(_kernel, _outputs, l2_regularizer=self.l2_regularization)
        #return tf.linalg.lstsq(_kernel, _outputs, l2_regularizer=self.l2_regularization)
    
    def __call__(self, inputs):
        """
        Compute the grid with given coefficients
        """
        kernel_result = self.kernel(inputs)
        #print(kernel_result, self.positions, self.scales)
        return kernel_result  @ self._sg_coeff
    
class SparseGridNetwork(keras.models.Model):
    """
    A neural network with a sparse grid at the end.
    """
    
    def __init__(self,
                 encoder: tf.keras.Model,
                 sparse_grids,
                 coefficients,
                 dim_data_in: int,
                 dim_data_out: int,
                 l2_regularizer = 1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.sparse_grids = sparse_grids
        self.dim_data_in = dim_data_in
        self.dim_data_out = dim_data_out
        self.l2_regularizer = l2_regularizer
        self.data_scaler = lambda x: x
        self.coefficients = coefficients
    
    def compute_data_scaler(self, data_in):
        _d_max = np.max(data_in)
        _d_min = np.min(data_in)
        self.data_scaler = lambda x: 2 * (x-_d_min) / (_d_max - _d_min) - 1
        
    def call_forward_pass(self, _inputs):
        _inputs_scaled = self.data_scaler(_inputs)
        _encoded = self.encoder(_inputs_scaled)
        return self.sparse_grids[0](_encoded)
    
    def call(self, _inputs):
        """
        This is only used in training. The "_inputs" have to be a combination of [input,output].
        """
        _data_in, _data_out = tf.split(_inputs, num_or_size_splits=[self.dim_data_in, self.dim_data_out], axis=1)
        _data_in_scaled = self.data_scaler(_data_in)
        
        _encoded = self.encoder(_data_in_scaled)
        #print(_encoded)
        sparse_grid = self.sparse_grids[0]
        coefficient = self.coefficients[0]
        sparse_grid.compute_coefficients(_encoded, _data_out)
        _sg_encoded = sparse_grid(_encoded)# * coefficient
        #print(_sg_encoded)
        length = len(self.sparse_grids)
        for k in range(1,length):
            sparse_grid = self.sparse_grids[k]
            coefficient = self.coefficients[k]
            sparse_grid.compute_coefficients(_encoded, _data_out)
            _sc_encoded = sparse_grid(_encoded)# * coefficient
            #print(_sg_encoded, _sg_coefficients, coefficient,sparse_grid(_encoded))
        
        loss = tf.square(_sg_encoded - _data_out)
        
        self.add_loss(loss)
        self.add_metric(loss, name="distortion", aggregation="mean")
        
        return _sg_encoded
        
    def train(self, data_in, data_out, **kwargs):
        
        data_full = np.column_stack([data_in, data_out])
        
        hist = self.fit(x=data_full.astype(np.float64),
                        verbose=0,
                        callbacks=[LossAndErrorPrintingCallback()],
                        **kwargs)
        return hist